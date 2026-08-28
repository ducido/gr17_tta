# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs.
        # NOTE: The action head's OOI/sensitivity loss differentiates through the frozen
        # ViT+LLM (grad of the action w.r.t. the raw vision patch embeddings) with
        # create_graph=True, which requires DOUBLE-backward through attention. The
        # flash / mem-efficient SDPA kernels do not implement a second derivative, so we
        # force the plain `sdpa` implementation and later run the training forward under
        # the MATH backend (see `forward`). Sequences here are short (~256 vision / ~156
        # text tokens), so the throughput cost of dropping flash-attention is small.
        extra_kwargs = {"attn_implementation": "sdpa"}
        if use_flash_attention:
            logger.warning(
                "use_flash_attention=True, but the OOI sensitivity loss needs double-backward "
                "through the backbone; forcing attn_implementation='sdpa' (flash_attention_2 has "
                "no second derivative)."
            )
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)

        # Handle used by the action head's OOI loss: captures the "first" vision
        # embedding (input to the first ViT transformer block, i.e. patch-embed +
        # positional embed) as a differentiable leaf so we can compute the gradient of
        # the predicted action w.r.t. each raw, spatially-localized vision patch.
        self._vision_patch_embeds = None
        # When True, the patch capture + MATH backend also run outside training mode
        # (e.g. Grad-CAM / test-time steering at inference). Defaults off so plain
        # inference is unaffected.
        self._force_capture_vision_patch = False
        self.model.visual.blocks[0].register_forward_pre_hook(self._capture_vision_patch_hook)

        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def _capture_vision_patch_hook(self, module, args):
        """forward_pre_hook on the first ViT block.

        Replaces the block's input (patch-embed + positional embedding, shape
        ``(total_patches, vit_hidden)``) with a detached leaf that requires grad, and
        stashes it on ``self._vision_patch_embeds``. The action head then computes
        ``grad(action, vision_patch_embeds)`` for a spatially-faithful saliency signal.

        Only active during training (grad is needed); a no-op otherwise so inference is
        unaffected.
        """
        if not torch.is_grad_enabled() or not (self.training or self._force_capture_vision_patch):
            return None
        hidden_states = args[0]
        leaf = hidden_states.detach().requires_grad_(True)
        self._vision_patch_embeds = leaf
        # One-shot debug: confirm the pre-hook fires and captures the first-block input.
        _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if _rank == 0 and not getattr(self, "_patch_hook_dbg_printed", False):
            self._patch_hook_dbg_printed = True
            print(
                f"[OOI DEBUG] visual.blocks[0] pre-hook fired: captured leaf "
                f"shape={tuple(leaf.shape)} dtype={leaf.dtype} "
                f"requires_grad={leaf.requires_grad} is_leaf={leaf.is_leaf}"
            )
        return (leaf,) + args[1:]

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}

        self._vision_patch_embeds = None
        if (self.training or self._force_capture_vision_patch) and torch.is_grad_enabled():
            # The OOI loss differentiates action -> vision patch embeddings with
            # create_graph=True, so every attention op on that path (whole ViT + LLM)
            # must be twice-differentiable. Force the MATH SDPA backend here (the sdpa
            # attn_implementation set at load time routes through F.sdpa, which honors
            # this context). flash / mem-efficient kernels have no second derivative.
            if getattr(self.model, "is_gradient_checkpointing", False):
                raise RuntimeError(
                    "Backbone gradient checkpointing is enabled, which is incompatible with the "
                    "double-backward required by the OOI sensitivity loss. Disable gradient "
                    "checkpointing for the backbone."
                )
            with sdpa_kernel([SDPBackend.MATH]):
                outputs = self.model(**vl_input, output_hidden_states=True)
        else:
            outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
                # First-block vision patch embeddings (differentiable leaf), or None
                # outside training. Consumed by the action head OOI loss.
                "vision_patch_embeds": self._vision_patch_embeds,
            }
        )  # [B, T2, hidden_size]
