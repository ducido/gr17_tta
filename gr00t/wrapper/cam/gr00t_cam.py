from pathlib import Path
from typing import Any
import copy

import numpy as np
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from gr00t.policy.gr00t_policy import *
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from transformers.feature_extraction_utils import BatchFeature
# from gr00t.data.types import MessageType, ModalityConfig, VLAStepData



class Gr00tCamPolicy(Gr00tPolicy):

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        self.model.action_head.reset()
        return {}

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations

        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        # with torch.inference_mode():
        collated_inputs.update(options=options)
        model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].detach().float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }

        # # convert tensor to numpy
        # for key, value in model_pred['cam_data'].items():
        #     model_pred['cam_data'][key] = value.cpu().numpy()
        
        return casted_action, {'cam_data': model_pred['cam_data'], 'input_ids': model_pred['input_ids']}


class Gr00tN1d7_CAM(Gr00tN1d7):

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        # print("=" * 80)
        # print("BACKBONE")
        # print("=" * 80)

        # total = sum(p.numel() for p in self.backbone.parameters())
        # trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        # print(f"total params     : {total:,}")
        # print(f"trainable params : {trainable:,}")
        # print(f"is trainable     : {trainable > 0}")

        # print()

        # print("=" * 80)
        # print("ACTION HEAD")
        # print("=" * 80)

        # total = sum(p.numel() for p in self.action_head.parameters())
        # trainable = sum(p.numel() for p in self.action_head.parameters() if p.requires_grad)

        # print(f"total params     : {total:,}")
        # print(f"trainable params : {trainable:,}")
        # print(f"is trainable     : {trainable > 0}")
        # breakpoint()

        # backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        # action_head_trainable = any(p.requires_grad for p in self.action_head.parameters())
        # assert not backbone_trainable, "Backbone should be frozen"
        # assert action_head_trainable, "Action head should be trainable"

        ooi_inputs = inputs['ooi_inputs']
        
        del inputs['ooi_inputs']

        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Grad-CAM differentiates the action w.r.t. the raw vision patch embeddings, which
        # are captured inside the (frozen, eval-mode) backbone. Enable the capture here so
        # the pre-hook fires at inference too, and make sure grad is enabled so the graph
        # from patch embeddings -> backbone_features survives (do NOT wrap in no_grad /
        # inference_mode).
        self.backbone._force_capture_vision_patch = True
        with torch.enable_grad():
            backbone_outputs = self.backbone(backbone_inputs)

        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, ooi_inputs=ooi_inputs, input_ids=backbone_inputs['input_ids'], options=options)
        
        action_outputs.update(input_ids=backbone_inputs['input_ids'].detach().cpu().numpy())


        return action_outputs


class Gr00tN1d7_CAM_ActionHead(Gr00tN1d7ActionHead):
    def init_optimizer_and_state(self):
        self.steering_lr = 1e-4

        self.steering_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.steering_lr,
        )

        self._initial_action_head_state = copy.deepcopy(
            self.state_dict()
        )
        print(self.state_dict().keys())

    def reset(self):
        self.load_state_dict(
            self._initial_action_head_state
        )

        self.zero_grad(set_to_none=True)

        self.steering_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.steering_lr,
        )
        print("=================== RESET ACTION HEAD ===================")

    def denoising_step(
        self,
        actions: torch.Tensor,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        ooi_inputs: dict[str, torch.Tensor] | None = None,
        input_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ):
        # Keep vl_embeds CONNECTED to the vision patch embeddings (do not detach / re-leaf),
        # so the action gradient can flow all the way back to the raw, spatially-localized
        # patch tokens. The differentiable leaf itself is the first-ViT-block input captured
        # by the backbone hook.
        vl_embeds = backbone_features
        vision_patch_embeds = backbone_output["vision_patch_embeds"]
        assert vision_patch_embeds is not None, (
            "vision_patch_embeds is None; set backbone._force_capture_vision_patch=True and "
            "run the backbone with grad enabled before Grad-CAM (see Gr00tN1d7_CAM.get_action)."
        )

        batch_size = backbone_features.shape[0]
        device = backbone_features.device

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        all_cam_data = []
        total_loss = 0
        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward. Force the MATH SDPA backend: the Grad-CAM grad uses
            # create_graph=True and (in the steering branch) is backpropped through, which
            # needs double-backward through the DiT attention. flash / mem-efficient have
            # no second derivative.
            with sdpa_kernel(SDPBackend.MATH):
                if self.config.use_alternate_vl_dit:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    )
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                    )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Grad-CAM: differentiate the action w.r.t. the RAW vision patch embeddings
            # (before any ViT block), so each unit maps 1:1 to a spatial image patch.
            action_mask = action_input.action_mask
            target = (pred_velocity * action_mask).norm()
            grads = torch.autograd.grad(
                outputs=target,
                inputs=vision_patch_embeds,  # (total_patches, vit_hidden)
                retain_graph=True,
                create_graph=True,
            )[0]  # (total_patches, vit_hidden)

            sensitivity = grads.norm(dim=-1)  # (total_patches,)
            token_importance = torch.relu((grads * vision_patch_embeds).sum(-1))  # (total_patches,)

            def process_ooi(ooi_inputs):
                new_ooi_inputs = {}
                for k, v in ooi_inputs.items():
                    # v = v.squeeze()[...,0] # B,256,256
                    v = v.squeeze(dim=1)[...,0]

                    v = 1-v # don't know why eval they reverse it so we have to reverse back

                    #### for visualize to debug
                    # debug_dir = Path("/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/gr17_tta/debug")
                    # debug_dir.mkdir(parents=True, exist_ok=True)
                    # vis_v = v.detach().cpu().numpy().astype(np.uint8)
                    # for i in range(vis_v.shape[0]):
                    #     vis_im = Image.fromarray(vis_v[i])
                    #     vis_im.save(debug_dir / f"debug_{k}_{i}.png")
                    ####

                    v = v.unsqueeze(dim=1).float() / 255.0
                    v = torch.nn.functional.adaptive_avg_pool2d(v, (16, 16))
                    new_ooi_inputs[k] = v.view(v.shape[0], -1)
                return new_ooi_inputs


            new_ooi_inputs = process_ooi(ooi_inputs)

            def cal_loss(current_behavior, target_behavior):
                # current_behavior: (B, n) masked image tokens for a single view
                # target_behavior: (B, n) e.g. (B, 64)
                assert current_behavior.shape == target_behavior.shape, f"Shape mismatch: {current_behavior.shape} vs {target_behavior.shape}"

                pred = current_behavior.clamp_min(1e-8)
                target = target_behavior.clamp_min(1e-8)
                pred = pred / (pred.sum(dim=-1, keepdim=True) + 1e-8)
                target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
                loss = torch.nn.functional.kl_div(pred.log(), target, reduction="batchmean")
                return loss

            image_ooi = new_ooi_inputs['image_ooi'].to(dtype=token_importance.dtype, device=token_importance.device) # (B,256)
            wrist_image_ooi = new_ooi_inputs['wrist_image_ooi'].to(dtype=token_importance.dtype, device=token_importance.device) # (B,256)
            merged_ooi = torch.cat([image_ooi, wrist_image_ooi], dim=-1)

            # Optimize each view independently (separate normalization + KL per view)
            token_importance = token_importance.view(merged_ooi.shape[0], merged_ooi.shape[1])
            sensitivity = sensitivity.view(merged_ooi.shape[0], merged_ooi.shape[1])
            ti_loss = cal_loss(token_importance, merged_ooi)
            sen_loss = cal_loss(sensitivity, merged_ooi)
            
            total_loss = ti_loss


            # sensitivity, token_importance = create_dummy_heatmaps(
            #     input_ids=input_ids,
            #     device=vl_embeds.device,
            #     image_token_id=151655,
            # )
            # sensitivity, token_importance = create_heatmaps_by_masks(
            #     input_ids=input_ids,
            #     device=vl_embeds.device,
            #     image_token_id=151655,
            #     options=new_options,
            # )
            cam_record = {
                "denoise_step": t,
                "t_discretized": int(t_discretized),

                # [149]
                "sensitivity": merged_ooi[0].detach().cpu().float().numpy(),

                # [149]
                "token_importance": merged_ooi[0].detach().cpu().float().numpy(),

                # optional
                "pred_velocity": pred_velocity[0].detach().cpu().float().numpy(),
            }

            all_cam_data.append(cam_record)

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength
        return total_loss, actions, vl_embeds, state_features, all_cam_data
        
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        ooi_inputs: dict[str, torch.Tensor] | None = None,
        input_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        # Set initial actions as the sampled noise.
        batch_size = backbone_features.shape[0]
        device = backbone_features.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=backbone_features.dtype,
            device=device,
        )
        # NOTE: backbone_features stays CONNECTED (not detached) to the vision patch
        # embeddings, so grad(action, patch_embeds) can flow through the frozen ViT+LLM.
        # Only actions / state_features are detached (they are not grad targets).
        if options['num_step_tt_in_traj'] <= 0:
            total_loss, actions, vl_embeds, state_features, all_cam_data = self.denoising_step(
                actions.detach(),
                backbone_features,
                state_features.detach(),
                embodiment_id,
                backbone_output,
                action_input,
                ooi_inputs,
                input_ids,
                options,
            )
            # print(f"num_step_tt_in_traj {options['num_step_tt_in_traj']}: loss = {total_loss.item()}")
        else:
            for i in range(options['tt_update']):
                total_loss, actions, vl_embeds, state_features, all_cam_data = self.denoising_step(
                    actions.detach(),
                    backbone_features,
                    state_features.detach(),
                    embodiment_id,
                    backbone_output,
                    action_input,
                    ooi_inputs,
                    input_ids,
                    options,
                )
                total_loss = total_loss.mean()
                print(f"Step {i}: loss = {total_loss.item()}")
                self.steering_optimizer.zero_grad()
                # retain_graph: the frozen backbone graph (patch_embeds -> backbone_features)
                # is shared across steering iterations and must survive each backward.
                total_loss.backward(retain_graph=True)
                self.steering_optimizer.step()
                # for n, p in self.model.named_parameters():
                #     if p.grad is not None:
                #         print(n, p.grad.norm().item())

        return BatchFeature(
            data={
                "action_pred": actions.detach(),
                "backbone_features": vl_embeds.detach(),
                "state_features": state_features.detach(),
                "cam_data": all_cam_data,
            }
        )

    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        ooi_inputs: dict[str, torch.Tensor] | None = None,
        input_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """

        # process_backbone_output (vlln + vl_self_attention) inside _encode_features is on
        # the patch-embed -> action path; force MATH SDPA so its attention is twice-
        # differentiable for the Grad-CAM / steering double-backward.
        with sdpa_kernel(SDPBackend.MATH):
            features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            ooi_inputs=ooi_inputs,
            input_ids=input_ids,
            options=options,
        )


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x

def create_dummy_heatmaps(
    input_ids,
    device,
    image_token_id,
):
    """
    Create dummy heatmaps for visualization debugging.

    Args:
        input_ids: [B, N]
        device: torch device
        image_token_id: IMAGE_TOKEN_ID

    Returns:
        sensitivity: [B, N]
        token_importance: [B, N]
    """

    batch_size, seq_len = input_ids.shape

    sensitivity = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.float32,
        device=device,
    )

    token_importance = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.float32,
        device=device,
    )

    image_mask = input_ids[0] == image_token_id

    num_img_tokens = int(image_mask.sum().item())

    assert num_img_tokens == 128, (
        f"Expected 128 image tokens, got {num_img_tokens}"
    )

    tokens_per_view = num_img_tokens // 2

    grid_size = int(np.sqrt(tokens_per_view))

    assert grid_size * grid_size == tokens_per_view, (
        f"tokens_per_view={tokens_per_view} is not square"
    )

    # ------------------------------------------------------------------
    # Sensitivity: large '+' pattern
    # ------------------------------------------------------------------
    front_sensitivity = torch.zeros(
        (grid_size, grid_size),
        dtype=torch.float32,
        device=device,
    )

    wrist_sensitivity = torch.zeros(
        (grid_size, grid_size),
        dtype=torch.float32,
        device=device,
    )

    center = grid_size // 2

    # front_sensitivity[:, center - 1:center + 1] = 1.0
    # front_sensitivity[center - 1:center + 1, :] = 1.0
    front_sensitivity[0,0] = 1.0

    # wrist_sensitivity[:, center - 1:center + 1] = 1.0
    # wrist_sensitivity[center - 1:center + 1, :] = 1.0
    wrist_sensitivity[0,-1] = 1.0

    # ------------------------------------------------------------------
    # Token importance: large 'X' pattern
    # ------------------------------------------------------------------
    front_importance = torch.zeros(
        (grid_size, grid_size),
        dtype=torch.float32,
        device=device,
    )

    wrist_importance = torch.zeros(
        (grid_size, grid_size),
        dtype=torch.float32,
        device=device,
    )

    # for i in range(grid_size):
    #     front_importance[i, i] = 1.0
    #     front_importance[i, grid_size - 1 - i] = 1.0

    #     wrist_importance[i, i] = 1.0
    #     wrist_importance[i, grid_size - 1 - i] = 1.0
    front_importance[-1,0] = 1.0
    wrist_importance[-1,-1] = 1.0

    sensitivity_img = torch.cat(
        [
            front_sensitivity.reshape(-1),
            wrist_sensitivity.reshape(-1),
        ],
        dim=0,
    )

    token_importance_img = torch.cat(
        [
            front_importance.reshape(-1),
            wrist_importance.reshape(-1),
        ],
        dim=0,
    )

    sensitivity[:, image_mask] = sensitivity_img.unsqueeze(0)

    token_importance[:, image_mask] = (
        token_importance_img.unsqueeze(0)
    )

    return sensitivity, token_importance


def create_heatmaps_by_masks(
    input_ids,
    device,
    image_token_id,
    options,
):
    """
    Create dummy heatmaps for visualization debugging.

    Args:
        input_ids: [B, N]
        device: torch device
        image_token_id: IMAGE_TOKEN_ID

    Returns:
        sensitivity: [B, N]
        token_importance: [B, N]
    """

    batch_size, seq_len = input_ids.shape

    sensitivity = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.float32,
        device=device,
    )

    token_importance = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.float32,
        device=device,
    )

    image_mask = input_ids[0] == image_token_id

    num_img_tokens = int(image_mask.sum().item())

    assert num_img_tokens == 128, (
        f"Expected 128 image tokens, got {num_img_tokens}"
    )

    tokens_per_view = num_img_tokens // 2

    grid_size = int(np.sqrt(tokens_per_view))

    assert grid_size * grid_size == tokens_per_view, (
        f"tokens_per_view={tokens_per_view} is not square"
    )

    sensitivity_img = torch.cat(
        [
            options['image_ooi'],
            options['wrist_image_ooi'],
        ],
        dim=1,
    )

    token_importance_img = torch.cat(
        [
            options['image_ooi'],
            options['wrist_image_ooi'],
        ],
        dim=1,
    )

    sensitivity[:, image_mask] = sensitivity_img.to(dtype=sensitivity.dtype, device=sensitivity.device)
    token_importance[:, image_mask] = token_importance_img.to(dtype=token_importance.dtype, device=token_importance.device)

    return sensitivity, token_importance