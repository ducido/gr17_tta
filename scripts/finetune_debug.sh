
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

CP_DIR=CP/GR00T-N1.7-3B
DATA_DIR=CP/merged_libero_mask_depth_noops_lerobot_10


NUM_GPUS=1
MAX_STEPS=2
GLOBAL_BATCH_SIZE=10
SAVE_STEPS=1000

NUM_GPUS=$NUM_GPUS MAX_STEPS=$MAX_STEPS GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE SAVE_STEPS=$SAVE_STEPS USE_WANDB=0 DATALOADER_NUM_WORKERS=0 bash examples/finetune.sh \
    --base-model-path $CP_DIR \
    --dataset-path $DATA_DIR \
    --embodiment-tag LIBERO_PANDA \
    --output-dir ./outputs/debug \
    --state-dropout-prob 0.2