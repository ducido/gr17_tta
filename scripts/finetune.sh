
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
CP_DIR=CP/GR00T-N1.7-3B
DATA_DIR=CP/merged_libero_mask_depth_noops_lerobot_10


NUM_GPUS=1
MAX_STEPS=20000
GLOBAL_BATCH_SIZE=100
SAVE_STEPS=500
DATE=$(date +%Y%m%d_%H%M%S)

NUM_GPUS=$NUM_GPUS MAX_STEPS=$MAX_STEPS GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE SAVE_STEPS=$SAVE_STEPS bash examples/finetune.sh \
    --base-model-path $CP_DIR \
    --dataset-path $DATA_DIR \
    --embodiment-tag LIBERO_PANDA \
    --output-dir ./outputs/${DATE}_senti_merged_libero_${NUM_GPUS}gpus_${MAX_STEPS}steps_${GLOBAL_BATCH_SIZE}bs_${SAVE_STEPS}ss \
    --state-dropout-prob 0.2