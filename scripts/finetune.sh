
export CUDA_VISIBLE_DEVICES=0

CP_DIR=CP/GR00T-N1.7-3B
DATA_DIR=CP/libero_10_no_noops_1.0.0_lerobot


NUM_GPUS=1 MAX_STEPS=2 GLOBAL_BATCH_SIZE=160 SAVE_STEPS=500 bash examples/finetune.sh \
    --base-model-path $CP_DIR \
    --dataset-path $DATA_DIR \
    --embodiment-tag LIBERO_PANDA \
    --output-dir ./outputs/libero_10 \
    --state-dropout-prob 0.2