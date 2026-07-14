
export CUDA_VISIBLE_DEVICES=0
# module load gcc/413.2.0
# module load ffmpeg/7.0.2
suite=$1

PYTHON=gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python

CATEGORY="Background Textures"
# Suites to pull from; leave empty to take every suite in task_classification.json.
SUITES=($suite)

mapfile -t TASKS < <("$PYTHON" scripts/plus/list_tasks.py "$CATEGORY" "${SUITES[@]}") || exit 1

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No tasks found for category '$CATEGORY'" >&2
    exit 1
fi
echo "Loaded ${#TASKS[@]} tasks for category '$CATEGORY'"


action_horizon=8
EPISODES=1
N_envs=1
max_episode_steps=720
PORT=$1


for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/libero_plus/background/${suite}/baseline_20k_${max_episode_steps}steps_eps${EPISODES}_ah${action_horizon}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python gr00t/eval/rollout_policy_libero_plus.py \
        --n_episodes $EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port $PORT \
        --max_episode_steps=$max_episode_steps \
        --env_name "$TASK" \
        --n_action_steps $action_horizon \
        --n_envs $N_envs \
        --video_dir "$VIDEO_DIR" \
        > "$LOG_DIR/${NAME}.txt" 2>&1

    # gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    #     --algo cam \
    #     --model_path CP/GR00T-N1.7-LIBERO/libero_10 \
    #     --n_episodes $EPISODES \
    #     --max_episode_steps=$max_episode_steps \
    #     --env_name "$TASK" \
    #     --n_action_steps $action_horizon \
    #     --n_envs $N_envs \
    #     --video_dir "$VIDEO_DIR" # \
    #     #> "$LOG_DIR/${NAME}.txt" 2>&1

    echo "Finished task: $TASK"
    echo ""
done

