
export CUDA_VISIBLE_DEVICES=0
# module load gcc/413.2.0
# module load ffmpeg/7.0.2

PYTHON=gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python

CATEGORY="Sensor Noise"
# Suites to pull from; leave empty to take every suite in task_classification.json.
SUITES=(libero_10)

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

tt_update=0
num_step_tt_in_traj=0

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/libero_plus/noise/sen_20k_tt_update${tt_update}_num_step_tt_in_traj${num_step_tt_in_traj}_${max_episode_steps}steps_eps${EPISODES}_ah${action_horizon}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python gr00t/eval/rollout_policy_vis_cam_libero_plus.py \
        --algo cam \
        --save_cam_video_dir $LOG_DIR/grad_cam_action_mask \
        --tt_update $tt_update \
        --num_step_tt_in_traj $num_step_tt_in_traj \
        --n_episodes $EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port $PORT \
        --max_episode_steps=$max_episode_steps \
        --env_name "$TASK" \
        --n_action_steps $action_horizon \
        --n_envs $N_envs \
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

