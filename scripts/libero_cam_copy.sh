
export CUDA_VISIBLE_DEVICES=0
# module load gcc/13.2.0
# module load ffmpeg/7.0.2
export PYOPENGL_PLATFORM=egl

TASKS=(
    libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
    libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
    libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
    libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
    libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
    libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
    libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
    libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
    libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
    libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
)

action_horizon=8
EPISODES=20
N_envs=1
max_episode_steps=720
PORT=$1

tt_update=0
num_step_tt_in_traj=0

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/libero_10/senti_alpha1.2_20k_tt_update${tt_update}_num_step_tt_in_traj${num_step_tt_in_traj}_${max_episode_steps}steps_eps${EPISODES}_ah${action_horizon}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy_vis_cam.py \
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

