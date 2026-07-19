
export CUDA_VISIBLE_DEVICES=0
# module load gcc/13.2.0
# module load ffmpeg/7.0.2

suite=$1


if [ "$suite" == "libero_10" ]; then
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
elif [ "$suite" == "libero_goal" ]; then
    TASKS=(
        libero_sim/open_the_middle_drawer_of_the_cabinet
        libero_sim/put_the_bowl_on_the_stove
        libero_sim/put_the_wine_bottle_on_top_of_the_cabinet
        libero_sim/open_the_top_drawer_and_put_the_bowl_inside
        libero_sim/put_the_bowl_on_top_of_the_cabinet
        libero_sim/push_the_plate_to_the_front_of_the_stove
        libero_sim/put_the_cream_cheese_in_the_bowl
        libero_sim/turn_on_the_stove
        libero_sim/put_the_bowl_on_the_plate
        libero_sim/put_the_wine_bottle_on_the_rack
    )
elif [ "$suite" == "libero_object" ]; then
    TASKS=(
        libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket
        libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket
        libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket
        libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket
        libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket
        libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket
        libero_sim/pick_up_the_butter_and_place_it_in_the_basket
        libero_sim/pick_up_the_milk_and_place_it_in_the_basket
        libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket
        libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket
    )
elif [ "$suite" == "libero_spatial" ]; then
    TASKS=(
        libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
        libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
    )
fi

action_horizon=8
EPISODES=20
N_envs=1
PORT=$2

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/${suite}/baseline_nenvs${N_envs}_eps${EPISODES}_ah${action_horizon}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
        --n_episodes $EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port $PORT \
        --max_episode_steps=720 \
        --env_name "$TASK" \
        --n_action_steps $action_horizon \
        --n_envs $N_envs \
        --video_dir "$VIDEO_DIR"  \
        > "$LOG_DIR/${NAME}.txt" 2>&1

    echo "Finished task: $TASK"
    echo ""
done

