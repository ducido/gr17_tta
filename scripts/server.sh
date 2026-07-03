
TAG=$1
PORT=$2
MODEL_PATH=$3

if [ -z "$TAG" ] || [ -z "$PORT" ]; then
  echo "Usage: $0 [wdx | gg_robot | robocasa] [PORT] [MODEL_PATH (optional)]"
  exit 1
fi

# ===== Common setup =====
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

module load gcc/13.2.0
module load cuda/12.6.2

# ===== Config by tag =====
case "$TAG" in
  wdx)
    DEFAULT_MODEL_PATH="CP/GR00T-N1.6-bridge"
    EMBODIMENT="OXE_WIDOWX"
    ;;
  gg_robot)
    DEFAULT_MODEL_PATH="CP/GR00T-N1.6-fractal"
    EMBODIMENT="OXE_GOOGLE"
    ;;
  robocasa)
    DEFAULT_MODEL_PATH="CP/GR00T-N1.6-3B"
    EMBODIMENT="ROBOCASA_PANDA_OMRON"
    ;;
  libero_10)
    DEFAULT_MODEL_PATH="CP/GR00T-N1.7-LIBERO/libero_10"
    EMBODIMENT="LIBERO_PANDA"
    ;;
  *)
    echo "Unknown tag: $TAG"
    exit 1
    ;;
esac

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"

# ===== Run =====
echo "Running with:"
echo "  TAG=$TAG"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  EMBODIMENT=$EMBODIMENT"
echo "  PORT=$PORT"

.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path $MODEL_PATH \
    --embodiment-tag $EMBODIMENT \
    --use-sim-policy-wrapper \
    --port $PORT