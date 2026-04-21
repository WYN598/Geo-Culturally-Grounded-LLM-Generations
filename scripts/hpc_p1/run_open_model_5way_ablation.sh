#!/bin/bash
set -euo pipefail

: "${MODEL_ID:?MODEL_ID must be set}"
: "${BENCH_TAG:?BENCH_TAG must be set}"
: "${CONFIG_PATH:?CONFIG_PATH must be set}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${LOG_PREFIX:?LOG_PREFIX must be set}"

TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"

cd /dtu/p1/yanwen/bias-rag
module purge
module load python3/3.11.9
source .venv/bin/activate
if [ -f /dtu/p1/yanwen/bias-rag/.env.private ]; then
  source /dtu/p1/yanwen/bias-rag/.env.private
fi
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.11/site-packages:${PYTHONPATH}"

export HF_HOME=/dtu/p1/yanwen/bias-rag/hf-cache
export HUGGINGFACE_HUB_CACHE=/dtu/p1/yanwen/bias-rag/hf-cache/hub
export VLLM_CACHE_ROOT=/dtu/p1/yanwen/bias-rag/vllm-cache
export TORCHINDUCTOR_CACHE_DIR=/dtu/p1/yanwen/bias-rag/torchinductor-cache
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

if [ -z "${OPENAI_EMBEDDING_BASE_URL:-}" ] || [ -z "${OPENAI_EMBEDDING_API_KEY:-}" ]; then
  echo "OPENAI_EMBEDDING_BASE_URL and OPENAI_EMBEDDING_API_KEY must be set." >&2
  exit 1
fi

cd /dtu/p1/yanwen/bias-rag/project
mkdir -p logs

VLLM_CMD=(
  vllm serve "$MODEL_ID"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --api-key local-key
)

"${VLLM_CMD[@]}" > "logs/${LOG_PREFIX}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID || true' EXIT

for i in $(seq 1 240); do
  if curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-key" > /dev/null; then
    break
  fi
  sleep 5
done

export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-key

python scripts/run_general_ablation.py \
  --config "$CONFIG_PATH" \
  --out-root "$OUT_ROOT" \
  --tag "$BENCH_TAG" \
  --provider openai \
  --model "$MODEL_ID" \
  --temperature "$TEMPERATURE" \
  --no-full-general
