#!/usr/bin/env bash
set -euo pipefail

cd /nfs/home/seymol/paper

PORT=8730
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping server (PID ${SERVER_PID})"
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=5 python \
  -m lora_offline.prefetch_rlvf_fastapi_server \
  --model gghfez/gemma-3-4b-novision \
  --port "$PORT" \
  --host 127.0.0.1 \
  --max-lora-rank 128 &
SERVER_PID=$!

echo "Server started with PID ${SERVER_PID}"

# Wait for /health (max 30 minutes)
for ((i=0;i<360;i++)); do

  # server crashed
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process exited unexpectedly." >&2
    exit 1
  fi

  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "Server is ready."
    break
  fi

  sleep 5
done

# final readiness check
if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "Server did not become ready." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=7 accelerate launch \
  --mixed_precision bf16 \
  -m lora_offline.train \
  -c /nfs/home/seymol/paper/config/rl/gm_3b/arc.json \
  &> logs.out
