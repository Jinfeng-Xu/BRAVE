#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-amazon}"
TOOLS="${TOOLS:-ultra}"
LLM_MODEL="${LLM_MODEL:-gpt-5.6-luna}"
BUDGET="${BUDGET:-0.6}"
ETA="${ETA:-0.25}"
TAU="${TAU:-0.5}"

python main.py \
  --dataset "${DATASET}" \
  --rec_tool ${TOOLS} \
  --allocator brave \
  --budget "${BUDGET}" \
  --eta "${ETA}" \
  --tau "${TAU}" \
  --llm_model "${LLM_MODEL}" \
  --local_eval 0.1 \
  --llm_cmpr 0.1 \
  --local_eval_num 20 \
  --llm_cmpr_num 20 \
  --rerank_infr 20 \
  --eval_k 20
