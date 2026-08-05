# BRAVE run commands

These commands follow the draft-v22 protocol. They assume an OpenAI-compatible
endpoint, e.g., CLIProxyAPI, is configured in `.env`.

## Default BRAVE setting

```bash
python main.py \
  --dataset amazon \
  --rec_tool ultra \
  --allocator brave \
  --budget 0.6 \
  --eta 0.25 \
  --tau 0.5 \
  --llm_model gpt-5.6-luna \
  --local_eval 0.1 \
  --llm_cmpr 0.1 \
  --local_eval_num 20 \
  --llm_cmpr_num 20 \
  --rerank_infr 20 \
  --eval_k 20
```

## Uniform-reflection agent baseline

```bash
python main.py \
  --dataset amazon \
  --rec_tool ultra \
  --allocator none \
  --budget 1.0 \
  --llm_model gpt-5.6-luna \
  --local_eval 0.1 \
  --llm_cmpr 0.1 \
  --local_eval_num 20 \
  --llm_cmpr_num 20 \
  --rerank_infr 20 \
  --eval_k 20
```

## Tool-pool variants

```bash
python main.py --dataset amazon --rec_tool simple --allocator brave
python main.py --dataset amazon --rec_tool nlgcl fearec --allocator brave
python main.py --dataset amazon --rec_tool nlgcl wgcl fearec --allocator brave
python main.py --dataset amazon --rec_tool nlgcl wgcl fearec diffrec --allocator brave
```

## Allocator ablations

```bash
python main.py --dataset amazon --rec_tool ultra --allocator fragility
python main.py --dataset amazon --rec_tool ultra --allocator conflict
python main.py --dataset amazon --rec_tool ultra --allocator ungated
python main.py --dataset amazon --rec_tool ultra --allocator agg_margin
python main.py --dataset amazon --rec_tool ultra --allocator topk_boundary
python main.py --dataset amazon --rec_tool ultra --allocator random
```

## Controller variants

```bash
python main.py --dataset amazon --rec_tool ultra --allocator brave --llm_model gpt-4o
python main.py --dataset amazon --rec_tool ultra --allocator brave --llm_model gpt-5.6-luna
python main.py --dataset amazon --rec_tool ultra --allocator brave --llm_model gpt-5.6-terra
python main.py --dataset amazon --rec_tool ultra --allocator brave --llm_model phi4api
```
