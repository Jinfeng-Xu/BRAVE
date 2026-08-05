# BRAVE: Budgeted Reflection Allocation for Recommendation Agents

## Dataset files

Place processed datasets under:

```text
data/{dataset}/train.txt
data/{dataset}/valid.txt
data/{dataset}/test.txt
data/{dataset}/item_descriptions.csv
```

See `data/README.md` for the exact format.

## Tool embeddings

For each tool and dataset, BRAVE expects:

```text
{tool}_{dataset}_user_embeddings.pt
{tool}_{dataset}_item_embeddings.pt
```

Put files under:

```text
embeddings/
embeddings/{tool}/
```

or pass a custom root:

```bash
python main.py --embedding_root /path/to/embeddings ...
```

Tool pools are open and composable. Built-in aliases:

```text
simple = lgn + srec + smpx
ultra  = nlgcl + wgcl + fearec
```

You can also pass any explicit combination:

```bash
--rec_tool nlgcl wgcl fearec
--rec_tool nlgcl fearec
--rec_tool nlgcl wgcl fearec diffrec
```

## LLM configuration

No API keys are stored in the code. Copy `.env.example` to `.env` and set the backend you need.

## Generate BRAVE-selected users without LLM calls

This reproduces the paper allocator:

```bash
python brave_allocator.py \
  --dataset amazon \
  --rec_tool ultra \
  --budget 0.6 \
  --eta 0.25 \
  --tau 0.5 \
  --top_k 20 \
  --start_batch 1 \
  --num_batch 40 \
  --batch_size 4
```

The script writes a selected-user file and a diagnostics CSV containing`g_frag`, `g_conf`, `g_brave`, gate membership, and ablation scores.

## Run BRAVE agent training

Default settings follow the paper: GPT-5.6 Luna controller, 60% budget,`eta=0.25`, `tau=0.5`, `k_eval=k_comp=k'_rank=20`, and full inference reranking.

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

## Outputs

- Logs: `log/{dataset}/`
- Selected users: `log/{dataset}/selected_*.txt`
- Allocator diagnostics: `log/{dataset}/selected_*.diagnostics.csv`
- Optional offline allocator results: `results/brave_selected_users/`

## Citation

The BibTeX entry will be added after acceptance.
