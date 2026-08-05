# Open-source release checklist

Include source code and documentation:

- `main.py`
- `brave_allocator.py`
- `tool_config.py`
- `dataloader.py`
- `memory_bank.py`
- `prompt_bank.py`
- `utils.py`
- `data_process.py`
- `experiments/selective_reflection_probe.py`
- `README.md`
- `.env.example`
- `.gitignore`
- `requirements.txt`
- `to_run_gpt.md`
- `runned.sh`

Do not include local artifacts:

- trained recommender checkpoints or embeddings (`*.pt`, `*.pth`, `saved_embeddings/`, `checkpoints/`)
- generated user profiles (`user_profile_*.csv`)
- LLM logs and nohup outputs (`log/`, `nohup/`)
- diagnostic result folders (`results/`)
- paper-only figures or LaTeX build products

The public repository should expose the composable interface. Users can provide
their own trained tool embeddings and select any pool through `--rec_tool`.
