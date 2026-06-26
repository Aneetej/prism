# Contributing

## Reporting bugs

Open a [GitHub Issue](../../issues). Include the Python version, OS, any relevant config values, and a minimal reproduction.

## Submitting changes

1. Fork the repo and create a branch off `main` (`git checkout -b your-feature`).
2. Make your changes. Keep commits focused — one logical change per commit.
3. Run the tests before opening a PR: `pytest tests/ -v`
4. Open a pull request against `main`. Describe what changed and why.

## Code style

- Follow existing patterns in the file you're editing.
- No new runtime dependencies without discussion in the issue first.
- Comments should explain *why*, not *what* — the code itself handles what.
- Tests are required for new safety logic (pre-check patterns, checker behaviour).

## Adding a new LLM provider

Implement `LLMAdapter` from `llm/base.py` (both `generate` and `stream`), add it to the `_adapters` registry in `pipeline.py → from_config()`, and document it in the config table in `README.md`. See `llm/ollama_adapter.py` for a reference implementation.

## Adding a new safety checker

Implement `SafetyChecker` from `checker/base.py`, add it to `pipeline.py → from_config()`, and add a row to the Stage 2 checker table in `README.md`.
