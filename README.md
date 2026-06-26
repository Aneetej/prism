# PRISM

**Pre/post-inference Runtime Inference Safety Monitor**

An LLM-agnostic two-stage inference-time safety layer. It wraps any language model and intercepts harmful content at two points: before the model generates a single token (Stage 1), and after the full output is produced (Stage 2). Neither stage is coupled to a specific model provider.

---

## Architecture

```
User prompt
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1 · Pre-Check            │  hybrid mode: regex patterns → ML classifier
│  pre_check.py                   │  ~0ms (regex hit) or ~30ms (classifier)
└─────────────────────────────────┘
     │
  FAIL──► Error message returned. LLM never runs.
     │
   PASS
     │
     ▼
┌─────────────────────────────────┐
│  LLM Inference                  │  any provider via LLMAdapter
│  llm/huggingface_adapter.py     │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Stage 2 · Output Check         │  RuleBasedChecker (default) or ClassifierChecker
│  checker/rule_based.py          │  runs on the complete generated output
└─────────────────────────────────┘
     │
  FAIL──► Error message returned. Output discarded.
     │
   PASS
     │
     ▼
  Output delivered to user
```

Both stages share a taxonomy of harm categories (`taxonomy/llama_guard_patterns.json`) based on Meta's Llama Guard taxonomy (S1–S14, extended with OWASP LLM Top 10 categories for prompt injection, system extraction, and resource abuse).

---

## Quick start

### Local

**Prerequisites:** Python 3.9+, a HuggingFace account with access to [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (accept the license on the model page before first use).

```bash
# 1. Clone and install
git clone <repo-url>
cd prism
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Set your HuggingFace token
cp .env.example .env
# Edit .env: replace hf_your_token_here with your actual token

# 3. Start the API
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# 4. Serve the frontend (separate terminal)
cd frontend && python -m http.server 3000
```

Open `http://localhost:3000` to use the UI. The API is at `http://localhost:8000/api/v1`.

### Docker

```bash
docker compose up --build
```

The API starts at `http://localhost:8000/api/v1`. Docker uses CPU-only PyTorch. MPS (Apple Silicon GPU) is not available inside Linux containers — inference will be slower than running locally on Apple Silicon.

---

## Switching models

PRISM wraps any LLM through a uniform adapter interface — the safety pipeline is identical regardless of what model sits behind it. Change the model by editing `config.yaml` and restarting the server.

### HuggingFace models

Any instruction-tuned model on HuggingFace Hub works. Change `llm.model_id` to the model's Hub path:

```yaml
llm:
  provider: huggingface
  model_id: meta-llama/Llama-3.2-3B-Instruct   # larger, slower, better quality
  device: mps                                    # mps | cuda | cpu
```

A few practical notes:
- **Gated models** (most Meta Llama models) require accepting the license on the model's HuggingFace page before your token grants download access. The error message at startup will say `401 Unauthorized` if this step is missing.
- **Memory**: a 1B model fits comfortably in ~2GB RAM; a 3B model needs ~6GB; a 7B model needs ~14GB on CPU (less with quantisation). If the model OOMs, add `load_in_4bit: true` and switch to `device: cuda`.
- **Chat format**: use instruction-tuned models (names ending in `-Instruct` or `-Chat`), not base models. Base models don't follow the prompt format PRISM uses.

### Ollama (no HuggingFace token required)

Pull any model from the Ollama library and point the config at it:

```bash
ollama pull mistral          # or llama3.2, gemma2, phi3, qwen2.5, etc.
```

```yaml
llm:
  provider: ollama
  model_id: mistral
  ollama_base_url: http://localhost:11434
```

Ollama handles quantisation and memory management automatically, making it the easiest way to try different models.

### Verifying provider-agnostic behaviour

The safety pipeline does not inspect or modify the LLM's weights — it only reads the text the model produces. You can confirm this by running the eval harness against two different providers:

```bash
# Test with Llama via HuggingFace
python eval/run_eval.py --checkers cascade --mode full_output

# Switch config to Ollama + Mistral, then re-run
python eval/run_eval.py --checkers cascade --mode full_output
```

The safety metrics (recall, FPR) should be driven by the checker, not the model — a weaker LLM may generate lower-quality responses but the pipeline will still intercept harmful content at both stages.

---

## Configuration (`config.yaml`)

| Key | Type | Default | Description |
|---|---|---|---|
| `llm.provider` | string | `huggingface` | LLM provider: `huggingface`, `openai`, `anthropic`, `ollama` |
| `llm.model_id` | string | `meta-llama/Llama-3.2-1B-Instruct` | HuggingFace model ID or local path |
| `llm.hf_token_env` | string | `HF_TOKEN` | Name of the env var holding the HuggingFace token |
| `llm.max_tokens` | int | `128` | Maximum tokens to generate per request |
| `llm.temperature` | float | `0.0` | Sampling temperature. `0.0` = greedy decoding (fastest, deterministic) |
| `llm.device` | string | `mps` | Device: `mps` (Apple Silicon), `cuda` (NVIDIA), `cpu`, `auto` |
| `pre_check.enabled` | bool | `true` | Enable/disable Stage 1 entirely |
| `pre_check.mode` | string | `hybrid` | `hybrid` (regex → classifier), `regex` (patterns only), `classifier` (ML only) |
| `pre_check.classifier_threshold` | float | `0.5` | Toxicity score above which the classifier blocks the prompt |
| `safety_checker.type` | string | `cascade` | Stage 2 checker: `rule_based`, `classifier`, `llm_judge`, `llama_guard`, `cascade`, `probe` |
| `safety_checker.device` | string | `cpu` | Device for the Stage 2 model (llama_guard, cascade, classifier) |
| `safety_checker.confidence_threshold` | float | `0.8` | Minimum confidence to block (classifier/probe checkers) |
| `safety_checker.cascade_skip_below` | float | `0.4` | `cascade` only — outputs with BERT unsafe_prob below this bypass Llama Guard |
| `llm.ollama_base_url` | string | `http://localhost:11434` | Ollama provider only |
| `experiment.mode` | string | `full_output` | Pipeline mode. `full_output` is the production setting. |
| `error_message` | string | `I'm not able to respond to that request.` | Message returned on any block |

---

## Stage 1: Pre-Check

The pre-check screens the incoming prompt synchronously before any LLM tokens are generated.

**`hybrid` (default):** Runs the regex/keyword taxonomy first (sub-millisecond). If no pattern matches, runs `unitary/toxic-bert` as a second pass (~30ms). Blocks if either fires. Regex catches structured harmful instructions cheaply ("how do I make a bomb?" → `S1_violent_crimes`); the ML classifier catches toxic language and threats that don't match any fixed pattern ("I want to kill my sister" → `classifier_toxic`).

**`regex`:** Taxonomy patterns only. Negligible latency but brittle — requires manual updates to `taxonomy/llama_guard_patterns.json` as new harmful phrasings emerge.

**`classifier`:** ML classifier only. Misses structured instruction requests because they don't contain offensive language.

### Tuning the threshold

`classifier_threshold: 0.3` blocks more aggressively (higher false positive rate). `0.7` passes more to Stage 2. Default `0.5` is balanced.

---

## Stage 2: Output Check

Stage 2 runs after the LLM finishes generating. It receives both the original prompt and the full output, giving it full context for the safety decision.

### Available checkers

| Checker | Config value | Latency | Notes |
|---|---|---|---|
| `RuleBasedChecker` | `rule_based` | ~0ms | Regex/keyword on output. No model required. Baseline. |
| `ClassifierChecker` | `classifier` | ~30ms CPU | `unitary/toxic-bert` on prompt + output. |
| `LLMJudgeChecker` | `llm_judge` | 50–150ms GPU | Small instruction-tuned model. Supports Ollama backend. |
| `LlamaGuardChecker` | `llama_guard` | ~130ms | Meta's Llama Guard 3 1B. Purpose-built safety classifier. Requires HF token. |
| `CascadeChecker` | `cascade` | ~30ms (skip) / ~130ms (full) | **Recommended.** BERT skip gate + Llama Guard. Fast for safe traffic, authoritative for borderline cases. |
| `RepresentationProbeChecker` | `probe` | ~2ms | Linear probe on hidden states. HuggingFace adapter only. Train first with `experiments/train_probe.py`. |

Switch checkers with `safety_checker.type` in `config.yaml`.

The `cascade` checker runs `unitary/toxic-bert` first. If the unsafe probability is below `cascade_skip_below` (default `0.4`), it returns immediately — Llama Guard never loads for that request. This keeps p50 latency low while still catching novel harmful outputs that BERT misses.

---

## API reference

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "checker": "RuleBasedChecker",
  "mode": "full_output"
}
```

### `POST /api/v1/run`

**Request:**
```json
{
  "prompt": "Tell me about the history of Rome.",
  "max_tokens": 128,
  "temperature": 0.0
}
```

**Response (passed):**
```json
{
  "output": "Rome was founded in 753 BC...",
  "passed": true,
  "blocked_at": null,
  "blocked_category": null,
  "latency_ms": 4821.3,
  "pre_check_latency_ms": 0.1,
  "llm_latency_ms": 4820.0,
  "checker_latency_ms": 0.5,
  "mode": "full_output",
  "model_id": "meta-llama/Llama-3.2-1B-Instruct"
}
```

**Response (blocked):**
```json
{
  "output": "I'm not able to respond to that request.",
  "passed": false,
  "blocked_at": "pre_check",
  "blocked_category": "S1_violent_crimes",
  "latency_ms": 0.02,
  "pre_check_latency_ms": 0.01,
  "llm_latency_ms": 0.0,
  "checker_latency_ms": 0.0,
  "mode": "full_output",
  "model_id": "meta-llama/Llama-3.2-1B-Instruct"
}
```

`blocked_at` is `"pre_check"` (LLM never ran) or `"safety_check"` (LLM ran but output was blocked). `llm_latency_ms` is `0` when blocked at pre-check.

### `POST /api/v1/stream`

Server-sent events (SSE) stream. Yields verified text chunks as they are released from the sliding-window buffer. On a safety block, yields the error message and closes the stream.

**Request:** same body as `/run`.

**Response:** `text/event-stream`, one `data: <chunk>` line per released buffer window.

### `GET /api/v1/config`

Returns the full parsed `config.yaml`.

---

## Running experiments

Compares Strategy A (sliding window, streaming) vs Strategy B (full output) on safety accuracy and latency across a labeled test set.

```bash
python experiments/compare_strategies.py

# With a specific checker
python experiments/compare_strategies.py --checker classifier
```

Outputs to `results/`:
- `latency_comparison.csv` — per-prompt latency by strategy
- `accuracy_comparison.csv` — pass/fail decisions vs ground truth
- `summary.json` — recall, FPR, F1, mean latency

---

## Roadmap

- **OpenAI and Anthropic adapters** — `llm/openai_adapter.py` and `llm/anthropic_adapter.py` are stubbed. Contributions welcome.
- **Probe + Stage 1 integration** — replace or augment the ML classifier in hybrid mode with the representation probe, reusing activations from the generating model at near-zero overhead.
- **Multi-turn context** — pass full conversation history to both stages so harmful intent that builds across turns is caught.

---

## Adding a new LLM provider

Implement `LLMAdapter` from `llm/base.py`:

```python
from llm.base import LLMAdapter, GenerationConfig, GenerationResult
from collections.abc import Iterator

class MyAdapter(LLMAdapter):
    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        ...  # call your provider API

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        ...  # yield text chunks
```

Register it in `pipeline.py` → `from_config()` under `_adapters`, then set `llm.provider: myadapter` in `config.yaml`.

---

## Docker

```bash
docker compose build          # build image
docker compose up -d          # start detached
curl localhost:8000/api/v1/health
docker compose logs -f api    # stream logs
docker compose down           # stop
```

**Volumes:** `./results` and `./models` are mounted into the container so experiment outputs and downloaded model weights survive container restarts. The `.env` file is injected via `env_file` and never baked into the image.

---

## CI/CD

| Workflow | File | Trigger | Action |
|---|---|---|---|
| CI | `ci.yml` | Every push and PR | `pytest tests/ -v` |
| Docker build | `docker.yml` | Push to `main` | `docker buildx build` (no push) |
| Experiment | `experiment.yml` | Manual or version tag `v*` | Runs A/B experiment, uploads `results/` as artifact (90 days) |

Add `HF_TOKEN` at **Settings → Secrets and variables → Actions → New repository secret** so CI can download models.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use a mock LLM (no model loading) and run in under one second. The CI workflow runs them on every push.

---

## Project structure

```
prism/
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pipeline.py               # PrismPipeline
├── pre_check.py              # Stage 1
├── buffer.py                 # Token buffer (sliding window mode)
├── stream_manager.py         # Stream gating (sliding window mode)
├── llm/
│   ├── base.py               # LLMAdapter interface
│   ├── huggingface_adapter.py
│   ├── openai_adapter.py     # stub — contributions welcome
│   ├── anthropic_adapter.py  # stub — contributions welcome
│   └── ollama_adapter.py
├── checker/
│   ├── base.py               # SafetyChecker interface
│   ├── rule_based.py
│   ├── classifier.py
│   ├── llm_judge.py
│   ├── llama_guard.py        # Llama Guard 3 1B
│   ├── cascade.py            # BERT skip gate + Llama Guard (recommended)
│   └── probe.py              # Representation probe (train first)
├── taxonomy/
│   └── llama_guard_patterns.json
├── api/
│   ├── app.py
│   ├── routes.py
│   └── schemas.py
├── frontend/
│   └── index.html
├── eval/
│   ├── datasets/             # labeled test sets (harmful / safe / boundary)
│   ├── run_eval.py           # evaluate any checker against all datasets
│   ├── metrics.py
│   └── fetch_xstest.py       # download XSTest false-positive benchmark
├── experiments/
│   ├── compare_strategies.py # sliding window vs full output A/B comparison
│   └── train_probe.py        # extract activations and train the probe
├── tests/
│   ├── conftest.py
│   ├── test_pre_check.py
│   ├── test_pipeline.py
│   └── test_checker_rule_based.py
└── .github/workflows/
    ├── ci.yml
    ├── docker.yml
    └── experiment.yml
```
