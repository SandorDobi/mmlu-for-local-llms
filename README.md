# MMLU Benchmark Suite for Local LLMs

Custom [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) task configs for running **MMLU benchmarks against local LLMs** via OpenAI-compatible chat/completions APIs.

Designed for **reasoning models** (MiniMax, DeepSeek-R1, QwQ, etc.) that tend to output long chain-of-thought instead of bare answer letters. Uses generative evaluation with regex-based answer extraction.

## Why This Exists

The standard MMLU task in lm-eval uses `loglikelihood` scoring, which requires the server to return logprobs -- many local inference servers don't support this. When switching to generative (`generate_until`) evaluation, reasoning models produce verbose explanations instead of clean A/B/C/D answers.

This suite solves that with:
- A prompt template that encourages structured output (`ANSWER: X`)
- A multi-pattern regex filter that extracts answer letters from various response formats
- 5-shot few-shot examples to teach the model the expected format
- Pre-built configs for both full MMLU (57 subjects) and a focused 10-subject subset

## Supported Answer Formats

The regex filter uses **right-to-left matching** to extract the answer from the **end** of the model's response. This prevents false matches when the model cites answer options during reasoning (e.g. `**A** is incorrect because...`).

### How it works

The regex uses `(?s)` (DOTALL flag) with a greedy `.*` prefix on each pattern alternative. Since `.*` is greedy and DOTALL allows `.` to match newlines, `re.findall()` finds the **last** occurrence of each pattern in the response. Alternatives are ordered by priority -- explicit `ANSWER: X` is tried first, fallback bare letter last.

**Problem it solves:** Without right-to-left matching, a response like:
```
**A** is incorrect because evolution does not aim for perfection.
**B** is correct. The phenotype represents a compromise.
ANSWER: B
```
Would incorrectly extract `A` (from the bold `**A**` in the reasoning) instead of `B` (the actual answer at the end).

### Supported patterns (priority order)

| Priority | Pattern | Example Match |
|---|---|---|
| 1 (highest) | `ANSWER: X` | `ANSWER: B` |
| 2 | `answer is X` | `The answer is C.` |
| 3 | `Answer: X` | `Answer: C` |
| 4 | `correct answer is X` | `The correct answer is D.` |
| 5 | `**X**` (bold) | `**B**` |
| 6 | `**X. ...` (bold + dot) | `**B. some text` |
| 7 | Bare letter on last line | `B` or `B.` |
| 8 (fallback) | Letter + whitespace/end | `A. text...` |

### Tested Models

This regex has been validated against results from:
- **JANGQ-AI/MiniMax-M2.7-JANG_3L** -- reasoning model, verbose chain-of-thought
- **KnucklesXBT/Qwen3.6-35B-A3B-mlx-8Bit** -- reasoning model, structured option analysis

## Requirements

- Python 3.10+
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`pip install lm-eval`)
- A local OpenAI-compatible API server (vLLM, llama.cpp, SGLang, Ollama, LM Studio, etc.)

## Directory Structure

```
mmlu_custom/
├── JANG_RGX/                    # Full MMLU (57 subjects)
│   ├── _default_template.yaml   # Shared task template
│   ├── _mmlu.yaml               # Group definition (stem/other/social/humanities)
│   ├── eval_config.yaml         # Run configuration
│   └── mmlu_*.yaml              # 57 subject task definitions
│
└── JANG_RGX_20/                 # Focused subset (10 subjects x 20 questions)
    ├── _default_template.yaml   # Shared task template
    ├── _group.yaml              # Group definition (10 subjects)
    ├── eval_config.yaml         # Run configuration
    ├── show_results.py          # Results viewer with loop detection
    └── mmlu_*.yaml              # 10 subject task definitions
```

## Quick Start

### 1. Install lm-eval

```bash
pip install lm-eval
```

### 2. Configure your model and server

Edit the `model_args` section in the `eval_config.yaml` file:

```yaml
model_args:
  model: your-model-name                    # model identifier sent to the server
  base_url: http://127.0.0.1:8080/v1/chat/completions  # your local API endpoint
  num_concurrent: 4                         # parallel requests
  max_retries: 3                            # retry count on errors
  timeout: 600                              # per-request timeout (seconds)
```

### 3. Run the benchmark

**Full MMLU (57 subjects, ~14,000 questions):**
```bash
lm_eval run --config JANG_RGX/eval_config.yaml
```

**Focused subset (10 subjects x 20 questions):**
```bash
lm_eval run --config JANG_RGX_20/eval_config.yaml
```

**Quick test (3 questions, single subject):**
```bash
lm_eval run --config JANG_RGX_20/eval_config.yaml --limit 3 --tasks JANG_RGX_20_abstract_algebra
```

### 4. View results

```bash
# Summary table (X/20 format)
python3 JANG_RGX_20/show_results.py

# Per-sample detail with loop and truncation detection
python3 JANG_RGX_20/show_results.py --detail
```

Output example:
```
| Subject                 | Score   | Accuracy |
|------------------------|---------|----------|
| Abstract Algebra       |   8/20  |    40%   |
| Anatomy                |  16/20  |    80%   |
| Astronomy              |  19/20  |    95%   |
| ...
| TOTAL                  | 151/200 |   75.5%  |
```

## CLI Overrides

CLI arguments always override config file values:

```bash
# Change few-shot count
lm_eval run --config JANG_RGX_20/eval_config.yaml --num_fewshot 0

# Limit samples per task
lm_eval run --config JANG_RGX_20/eval_config.yaml --limit 5

# Run a single subject
lm_eval run --config JANG_RGX_20/eval_config.yaml --tasks JANG_RGX_20_anatomy

# Change model on the fly
lm_eval run --config JANG_RGX_20/eval_config.yaml \
  --model_args "model=meta-llama/Llama-3-8B,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=4,max_retries=3,timeout=600"
```

## Configuration Reference

### eval_config.yaml

| Parameter | Default | Description |
|---|---|---|
| `model` | `local-chat-completions` | Backend type (use `local-chat-completions` for API servers) |
| `model_args.model` | - | Model name sent to the API |
| `model_args.base_url` | - | Full URL to `/v1/chat/completions` endpoint |
| `model_args.num_concurrent` | 4 | Parallel API requests |
| `model_args.max_retries` | 3 | Retries on transient errors |
| `model_args.timeout` | 600 | Per-request timeout in seconds |
| `tasks` | `JANG_RGX_20` or `JANG_RGX_mmlu` | Task group to run |
| `num_fewshot` | 5 | Number of few-shot examples (5 is MMLU standard) |
| `limit` | 20 or null | Max questions per subject (null = all) |
| `log_samples` | true | Save per-sample results to JSONL |
| `output_path` | `./mmlu_results` | Where to save results |

### Task Template (_default_template.yaml)

| Parameter | Description |
|---|---|
| `doc_to_text` | Prompt format with "Think step by step, then end with ANSWER: X" instruction |
| `generation_kwargs.max_tokens` | 4096 -- enough for full reasoning chains |
| `generation_kwargs.until` | `["</s>"]` -- stop on EOS token |
| `filter_list` | Regex patterns for answer extraction |

## Results Viewer (show_results.py)

The `show_results.py` script reads JSONL results and displays a summary table with X/20 scores. It supports interactive model and run selection.

### Usage

```bash
# Default: latest model, latest run
python3 JANG_RGX_20/show_results.py

# Per-sample detail
python3 JANG_RGX_20/show_results.py --detail

# List all available models and runs
python3 JANG_RGX_20/show_results.py --list

# Interactively choose model
python3 JANG_RGX_20/show_results.py --model

# Interactively choose both model and result set
python3 JANG_RGX_20/show_results.py --model --run

# Point to specific model directory (legacy)
python3 JANG_RGX_20/show_results.py /path/to/mmlu_results/model-name --detail
```

### Flags

| Flag | Description |
|---|---|
| `-d`, `--detail` | Show per-sample detail with tags |
| `--model` | Interactively choose model from list |
| `--run` | Interactively choose result set (timestamp) within a model |
| `--list` | List all models and runs, then exit |
| `--latest` | Use latest run (default behavior) |
| `-h`, `--help` | Show help |

### Interactive selection

`--model` shows a numbered list of models found in `mmlu_results/` and prompts for selection. Press Enter to use the latest.

`--run` shows all timestamped result sets within the selected model. Each run is numbered with its timestamp and subject count. Press Enter or type `L` for the latest run. This lets you compare different runs (e.g. different `num_fewshot` values or config changes) against the same model.

```
$ python3 JANG_RGX_20/show_results.py --model --run

Available models:
  [1] JANGQ-AI__MiniMax-M2.7-JANG_3L
  [2] KnucklesXBT__Qwen3.6-35B-A3B-mlx-8Bit

Select model (number) or Enter for latest: 1

Available result sets:
  [1] 2026-04-16T09-06-14.718350  (10 subjects)
  [2] 2026-04-16T09-29-38.353794  (10 subjects)
  [3] 2026-04-16T15-27-03.624255  (10 subjects)
  [L] Latest

Select run (number/L) or Enter for latest: 2

Model: JANGQ-AI__MiniMax-M2.7-JANG_3L
Run:   2026-04-16T09-29-38.353794

| Subject                |   Score | Accuracy |
|------------------------|---------|----------|
| Abstract Algebra       |   5/20  |      25% |
| ...
```

### Tags in --detail mode

| Tag | Meaning |
|---|---|
| `OK` | Correct answer |
| `MISS` | Wrong answer |
| `[NO_ANSWER]` | Regex couldn't extract any letter |
| `[LOOP]` | Model entered a repetition loop |
| `[TRUNCATED]` | Response cut off by max_tokens |

## JANG_RGX_20 Subjects

The focused subset covers 10 representative subjects:

1. Abstract Algebra
2. Anatomy
3. Astronomy
4. College Computer Science
5. College Physics
6. High School Biology
7. High School Chemistry
8. High School Mathematics
9. Logical Fallacies
10. World Religions

## Adapting for Other Models

1. Edit `model_args` in `eval_config.yaml` to point to your server
2. Adjust `num_fewshot` (0 for no examples, 5 for standard MMLU)
3. Adjust `max_tokens` in `_default_template.yaml` (lower = faster but more truncation)
4. Adjust `num_concurrent` based on your server's capacity

## Regex Version History

### v2 -- Right-to-left matching (current)

**Problem:** When models like Qwen3 analyze each option in their reasoning (e.g. `**A.** This is incorrect because...`, `**B.** This is the correct answer.`), the old left-to-right regex matched the first occurrence -- typically option A or B from the analysis -- instead of the actual `ANSWER: X` at the end.

**Fix:** Added `(?s)` inline DOTALL flag and greedy `.*` prefix to each alternative. This forces the regex engine to match the **last** occurrence of each pattern, effectively scanning from right-to-left. Alternatives are ordered by specificity: explicit `ANSWER: X` markers first, fuzzy patterns last.

**Impact on test data (2860 samples across 2 models):**
- Old regex: 65.9% accuracy (15 invalid, 1885 correct)
- New regex: 73.8% accuracy (3 invalid, 2110 correct)
- 228 false-match errors fixed, 0 real regressions

### v1 -- Left-to-right matching (initial)

First version using standard left-to-right `re.search`. Worked well for models that always output `ANSWER: X` cleanly (MiniMax), but produced systematic false matches on models that enumerate options in their reasoning (Qwen).

## Notes

- Generative MMLU scores are typically lower than loglikelihood-based MMLU. The two methods are not directly comparable.
- `num_fewshot: 5` is the canonical MMLU setting used in most published benchmarks.
- The `--limit` flag should only be used for testing. Real metrics should use the full dataset.
- Results are saved as JSONL in `mmlu_results/<model-name>/` with timestamps, so multiple runs coexist.

## License

MIT
