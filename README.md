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

The regex filter extracts `A`, `B`, `C`, or `D` from any of these response patterns:

| Model Output | Extracted |
|---|---|
| `ANSWER: B` | B |
| `The answer is C.` | C |
| `The correct answer is D.` | D |
| `**B. some text**` | B |
| `**Answer: C**` | C |
| `A` (bare letter) | A |
| `A.` (letter + period) | A |
| `A. some explanation text` | A |

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

The `show_results.py` script reads the latest JSONL results and displays them.

```bash
# Summary only
python3 JANG_RGX_20/show_results.py

# Per-sample detail
python3 JANG_RGX_20/show_results.py --detail

# Point to specific model directory
python3 JANG_RGX_20/show_results.py /path/to/mmlu_results/model-name --detail
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

## Notes

- Generative MMLU scores are typically lower than loglikelihood-based MMLU. The two methods are not directly comparable.
- `num_fewshot: 5` is the canonical MMLU setting used in most published benchmarks.
- The `--limit` flag should only be used for testing. Real metrics should use the full dataset.
- Results are saved as JSONL in `mmlu_results/<model-name>/` with timestamps, so multiple runs coexist.

## License

MIT
