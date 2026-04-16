#!/usr/bin/env python3
"""Show JANG_RGX_20 benchmark results in X/20 format with per-sample details."""

import json
import sys
import glob
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "mmlu_results")

# Task display order and names
TASK_DISPLAY = [
    ("abstract_algebra", "Abstract Algebra"),
    ("anatomy", "Anatomy"),
    ("astronomy", "Astronomy"),
    ("college_computer_science", "College CS"),
    ("college_physics", "College Physics"),
    ("high_school_biology", "HS Biology"),
    ("high_school_chemistry", "HS Chemistry"),
    ("high_school_mathematics", "HS Mathematics"),
    ("logical_fallacies", "Logical Fallacies"),
    ("world_religions", "World Religions"),
]


def discover_models(results_dir):
    """Find all model directories with JANG_RGX_20 results."""
    models = []
    for d in sorted(glob.glob(os.path.join(results_dir, "*/"))):
        model = os.path.basename(d.rstrip("/"))
        if glob.glob(os.path.join(d, "samples_JANG_RGX_20_*.jsonl")):
            models.append(model)
    return models


def discover_runs(model_dir):
    """Find all run timestamps and their sample files within a model directory."""
    timestamps = set()
    for f in glob.glob(os.path.join(model_dir, "samples_JANG_RGX_20_*.jsonl")):
        parts = os.path.basename(f).split("_2026-")
        if len(parts) == 2:
            ts = "2026-" + parts[1].replace(".jsonl", "")
            timestamps.add(ts)
    runs = {}
    for ts in sorted(timestamps):
        files = sorted(glob.glob(os.path.join(model_dir, f"samples_JANG_RGX_20_*_{ts}.jsonl")))
        subjects = []
        for f in files:
            name = os.path.basename(f).split("JANG_RGX_20_")[1].split(f"_{ts}")[0]
            subjects.append(name)
        runs[ts] = {"files": files, "subjects": subjects}
    return runs


def find_latest_samples(model_dir):
    """Find the latest sample files, grouped by task."""
    task_files = defaultdict(list)
    for f in glob.glob(os.path.join(model_dir, "samples_JANG_RGX_20_*.jsonl")):
        task_name = os.path.basename(f).split("JANG_RGX_20_")[1].split("_2026")[0]
        task_files[task_name].append(f)
    latest = {}
    for task, files in task_files.items():
        files.sort(reverse=True)
        latest[task] = files[0]
    return latest


def load_run_samples(run_files):
    """Load sample files from a specific run, keyed by subject name."""
    samples = {}
    for f in run_files:
        basename = os.path.basename(f)
        subject = basename.split("JANG_RGX_20_")[1].split("_2026-")[0]
        samples[subject] = f
    return samples


def detect_loop(resp, min_chunks=3, chunk_size=120):
    """Detect if the model response has repetition loops.

    Distinguishes between genuine looping (model stuck repeating itself)
    and normal repetition (quoting options, reusing formulas, enumerations).
    """
    if len(resp) < chunk_size * min_chunks:
        return False
    chunks = [resp[i:i+chunk_size] for i in range(0, len(resp), chunk_size)]
    for i in range(len(chunks) - min_chunks):
        pattern = chunks[i]
        matches = sum(1 for c in chunks[i+1:] if c == pattern)
        if matches >= min_chunks:
            return True
    words = resp.split()
    if len(words) < 30:
        return False
    for span_len in [12, 16]:
        for i in range(len(words) - span_len):
            phrase = " ".join(words[i:i+span_len])
            count = resp.count(phrase)
            if count >= 5:
                return True
    return False


def pick_model(models):
    """Interactive model selection. Returns model name or None."""
    print("Available models:")
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m}")
    print()
    try:
        choice = input("Select model (number) or Enter for latest: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not choice:
        return models[-1]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        pass
    print(f"Invalid choice: {choice}")
    return None


def pick_run(runs):
    """Interactive run selection. Returns timestamp or None."""
    run_list = sorted(runs.keys())
    print("Available result sets:")
    for i, ts in enumerate(run_list, 1):
        n = len(runs[ts]["subjects"])
        print(f"  [{i}] {ts}  ({n} subjects)")
    print(f"  [L] Latest")
    print()
    try:
        choice = input("Select run (number/L) or Enter for latest: ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not choice or choice == "L":
        return run_list[-1]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(run_list):
            return run_list[idx]
    except ValueError:
        pass
    print(f"Invalid choice: {choice}")
    return None


def list_models_runs(results_dir):
    """List all models and runs."""
    models = discover_models(results_dir)
    if not models:
        print("No results found in", results_dir)
        return
    for model in models:
        model_dir = os.path.join(results_dir, model)
        runs = discover_runs(model_dir)
        print(f"{model}:")
        for i, (ts, info) in enumerate(sorted(runs.items()), 1):
            print(f"  [{i}] {ts}  ({len(info['subjects'])} subjects)")
        print()


def print_usage():
    print("Usage: show_results.py [OPTIONS] [MODEL_DIR]")
    print()
    print("Options:")
    print("  -d, --detail     Show per-sample detail")
    print("  --model          Interactively choose model")
    print("  --run            Interactively choose result set")
    print("  --list           List available models and runs")
    print("  --latest         Use latest run (default)")
    print("  -h, --help       Show this help")
    print()
    print("If MODEL_DIR is given, uses that model's latest results.")
    print("Otherwise auto-selects the most recent model directory.")


def display_results(sample_files, model_name, run_label=""):
    """Display the summary results table."""
    print(f"Model: {model_name}")
    if run_label:
        print(f"Run:   {run_label}")
    print()

    print(f"| {'Subject':<22} | {'Score':>7} | {'Accuracy':>8} |")
    print(f"|{'-'*24}|{'-'*9}|{'-'*10}|")

    total_correct = 0
    total_questions = 0

    for dataset_name, display_name in TASK_DISPLAY:
        f = sample_files.get(dataset_name)
        if not f:
            print(f"| {display_name:<22} | {'N/A':>7} | {'N/A':>8} |")
            continue

        correct = 0
        total = 0
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                total += 1
                if d.get("exact_match", 0) == 1.0:
                    correct += 1

        total_correct += correct
        total_questions += total
        pct = f"{correct/total*100:.0f}%" if total else "N/A"
        print(f"| {display_name:<22} | {correct:>3}/{total:<3} | {pct:>8} |")

    print(f"|{'-'*24}|{'-'*9}|{'-'*10}|")
    if total_questions:
        pct = f"{total_correct/total_questions*100:.1f}%"
        print(f"| {'TOTAL':<22} | {total_correct:>3}/{total_questions:<3} | {pct:>8} |")


def display_detail(sample_files):
    """Display per-sample detail."""
    print(f"\n{'='*60}")
    print("Per-sample detail")
    print(f"{'='*60}")
    for dataset_name, display_name in TASK_DISPLAY:
        f = sample_files.get(dataset_name)
        if not f:
            continue
        print(f"\n--- {display_name} ---")
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                resp_text = d["resps"][0][0]
                filt = d["filtered_resps"][0] if d.get("filtered_resps") else "?"
                target = d.get("target", "?")
                q = d["doc"].get("question", "")[:50]

                tags = []
                if d.get("exact_match") == 1.0:
                    mark = "OK  "
                else:
                    mark = "MISS"
                    if filt == "[invalid]":
                        tags.append("NO_ANSWER")

                if detect_loop(resp_text):
                    tags.append("LOOP")

                is_truncated = False
                if resp_text and len(resp_text) > 200:
                    last = resp_text.rstrip()[-1]
                    if last not in ".!?\n":
                        stripped = resp_text.rstrip()
                        if not (stripped.endswith(tuple("ABCD")) and
                                ("ANSWER:" in stripped[-20:] or
                                 stripped[-1] in "ABCD" and len(stripped) < 5)):
                            is_truncated = True
                if is_truncated:
                    tags.append("TRUNCATED")

                tag_str = f" [{','.join(tags)}]" if tags else ""
                print(f"  [{mark}]{tag_str} target={target} got={filt:<10} {q}...")


def main():
    argv = sys.argv[1:]
    show_detail = "--detail" in argv or "-d" in argv
    use_model_picker = "--model" in argv
    use_run_picker = "--run" in argv
    do_list = "--list" in argv

    if "--help" in argv or "-h" in argv:
        print_usage()
        sys.exit(0)

    if do_list:
        list_models_runs(RESULTS_DIR)
        sys.exit(0)

    # Determine model directory
    pos_args = [a for a in argv if not a.startswith("-")]
    model_dir = None

    if pos_args:
        model_dir = os.path.abspath(pos_args[0])
    elif use_model_picker:
        models = discover_models(RESULTS_DIR)
        if not models:
            print("No results found in", RESULTS_DIR)
            sys.exit(1)
        model = pick_model(models)
        if model is None:
            sys.exit(0)
        model_dir = os.path.join(RESULTS_DIR, model)
    else:
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")),
                       key=os.path.getmtime, reverse=True)
        if not dirs:
            print("No results found in", RESULTS_DIR)
            sys.exit(1)
        model_dir = dirs[0]

    model_name = os.path.basename(model_dir)

    # Determine which run to use
    runs = discover_runs(model_dir)
    if not runs:
        print("No JANG_RGX_20 results found in", model_dir)
        sys.exit(1)

    sample_files = None
    run_label = ""

    if use_run_picker:
        ts = pick_run(runs)
        if ts is None:
            sys.exit(0)
        run_info = runs[ts]
        sample_files = load_run_samples(run_info["files"])
        run_label = ts
    else:
        sample_files = find_latest_samples(model_dir)
        latest_ts = sorted(runs.keys())[-1]
        run_label = f"latest ({latest_ts})"

    display_results(sample_files, model_name, run_label)

    if show_detail:
        display_detail(sample_files)


if __name__ == "__main__":
    main()
