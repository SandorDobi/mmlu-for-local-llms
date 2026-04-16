#!/usr/bin/env python3
"""Show JANG_RGX_20 benchmark results in X/20 format with per-sample details."""

import json
import sys
import glob
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "mmlu_results")

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

def load_results(samples_dir):
    """Load all sample results from JSONL files."""
    results = {}
    for f in glob.glob(os.path.join(samples_dir, "results_*.json")):
        with open(f) as fh:
            data = json.load(fh)
            for task_name, task_data in data.get("results", {}).items():
                if task_name.startswith("JANG_RGX_20_"):
                    short = task_name.replace("JANG_RGX_20_", "")
                    results[short] = task_data
    return results

def detect_loop(resp, min_chunks=3, chunk_size=120):
    """Detect if the model response has repetition loops.

    Distinguishes between genuine looping (model stuck repeating itself)
    and normal repetition (quoting options, reusing formulas, enumerations).
    """
    if len(resp) < chunk_size * min_chunks:
        return False
    # Chunk-level repetition: identical 120-char blocks repeating 3+ times
    chunks = [resp[i:i+chunk_size] for i in range(0, len(resp), chunk_size)]
    for i in range(len(chunks) - min_chunks):
        pattern = chunks[i]
        matches = sum(1 for c in chunks[i+1:] if c == pattern)
        if matches >= min_chunks:
            return True
    # Phrase-level repetition: only flag long phrases (12+ words) repeated
    # many times (5+). Short 8-word phrases at count 3 are too aggressive --
    # they trigger on normal reasoning (quoting options, reusing formulas,
    # enumeration patterns like "Therefore, option X is the correct...").
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

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show_detail = "--detail" in sys.argv or "-d" in sys.argv

    if args:
        model_dir = args[0]
    else:
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), key=os.path.getmtime, reverse=True)
        if not dirs:
            print("No results found in", RESULTS_DIR)
            sys.exit(1)
        model_dir = dirs[0]

    model_name = os.path.basename(model_dir)
    print(f"Model: {model_name}")
    print()

    sample_files = find_latest_samples(model_dir)

    # Header
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

    # Detailed per-sample view
    if show_detail:
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

                    # Build status tag
                    tags = []
                    if d.get("exact_match") == 1.0:
                        mark = "OK  "
                    else:
                        mark = "MISS"
                        if filt == "[invalid]":
                            tags.append("NO_ANSWER")

                    if detect_loop(resp_text):
                        tags.append("LOOP")

                    # Truncated: response appears cut off mid-sentence.
                    # A clean answer like "ANSWER: B" ending with a letter is NOT truncated.
                    is_truncated = False
                    if resp_text and len(resp_text) > 200:
                        last = resp_text.rstrip()[-1]
                        if last not in ".!?\n":
                            # Check if it ends with a known answer pattern (not truncated)
                            stripped = resp_text.rstrip()
                            if not (stripped.endswith(tuple("ABCD")) and
                                    ("ANSWER:" in stripped[-20:] or
                                     stripped[-1] in "ABCD" and len(stripped) < 5)):
                                # Ends mid-word or mid-sentence without an answer marker
                                is_truncated = True
                    if is_truncated:
                        tags.append("TRUNCATED")

                    tag_str = f" [{','.join(tags)}]" if tags else ""
                    print(f"  [{mark}]{tag_str} target={target} got={filt:<10} {q}...")

if __name__ == "__main__":
    main()
