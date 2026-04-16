#!/usr/bin/env python3
"""Run MMLU benchmark against a local inference server.

Reads an eval_config.yaml, queries the server for available models,
and either runs lm_eval or prints the command.
"""

import json
import sys
import os
import subprocess
import urllib.request
import urllib.error
import glob
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_configs(script_dir):
    """Find all eval_config.yaml files in subdirectories."""
    configs = []
    for f in sorted(glob.glob(os.path.join(script_dir, "*/eval_config.yaml"))):
        name = os.path.basename(os.path.dirname(f))
        configs.append((name, f))
    return configs


def load_config(path):
    """Load YAML config as dict (simple parser, no pyyaml dependency)."""
    with open(path) as f:
        return f.read()


def parse_yaml_simple(text):
    """Parse a flat subset of YAML (enough for eval_config structure)."""
    result = {}
    current_path = []
    lines = text.split('\n')

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # Measure indent
        indent = len(line) - len(line.lstrip())

        # Key: value
        m = re.match(r'^(\w+):\s*(.*)', stripped)
        if not m:
            continue

        key = m.group(1)
        value = m.group(2).strip()

        # Determine nesting level (2 spaces per level)
        level = indent // 2

        # Adjust current_path to this level
        current_path = current_path[:level]

        if value and not value.startswith('#'):
            # Leaf value
            parsed_value = parse_yaml_value(value)
            set_nested(result, current_path + [key], parsed_value)
        else:
            # Section header (will have children)
            set_nested(result, current_path + [key], {})
            current_path.append(key)

    return result


def parse_yaml_value(value):
    """Parse a YAML scalar value."""
    if not value or value == 'null' or value == '~':
        return None
    if value == 'true':
        return True
    if value == 'false':
        return False
    # Quoted string
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    # Number
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # List like [0, 1234, ...]
    if value.startswith('[') and value.endswith(']'):
        items = value[1:-1].split(',')
        return [parse_yaml_value(i.strip()) for i in items]
    return value


def set_nested(d, path, value):
    """Set a value in a nested dict by path."""
    for key in path[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[path[-1]] = value


def extract_base_url(config_text):
    """Extract base_url from raw YAML text (avoids full parse)."""
    for line in config_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('base_url:'):
            # Split on first colon after 'base_url'
            rest = stripped[len('base_url:'):].strip()
            # Strip inline comment
            if '#' in rest:
                rest = rest[:rest.index('#')].strip()
            url = rest.strip('"').strip("'")
            if url:
                return url
    return None


def extract_model_args(config_text):
    """Extract all key-value pairs from the model_args section."""
    args = {}
    in_model_args = False
    for line in config_text.split('\n'):
        stripped = line.strip()
        if stripped == 'model_args:':
            in_model_args = True
            continue
        if in_model_args:
            if not line.startswith(' ') and not line.startswith('\t'):
                break
            if stripped.startswith('#') or not stripped:
                continue
            # Parse "key: value"
            if ':' in stripped:
                key, val = stripped.split(':', 1)
                key = key.strip()
                val = val.strip()
                # Skip comment-only lines
                if not val or val.startswith('#'):
                    continue
                # Strip inline comment
                if '#' in val:
                    val = val[:val.index('#')].strip()
                val = val.strip('"').strip("'")
                if val:
                    args[key] = val
    return args


def get_server_origin(base_url):
    """Extract origin (scheme://host:port) from base_url."""
    # base_url is like http://127.0.0.1:8080/v1/chat/completions
    parts = base_url.split('/')
    if len(parts) >= 3:
        return '/'.join(parts[:3])
    return base_url


def query_models(origin):
    """Query /v1/models endpoint. Returns list of model dicts or None."""
    url = origin.rstrip('/') + '/v1/models'
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get('data', [])
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def pick_model(models, configured_model):
    """Interactive model selection."""
    if not models:
        print("  No models available from server.")
        if configured_model:
            print(f"  Falling back to configured model: {configured_model}")
            return configured_model
        return None

    print(f"  Available models ({len(models)}):")
    for i, m in enumerate(models, 1):
        mid = m.get('id', '?')
        marker = " <-- configured" if mid == configured_model else ""
        print(f"    [{i}] {mid}{marker}")
    if configured_model:
        print(f"    [C] Use configured: {configured_model}")
    print()

    try:
        choice = input("  Select model (number/C) or Enter for configured: ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        print()
        return configured_model

    if not choice or choice == 'C':
        return configured_model

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx].get('id')
    except ValueError:
        pass

    print(f"  Invalid choice: {choice}, using configured model.")
    return configured_model


def print_usage():
    print("Usage: run_benchmark.py [OPTIONS] [CONFIG_PATH]")
    print()
    print("Options:")
    print("  --run       Execute lm_eval directly (otherwise print command)")
    print("  --dry-run   Only show what would be run, don't execute")
    print("  --list      List available configs and exit")
    print("  -h, --help  Show this help")
    print()
    print("If CONFIG_PATH is given, uses that config file.")
    print("Otherwise interactively chooses from discovered configs.")


def build_command(config_path, model_name, model_args):
    """Build the lm_eval command string with all model_args."""
    rel = os.path.relpath(config_path)
    # Build model_args string: all args from config, with model overridden
    args = dict(model_args)
    args['model'] = model_name
    args_str = ','.join(f'{k}={v}' for k, v in args.items())
    return f"lm_eval run --config {rel} --model_args \"{args_str}\""


def main():
    argv = sys.argv[1:]
    do_run = "--run" in argv
    do_dry = "--dry-run" in argv
    do_list = "--list" in argv

    if "--help" in argv or "-h" in argv:
        print_usage()
        sys.exit(0)

    # Find config
    pos_args = [a for a in argv if not a.startswith("-")]

    if do_list:
        configs = find_configs(SCRIPT_DIR)
        if not configs:
            print("No eval_config.yaml files found.")
        else:
            print("Available configs:")
            for name, path in configs:
                print(f"  {name}/  ({path})")
        sys.exit(0)

    config_path = None
    if pos_args:
        config_path = os.path.abspath(pos_args[0])
        if not os.path.isfile(config_path):
            print(f"Config not found: {config_path}")
            sys.exit(1)
    else:
        configs = find_configs(SCRIPT_DIR)
        if not configs:
            print("No eval_config.yaml files found in subdirectories.")
            sys.exit(1)
        if len(configs) == 1:
            config_path = configs[0][1]
            print(f"Using config: {configs[0][0]}/eval_config.yaml")
        else:
            print("Available configs:")
            for i, (name, _) in enumerate(configs, 1):
                print(f"  [{i}] {name}")
            print()
            try:
                choice = input("Select config (number) or Enter for first: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(0)
            if not choice:
                config_path = configs[0][1]
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(configs):
                        config_path = configs[idx][1]
                except ValueError:
                    print(f"Invalid choice: {choice}")
                    sys.exit(1)

    print(f"\nConfig: {config_path}")

    # Read config
    config_text = load_config(config_path)
    base_url = extract_base_url(config_text)
    model_args = extract_model_args(config_text)
    configured_model = model_args.get('model')

    if not base_url:
        print("ERROR: No base_url found in config.")
        sys.exit(1)

    print(f"Server: {base_url}")
    if configured_model:
        print(f"Configured model: {configured_model}")

    # Query server
    origin = get_server_origin(base_url)
    print(f"\nQuerying {origin}/v1/models ...")

    models = query_models(origin)
    model_name = None

    if models is not None:
        print(f"Server responded with {len(models)} model(s).")
        model_name = pick_model(models, configured_model)
    else:
        print("WARNING: Could not query server models endpoint.")
        if configured_model:
            print(f"Falling back to configured model: {configured_model}")
            model_name = configured_model
        else:
            print("ERROR: No configured model and server unreachable.")
            sys.exit(1)

    if not model_name:
        print("ERROR: No model selected.")
        sys.exit(1)

    print(f"\nModel: {model_name}")

    # Build command
    cmd = build_command(config_path, model_name, model_args)

    if do_dry:
        print(f"\n[DRY RUN] Would run:")
        print(f"  {cmd}")
        sys.exit(0)

    if do_run:
        print(f"\nRunning:")
        print(f"  {cmd}")
        print()
        try:
            result = subprocess.run(cmd, shell=True)
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(130)
    else:
        print(f"\nCommand to run:")
        print(f"  {cmd}")
        print()
        try:
            answer = input("Run now? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if answer in ('y', 'yes'):
            print()
            try:
                result = subprocess.run(cmd, shell=True)
                sys.exit(result.returncode)
            except KeyboardInterrupt:
                print("\nInterrupted.")
                sys.exit(130)
        else:
            print("Command printed above. Run it manually when ready.")


if __name__ == "__main__":
    main()
