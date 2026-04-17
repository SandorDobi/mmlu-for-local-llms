#!/bin/bash
# =============================================================================
# patch_response_log.sh - Install or revert the JANG response logging patch
# for lm-eval's local-chat-completions backend.
#
# Usage:
#   ./patch_response_log.sh          # Apply patch
#   ./patch_response_log.sh revert   # Revert to original
#
# The patch logs the FULL raw API response to /tmp/lm_eval_responses.jsonl
# Each line contains: {"ts": "...", "raw": {entire server response including reasoning_content, usage, etc.}}
#
# Change log path: JANG_RESPONSE_LOG=/path/to/log.jsonl ./patch_response_log.sh
# =============================================================================

OPENAI_FILE="/Users/draco/.local/pipx/venvs/lm-eval/lib/python3.14/site-packages/lm_eval/models/openai_completions.py"
BACKUP_FILE="${OPENAI_FILE}.orig"
PYCACHE="/Users/draco/.local/pipx/venvs/lm-eval/lib/python3.14/site-packages/lm_eval/models/__pycache__/openai_completions.cpython-314.pyc"

if [ "$1" = "revert" ]; then
    if [ -f "$BACKUP_FILE" ]; then
        cp "$BACKUP_FILE" "$OPENAI_FILE"
        rm -f "$PYCACHE"
        echo "PATCH REVERTED - restored original openai_completions.py"
        echo "Cached bytecode cleared."
    else
        echo "ERROR: No backup found at ${BACKUP_FILE}"
        exit 1
    fi
else
    # Check if already patched
    if grep -q "JANG_PATCH" "$OPENAI_FILE" 2>/dev/null; then
        echo "Patch is already applied."
        exit 0
    fi

    if [ ! -f "$BACKUP_FILE" ]; then
        cp "$OPENAI_FILE" "${OPENAI_FILE}.orig"
        echo "Backup created at ${BACKUP_FILE}"
    fi

    python3 << 'PYEOF'
filepath = "/Users/draco/.local/pipx/venvs/lm-eval/lib/python3.14/site-packages/lm_eval/models/openai_completions.py"

with open(filepath, "r") as f:
    content = f.read()

old = '''    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []'''

new = '''    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        # >>> JANG_PATCH: Log raw API responses >>>
        import json as _jang_json, datetime as _jang_dt, os as _jang_os
        _jang_log = _jang_os.environ.get("JANG_RESPONSE_LOG", "/tmp/lm_eval_responses.jsonl")
        try:
            with open(_jang_log, "a") as _jang_f:
                _jang_f.write(_jang_json.dumps({
                    "ts": _jang_dt.datetime.now().isoformat(),
                    "raw": outputs,
                }, ensure_ascii=False, default=str) + "\\n")
        except Exception:
            pass
        # <<< JANG_PATCH <<<
        res = []'''

first_pos = content.index(old)
second_pos = content.index(old, first_pos + len(old))
content = content[:second_pos] + new + content[second_pos + len(old):]

with open(filepath, "w") as f:
    f.write(content)
print("PATCH APPLIED to LocalChatCompletion.parse_generations")
PYEOF

    rm -f "$PYCACHE"
    echo "Cached bytecode cleared."
    echo ""
    echo "Responses will be logged to: /tmp/lm_eval_responses.jsonl"
    echo "Custom path: JANG_RESPONSE_LOG=/path/to/log.jsonl lm_eval ..."
fi
