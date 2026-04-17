#!/bin/bash
# =============================================================================
# patch_response_log.sh - Install or revert the JANG patch suite
# for lm-eval's local-chat-completions backend.
#
# Usage:
#   ./patch_response_log.sh          # Apply all patches
#   ./patch_response_log.sh revert   # Revert to original
#
# Patches applied:
#   1. __init__: saves extra model_args (enable_thinking, etc.) to self._jang_extra
#   2. _create_payload: merges self._jang_extra into API request + logs request payload
#   3. parse_generations: logs full raw server response
#
# Log files:
#   /tmp/lm_eval_requests.jsonl  - full request payloads (check if enable_thinking sent)
#   /tmp/lm_eval_responses.jsonl - full raw responses (check reasoning_content)
#
# Custom paths:
#   JANG_REQUEST_LOG=/path/req.jsonl JANG_RESPONSE_LOG=/path/resp.jsonl lm_eval ...
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

# --- Patch 1: __init__ (save extra model_args) ---
old_init = """        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        if self._batch_size > 1:"""

new_init = """        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        # >>> JANG_PATCH: Forward extra model_args to API requests >>>
        self._jang_extra = kwargs
        # <<< JANG_PATCH <<<
        if self._batch_size > 1:"""

if old_init in content:
    content = content.replace(old_init, new_init)
    print("PATCH 1/3: __init__ saving extra model_args")
else:
    print("ERROR: Could not find __init__ block")
    import sys; sys.exit(1)

# --- Patch 2: _create_payload (merge extra + log request) ---
old_payload = """        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }"""

new_payload = """        _jang_payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
            **self._jang_extra,
        }
        # >>> JANG_PATCH: Log request payload >>>
        import json as _jang_json, datetime as _jang_dt, os as _jang_os
        _jang_log = _jang_os.environ.get("JANG_REQUEST_LOG", "/tmp/lm_eval_requests.jsonl")
        try:
            with open(_jang_log, "a") as _jang_f:
                _jang_f.write(_jang_json.dumps({
                    "ts": _jang_dt.datetime.now().isoformat(),
                    "raw": _jang_payload,
                }, ensure_ascii=False, default=str) + "\\n")
        except Exception:
            pass
        # <<< JANG_PATCH <<<
        return _jang_payload"""

if old_payload in content:
    content = content.replace(old_payload, new_payload)
    print("PATCH 2/3: _create_payload merging extra model_args + logging request")
else:
    print("ERROR: Could not find _create_payload return block")
    import sys; sys.exit(1)

# --- Patch 3: parse_generations (log response, LocalChatCompletion only) ---
old_parse = """    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []"""

new_parse = """    @staticmethod
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
        res = []"""

first_pos = content.index(old_parse)
second_pos = content.index(old_parse, first_pos + len(old_parse))
content = content[:second_pos] + new_parse + content[second_pos + len(old_parse):]
print("PATCH 3/3: parse_generations response logging applied")

with open(filepath, "w") as f:
    f.write(content)
PYEOF

    rm -f "$PYCACHE"
    echo "Cached bytecode cleared."
    echo ""
    echo "Request payloads logged to:  /tmp/lm_eval_requests.jsonl"
    echo "Response payloads logged to:  /tmp/lm_eval_responses.jsonl"
    echo ""
    echo "Custom paths:"
    echo "  JANG_REQUEST_LOG=/path/req.jsonl JANG_RESPONSE_LOG=/path/resp.jsonl lm_eval ..."
fi
