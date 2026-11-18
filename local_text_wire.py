"""Local text worker API: request formatting and context-tier choice, never GPU placement."""
from __future__ import annotations
import json
from typing import Any


QWEN38_AUTO_MODEL = "qwen3.8-27b-local"
QWEN38_NATIVE_MODEL = "qwen3.8-27b-native-local"
QWEN38_YARN_MODEL = "qwen3.8-27b-512k-local"
QWEN38_NATIVE_CONTEXT = 262_144
QWEN38_ROUTING_MARGIN = 4_096


def force_glm_max_reasoning(body: bytes | None) -> bytes | None:
    if not body:
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return body
    if not isinstance(payload, dict):
        return body

    payload["reasoning_effort"] = "max"
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    chat_template_kwargs["enable_thinking"] = True
    chat_template_kwargs["thinking"] = True
    payload["chat_template_kwargs"] = chat_template_kwargs
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def force_deepseek_max_reasoning(body: bytes | None) -> bytes | None:
    if not body:
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return body
    if not isinstance(payload, dict):
        return body

    payload["reasoning_effort"] = "max"
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    chat_template_kwargs["enable_thinking"] = True
    payload["chat_template_kwargs"] = chat_template_kwargs
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def force_qwen38_max_reasoning(body: bytes | None) -> bytes | None:
    if not body:
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return body
    if not isinstance(payload, dict):
        return body

    payload["model"] = QWEN38_AUTO_MODEL
    # SGLang's request schema rejects Qwen's model-specific "xhigh" value,
    # while the Qwen template rejects the schema's "max" value. Pass the
    # highest effort through template kwargs and omit the conflicting field.
    payload.pop("reasoning_effort", None)
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    chat_template_kwargs["enable_thinking"] = True
    chat_template_kwargs["preserve_thinking"] = True
    chat_template_kwargs["reasoning_effort"] = "xhigh"
    payload["chat_template_kwargs"] = chat_template_kwargs
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _estimate_qwen38_tokens(value: Any) -> int:
    if isinstance(value, str):
        ascii_count = sum(ord(character) < 128 for character in value)
        return (ascii_count + 3) // 4 + len(value) - ascii_count
    if isinstance(value, list):
        return sum(_estimate_qwen38_tokens(item) for item in value)
    if isinstance(value, dict):
        return 4 + sum(
            _estimate_qwen38_tokens(key) + _estimate_qwen38_tokens(item)
            for key, item in value.items()
            if key not in {"model", "stream", "reasoning_effort"}
        )
    return 1


def qwen38_context_workload(body: bytes | None) -> str:
    try:
        payload = json.loads(body or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return "qwen38_native"
    if not isinstance(payload, dict):
        return "qwen38_native"
    model = str(payload.get("model") or QWEN38_AUTO_MODEL)
    if model == QWEN38_YARN_MODEL:
        return "qwen38_yarn"
    if model == QWEN38_NATIVE_MODEL:
        return "qwen38_native"

    output_budget = payload.get("max_completion_tokens", payload.get("max_tokens", 0))
    try:
        output_tokens = max(0, int(output_budget or 0))
    except (TypeError, ValueError):
        output_tokens = 0
    prompt_tokens = _estimate_qwen38_tokens(payload.get("messages") or [])
    return (
        "qwen38_yarn"
        if prompt_tokens + output_tokens > QWEN38_NATIVE_CONTEXT - QWEN38_ROUTING_MARGIN
        else "qwen38_native"
    )


def generation_body(workload: str, path: str, body: bytes | None) -> bytes | None:
    if path.split("?", 1)[0] != "/v1/chat/completions":
        return body
    adapt = {
        "glm": force_glm_max_reasoning, "deepseek": force_deepseek_max_reasoning,
        "qwen38_native": force_qwen38_max_reasoning, "qwen38_yarn": force_qwen38_max_reasoning,
    }.get(workload)
    return adapt(body) if adapt else body


_CATALOG = {
    "/glm/v1/models": ("glm-5.2-local",),
    "/deepseek/v1/models": ("deepseek-v4-flash-local",),
    "/qwen38/v1/models": (QWEN38_AUTO_MODEL, QWEN38_NATIVE_MODEL, QWEN38_YARN_MODEL),
}


def model_catalog(path: str) -> dict | None:
    names = _CATALOG.get(path)
    if names is None:
        return None
    return {"object": "list", "data": [
        {"id": name, "object": "model", "created": 0, "owned_by": "local"} for name in names
    ]}
