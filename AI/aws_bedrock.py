from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional


class BedrockNotConfigured(RuntimeError):
    pass


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def bedrock_region() -> str:
    return _env("AWS_REGION") or _env("AWS_DEFAULT_REGION") or _env("BEDROCK_REGION")


def bedrock_model_id() -> str:
    # Default Bedrock model id for Llama 3.1 8B Instruct.
    return _env("BEDROCK_MODEL_ID", "meta.llama3-1-8b-instruct-v1:0")


def bedrock_enabled() -> bool:
    prefer = _env("PREFER_LLM_PROVIDER").lower() or _env("LLM_PROVIDER").lower()
    if prefer in {"aws", "bedrock"}:
        return True
    return _env("ENABLE_BEDROCK", "0").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _bedrock_runtime_client():
    try:
        import boto3
    except Exception as e:
        raise BedrockNotConfigured("boto3 is not installed; install boto3 to use Bedrock") from e

    region = bedrock_region()
    if not region:
        raise BedrockNotConfigured("Bedrock region not configured; set AWS_REGION (or BEDROCK_REGION)")

    # Uses standard AWS credential chain (env vars, profile, instance role, etc).
    return boto3.client("bedrock-runtime", region_name=region)


def _llama3_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert OpenAI-style chat messages into Llama 3.x Instruct prompt format.
    """
    parts: List[str] = []
    parts.append("<|begin_of_text|>")

    for msg in messages or []:
        role = (msg.get("role") or "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"

        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>")

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "\n".join(parts)


def _read_body(response: Any) -> bytes:
    body = getattr(response, "get", None)
    if callable(body):
        b = response.get("body")
    else:
        b = response["body"]
    if hasattr(b, "read"):
        return b.read()
    if isinstance(b, (bytes, bytearray)):
        return bytes(b)
    return str(b).encode("utf-8", errors="ignore")


def invoke_llama31_text(
    *,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """
    Invoke Bedrock Llama 3.1 (8B) Instruct via `InvokeModel`.

    Requires:
      - `AWS_REGION` (or `BEDROCK_REGION`)
      - AWS credentials available to boto3
      - `BEDROCK_MODEL_ID` optionally (defaults to Llama 3.1 8B Instruct)
    """
    client = _bedrock_runtime_client()
    model_id = bedrock_model_id()

    prompt = _llama3_messages_to_prompt(messages)
    payload = {
        "prompt": prompt,
        "max_gen_len": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )

    raw = _read_body(resp)
    data = json.loads(raw.decode("utf-8", errors="replace"))
    # Meta Llama responses typically return `generation`
    text = (
        data.get("generation")
        or data.get("output")
        or data.get("completion")
        or ""
    )
    return str(text).strip()

