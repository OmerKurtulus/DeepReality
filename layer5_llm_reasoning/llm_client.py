"""
DeepReality — LLM Provider Client
=================================

Thin client for OpenAI-compatible chat-completions endpoints, used by
PIN-E1 to reach the configured reasoning model. OpenRouter is the
default gateway, but any compatible provider works by overriding
`DEEPREALITY_LLM_API_BASE`.

The client is deliberately minimal — no provider SDK is introduced as a
dependency — but it handles the three failure classes that matter in
production: transient network faults (retried with backoff),
unsupported request features (structured-output mode is withdrawn and
the call retried), and malformed model output (fenced or prose-wrapped
JSON is recovered before parsing).
"""

import json
import re
import time

import requests

from config.settings import LLM_CONFIG


class LLMError(RuntimeError):
    """Raised when the reasoning model cannot be reached or understood."""


class LLMNotConfiguredError(LLMError):
    """Raised when no API credential is available."""


# Matches a fenced code block, with or without a language tag, so that
# JSON wrapped in markdown can be recovered rather than rejected.
_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def get_api_key() -> str | None:
    """Return the configured API key, or None when unset."""
    import os

    key = os.environ.get(LLM_CONFIG["api_key_env"], "").strip()
    return key or None


def is_configured() -> bool:
    """True when Layer 5 has the credentials required to run."""
    return get_api_key() is not None


def extract_json_object(text: str) -> dict:
    """
    Recover a JSON object from a model response.

    Attempts, in order: direct parse, fenced-block extraction, and
    outermost brace-span extraction. Models instructed to emit bare
    JSON occasionally add a fence or a leading sentence; discarding an
    otherwise valid response over that would be needlessly brittle.
    """
    text = (text or "").strip()
    if not text:
        raise LLMError("Model returned an empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_PATTERN.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise LLMError(f"Model response was not valid JSON: {text[:200]}")


def complete(system_prompt: str, user_prompt: str) -> dict:
    """
    Submit one adjudication request and return the parsed JSON verdict.

    Args:
        system_prompt: The forensic reasoning protocol.
        user_prompt:   The serialised evidence digest.

    Returns:
        {"content": <parsed dict>, "usage": {...}, "model": str,
         "latency_seconds": float}

    Raises:
        LLMNotConfiguredError: No API key is available.
        LLMError:              The provider or response failed.
    """
    api_key = get_api_key()
    if not api_key:
        raise LLMNotConfiguredError(
            f"{LLM_CONFIG['api_key_env']} is not set. "
            f"Add it to the .env file at the project root."
        )

    endpoint = LLM_CONFIG["api_base"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Attribution headers; ignored by providers that do not use them.
        "HTTP-Referer": LLM_CONFIG["referer"],
        "X-Title": LLM_CONFIG["app_title"],
    }

    payload = {
        "model": LLM_CONFIG["model"],
        "temperature": LLM_CONFIG["temperature"],
        "max_tokens": LLM_CONFIG["max_tokens"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Structured-output mode where the provider supports it; withdrawn
        # automatically on rejection (see below).
        "response_format": {"type": "json_object"},
    }

    last_error: Exception | None = None
    attempts = LLM_CONFIG["max_retries"] + 1

    for attempt in range(attempts):
        try:
            started = time.perf_counter()
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=LLM_CONFIG["timeout_seconds"],
            )
            latency = time.perf_counter() - started

            # A 400 commonly indicates the model rejected response_format.
            # Withdraw it once and retry rather than failing the request.
            if response.status_code == 400 and "response_format" in payload:
                payload.pop("response_format")
                continue

            if response.status_code == 401:
                raise LLMNotConfiguredError(
                    "Provider rejected the API key (HTTP 401). "
                    "Verify the value of "
                    f"{LLM_CONFIG['api_key_env']} in .env."
                )

            if response.status_code != 200:
                raise LLMError(
                    f"Provider returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise LLMError(f"Provider returned no choices: {body}")

            content = choices[0].get("message", {}).get("content", "")
            return {
                "content": extract_json_object(content),
                "usage": body.get("usage", {}),
                "model": body.get("model", LLM_CONFIG["model"]),
                "latency_seconds": round(latency, 2),
            }

        except LLMNotConfiguredError:
            raise  # Credentials will not become valid on retry
        except (requests.RequestException, LLMError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(LLM_CONFIG["retry_backoff_seconds"] * (attempt + 1))

    raise LLMError(f"Reasoning request failed after {attempts} attempts: {last_error}")
