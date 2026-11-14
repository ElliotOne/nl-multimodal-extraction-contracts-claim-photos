from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import FoundryConfig, resolve_foundry_openai_base_url
from .models import Claim, ModelCall, PreparedImage
from .schema import build_messages, response_format

Transport = Callable[[str, bytes, dict[str, str], int], tuple[int, str]]


class VisionModelClient(Protocol):
    def observe(self, claim: Claim, images: list[PreparedImage]) -> ModelCall: ...


class FoundryVisionClient:
    """Calls a Microsoft Foundry vision deployment through the OpenAI-compatible chat endpoint."""

    def __init__(self, config: FoundryConfig, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or _urllib_transport

    def build_payload(self, claim: Claim, images: list[PreparedImage]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": build_messages(claim, images, self._config.image_detail),
            "response_format": response_format(),
        }
        if uses_reasoning_parameters(self._config.model):
            payload["max_completion_tokens"] = self._config.max_output_tokens
            payload["reasoning_effort"] = self._config.reasoning_effort
        else:
            payload["max_tokens"] = self._config.max_output_tokens
            payload["temperature"] = 0.0
        return payload

    def observe(self, claim: Claim, images: list[PreparedImage]) -> ModelCall:
        base_url = resolve_foundry_openai_base_url(self._config.base_url).rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = self.build_payload(claim, images)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
            "api-key": self._config.api_key,
        }
        try:
            status, body = self._transport(url, json.dumps(payload).encode("utf-8"), headers, self._config.timeout_seconds)
        except HTTPError as exc:
            return _failed(self._config.model, f"Foundry HTTP {exc.code}: {_error_detail(exc)}")
        except URLError as exc:
            return _failed(self._config.model, f"Foundry request failed: {exc.reason}")

        if status >= 400:
            return _failed(self._config.model, f"Foundry HTTP {status}: {body[:600]}")

        return parse_chat_completion(body, model=self._config.model, source="foundry")


class ReplayVisionClient:
    """Replays a recorded or hand-authored chat completion body. No network access."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def observe(self, claim: Claim, images: list[PreparedImage]) -> ModelCall:
        body = self._path.read_text(encoding="utf-8")
        data = json.loads(body)
        model = str(data.get("model") or "replay")
        return parse_chat_completion(body, model=model, source=f"replay:{self._path.name}")


def parse_chat_completion(body: str, *, model: str, source: str) -> ModelCall:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return _failed(model, f"Response body was not JSON: {exc.msg}", source=source)

    choices = data.get("choices") if isinstance(data, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    usage = data.get("usage") if isinstance(data, dict) and isinstance(data.get("usage"), dict) else None
    if not isinstance(message, dict):
        return _failed(model, "Response had no choices[0].message", source=source, usage=usage)

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return ModelCall(
            enabled=True, called=True, source=source, model=model, raw_output=None,
            parsed_output=None, refusal=refusal, usage=usage, error=None,
        )

    content = message.get("content")
    raw = content if isinstance(content, str) else ""
    finish_reason = choices[0].get("finish_reason")
    if finish_reason == "length":
        return _failed(model, "Output truncated: finish_reason=length", source=source, usage=usage, raw=raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _failed(model, f"Structured output was not valid JSON: {exc.msg}", source=source, usage=usage, raw=raw)
    if not isinstance(parsed, dict):
        return _failed(model, "Structured output was not a JSON object", source=source, usage=usage, raw=raw)

    return ModelCall(
        enabled=True, called=True, source=source, model=model, raw_output=raw,
        parsed_output=parsed, refusal=None, usage=usage, error=None,
    )


def disabled_model_call() -> ModelCall:
    return ModelCall(
        enabled=False, called=False, source="disabled", model=None, raw_output=None,
        parsed_output=None, refusal=None, usage=None, error=None,
    )


def skipped_model_call(reason: str) -> ModelCall:
    return ModelCall(
        enabled=True, called=False, source="skipped", model=None, raw_output=None,
        parsed_output=None, refusal=None, usage=None, error=reason,
    )


def uses_reasoning_parameters(model: str) -> bool:
    normalized = model.lower().strip()
    return normalized.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4", "codex"))


def _failed(
    model: str | None,
    error: str,
    *,
    source: str = "foundry",
    usage: dict[str, Any] | None = None,
    raw: str | None = None,
) -> ModelCall:
    return ModelCall(
        enabled=True, called=True, source=source, model=model, raw_output=raw,
        parsed_output=None, refusal=None, usage=usage, error=error,
    )


def _urllib_transport(url: str, body: bytes, headers: dict[str, str], timeout: int) -> tuple[int, str]:
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")


def _error_detail(exc: HTTPError) -> str:
    try:
        text = exc.read(1200).decode("utf-8", errors="replace").strip()
    except OSError:
        return str(exc.reason)
    try:
        data = json.loads(text)
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            code = error.get("code")
            return f"{code}: {error['message']}" if isinstance(code, str) else error["message"]
    except json.JSONDecodeError:
        pass
    return text or str(exc.reason)
