from __future__ import annotations

import json
from dataclasses import replace

from claim_photo_extraction.config import AppConfig
from claim_photo_extraction.foundry import FoundryVisionClient, parse_chat_completion, uses_reasoning_parameters
from claim_photo_extraction.models import Claim
from claim_photo_extraction.prepare import prepare_image


def _live_config(dry_config: AppConfig, model: str) -> AppConfig:
    foundry = replace(
        dry_config.foundry, enabled=True, base_url="https://r.services.ai.azure.com/api/projects/p", api_key="k", model=model
    )
    return replace(dry_config, foundry=foundry)


def test_payload_shape_for_reasoning_model(dry_config: AppConfig, sample_claim: Claim) -> None:
    config = _live_config(dry_config, "gpt-5-mini")
    images = [prepare_image(photo, config.preparation) for photo in sample_claim.photos[:1]]
    payload = FoundryVisionClient(config.foundry).build_payload(sample_claim, images)

    assert payload["model"] == "gpt-5-mini"
    assert payload["max_completion_tokens"] == 4000
    assert payload["reasoning_effort"] == "low"
    assert "temperature" not in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True

    user_content = payload["messages"][1]["content"]
    image_parts = [part for part in user_content if part["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image_parts[0]["image_url"]["detail"] == "high"

    text = " ".join(part["text"] for part in user_content if part["type"] == "text")
    assert "LM64" not in text, "policy plate must not be handed to the model"
    assert "Suzuki" in text and "blue" in text


def test_payload_shape_for_classic_chat_model(dry_config: AppConfig, sample_claim: Claim) -> None:
    config = _live_config(dry_config, "gpt-4.1-mini")
    images = [prepare_image(photo, config.preparation) for photo in sample_claim.photos[:1]]
    payload = FoundryVisionClient(config.foundry).build_payload(sample_claim, images)
    assert payload["max_tokens"] == 4000
    assert payload["temperature"] == 0.0
    assert "reasoning_effort" not in payload


def test_transport_is_called_with_resolved_v1_endpoint(dry_config: AppConfig, sample_claim: Claim) -> None:
    config = _live_config(dry_config, "gpt-5-mini")
    seen: dict[str, object] = {}

    def fake_transport(url: str, body: bytes, headers: dict[str, str], timeout: int) -> tuple[int, str]:
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = json.loads(body)
        return 200, json.dumps(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "{\"x\":1}", "refusal": None}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        )

    images = [prepare_image(photo, config.preparation) for photo in sample_claim.photos[:1]]
    result = FoundryVisionClient(config.foundry, transport=fake_transport).observe(sample_claim, images)
    assert seen["url"] == "https://r.services.ai.azure.com/openai/v1/chat/completions"
    assert seen["headers"]["api-key"] == "k"
    assert result.parsed_output == {"x": 1}
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def test_http_error_becomes_model_error(dry_config: AppConfig, sample_claim: Claim) -> None:
    config = _live_config(dry_config, "gpt-5-mini")

    def failing(url: str, body: bytes, headers: dict[str, str], timeout: int) -> tuple[int, str]:
        return 429, json.dumps({"error": {"code": "429", "message": "Rate limit exceeded"}})

    images = [prepare_image(photo, config.preparation) for photo in sample_claim.photos[:1]]
    result = FoundryVisionClient(config.foundry, transport=failing).observe(sample_claim, images)
    assert result.error is not None and result.error.startswith("Foundry HTTP 429")
    assert result.parsed_output is None


def test_parse_handles_refusal_and_truncation() -> None:
    refusal = parse_chat_completion(
        json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "No."}}]}), model="m", source="t"
    )
    assert refusal.refusal == "No." and refusal.error is None

    truncated = parse_chat_completion(
        json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "{\"photos\": ["}}]}), model="m", source="t"
    )
    assert truncated.error is not None and "finish_reason=length" in truncated.error


def test_reasoning_parameter_detection() -> None:
    assert uses_reasoning_parameters("gpt-5-mini")
    assert uses_reasoning_parameters("GPT-5.1")
    assert uses_reasoning_parameters("o4-mini")
    assert not uses_reasoning_parameters("gpt-4.1-mini")
    assert not uses_reasoning_parameters("gpt-4o")
