from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .admission import ImageAdmissionGate
from .config import AppConfig
from .consensus import aggregate_samples
from .contract import ObservationContractGate
from .foundry import VisionModelClient, disabled_model_call, skipped_model_call
from .models import Claim, ContractResult, ModelCall, RunReport
from .prepare import budget_images, prepare_image, write_prepared_copy
from .routing import route


class ClaimPhotoEngine:
    """Admission -> preparation and budgeting -> model observation -> contract -> routing."""

    def __init__(self, config: AppConfig, model_client: VisionModelClient | None = None) -> None:
        self._config = config
        self._model_client = model_client
        self._admission = ImageAdmissionGate(config.admission)
        self._contract = ObservationContractGate(config.contract)

    def run(self, claim: Claim, now_utc: datetime | None = None, sent_copies_dir: Path | None = None) -> RunReport:
        now = (now_utc or datetime.now(UTC)).astimezone(UTC)
        admission = self._admission.evaluate_claim(claim)
        admitted = {record.photo_id: record for record in admission if record.is_admitted}
        photos_by_id = {photo.photo_id: photo for photo in claim.photos}

        prepared = [prepare_image(photos_by_id[photo_id], self._config.preparation) for photo_id in admitted]
        included, budget = budget_images(prepared, admitted, self._config.preparation, self._config.foundry.image_detail)
        if sent_copies_dir is not None:
            for image in included:
                write_prepared_copy(image, sent_copies_dir)

        contract: ContractResult | None = None
        if self._model_client is None:
            model_call = disabled_model_call()
        elif len(included) < self._config.contract.min_admitted_photos:
            model_call = skipped_model_call(
                f"only {len(included)} photo(s) admitted; minimum is {self._config.contract.min_admitted_photos}"
            )
        else:
            model_call = self._observe_with_consensus(claim, included)
            if model_call.parsed_output is not None:
                contract = self._contract.evaluate(
                    model_call.parsed_output, claim, list(budget.included_photo_ids), agreement=model_call.agreement
                )

        routing = route(admission, model_call, contract, self._config.contract.min_admitted_photos)
        return RunReport(
            claim_id=claim.claim_id,
            run_at_utc=now,
            admission=admission,
            budget=budget,
            model_call=model_call,
            contract=contract,
            routing=routing,
        )

    def _observe_with_consensus(self, claim: Claim, images: list) -> ModelCall:
        """Ask for N independent samples and keep what a majority agree on."""
        assert self._model_client is not None
        wanted = self._config.foundry.samples
        calls = [self._model_client.observe(claim, images) for _ in range(wanted)]
        good = [call for call in calls if call.parsed_output is not None]
        usage = _sum_usage([call.usage for call in calls if call.usage])

        if not good:
            first = calls[0]
            refusal = next((call.refusal for call in calls if call.refusal), None)
            return ModelCall(
                enabled=True, called=True, source=first.source, model=first.model, raw_output=first.raw_output,
                parsed_output=None, refusal=refusal, usage=usage, error=first.error if not refusal else None,
                sample_count=len(calls), sample_outputs=(), agreement={},
            )

        min_agreement = min(self._config.contract.min_sample_agreement, len(good))
        output, agreement = aggregate_samples([call.parsed_output for call in good], min_agreement)
        failures = [call.error for call in calls if call.error]
        note = f"{len(failures)} of {len(calls)} samples failed: {failures[0]}" if failures else None
        return ModelCall(
            enabled=True, called=True, source=good[0].source, model=good[0].model,
            raw_output=good[0].raw_output if len(calls) == 1 else None,
            parsed_output=output, refusal=None, usage=usage, error=note,
            sample_count=len(calls), sample_outputs=tuple(call.parsed_output for call in good), agreement=agreement,
        )


def _sum_usage(usages: list[dict]) -> dict | None:
    if not usages:
        return None
    if len(usages) == 1:
        return usages[0]
    total: dict = {"samples": len(usages)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = sum(int(usage.get(key, 0) or 0) for usage in usages)
    reasoning = [int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0) for usage in usages]
    if any(reasoning):
        total["completion_tokens_details"] = {"reasoning_tokens": sum(reasoning)}
    return total
