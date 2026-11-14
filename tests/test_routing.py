from __future__ import annotations

from claim_photo_extraction.foundry import disabled_model_call, skipped_model_call
from claim_photo_extraction.models import (
    AdmissionDecision,
    AdmissionRecord,
    ContractFinding,
    ContractResult,
    DeclaredView,
    ModelCall,
    RoutingDecision,
)
from claim_photo_extraction.routing import route


def _admitted(photo_id: str, flags: tuple[str, ...] = ()) -> AdmissionRecord:
    decision = AdmissionDecision.ADMIT_FLAGGED if flags else AdmissionDecision.ADMIT
    return AdmissionRecord(photo_id, DeclaredView.FRONT, 10, decision, (), flags, None)


def _ok_call() -> ModelCall:
    return ModelCall(True, True, "replay", "gpt-5-mini", "{}", {}, None, None, None)


def _contract(**overrides) -> ContractResult:
    base = dict(
        valid_schema=True, findings=(), confirmed_damages=({"part": "front_bumper"},), unconfirmed_damages=(),
        observed_zones=("front",), uncovered_reported_zones=(), unreported_observed_zones=(),
        overall_severity="minor", identity_flags=(),
    )
    base.update(overrides)
    return ContractResult(**base)


def test_integrity_flags_route_to_investigate_before_anything_else() -> None:
    routing = route((_admitted("P1", ("editing_software_detected",)), _admitted("P2")), _ok_call(), _contract(), 2)
    assert routing.decision is RoutingDecision.INVESTIGATE
    assert routing.reasons == ("P1:editing_software_detected",)


def test_identity_flags_from_contract_route_to_investigate() -> None:
    routing = route((_admitted("P1"), _admitted("P2")), _ok_call(), _contract(identity_flags=("plate_mismatch",)), 2)
    assert routing.decision is RoutingDecision.INVESTIGATE


def test_too_few_admitted_photos_requests_resubmission() -> None:
    routing = route((_admitted("P1"),), skipped_model_call("only 1"), None, 2)
    assert routing.decision is RoutingDecision.RESUBMIT_PHOTOS


def test_model_unavailable_falls_back_to_adjuster() -> None:
    failed = ModelCall(True, True, "foundry", "gpt-5-mini", None, None, None, None, "Foundry HTTP 429: rate limit")
    routing = route((_admitted("P1"), _admitted("P2")), failed, None, 2)
    assert routing.decision is RoutingDecision.ADJUSTER_REVIEW
    assert routing.reasons[0].startswith("model_unavailable")


def test_disabled_model_is_adjuster_review_not_fast_track() -> None:
    routing = route((_admitted("P1"), _admitted("P2")), disabled_model_call(), None, 2)
    assert routing.decision is RoutingDecision.ADJUSTER_REVIEW


def test_refusal_routes_to_adjuster() -> None:
    refusal = ModelCall(True, True, "foundry", "gpt-5-mini", None, None, "I cannot help with that.", None, None)
    routing = route((_admitted("P1"), _admitted("P2")), refusal, None, 2)
    assert routing.reasons == ("model_refusal",)


def test_uncovered_zone_requests_more_photos() -> None:
    routing = route((_admitted("P1"), _admitted("P2")), _ok_call(), _contract(uncovered_reported_zones=("left",)), 2)
    assert routing.decision is RoutingDecision.RESUBMIT_PHOTOS


def test_escalation_findings_route_to_adjuster() -> None:
    contract = _contract(findings=(ContractFinding("airbag_deployed", "escalate", "x", "P1"),))
    routing = route((_admitted("P1"), _admitted("P2")), _ok_call(), contract, 2)
    assert routing.decision is RoutingDecision.ADJUSTER_REVIEW
    assert routing.reasons == ("airbag_deployed",)


def test_clean_confirmed_damage_fast_tracks() -> None:
    routing = route((_admitted("P1"), _admitted("P2", ("capture_time_missing",))), _ok_call(), _contract(), 2)
    assert routing.decision is RoutingDecision.FAST_TRACK
