from __future__ import annotations

from .models import AdmissionRecord, ContractResult, ModelCall, Routing, RoutingDecision

INVESTIGATE_ADMISSION_FLAGS = {"editing_software_detected", "capture_time_outside_window"}


def route(
    admission: tuple[AdmissionRecord, ...],
    model_call: ModelCall,
    contract: ContractResult | None,
    min_admitted_photos: int,
) -> Routing:
    """Deterministic routing table. Order matters: integrity, then evidence, then risk."""
    reasons: list[str] = []
    admitted = [record for record in admission if record.is_admitted]

    for record in admitted:
        for flag in record.flags:
            if flag in INVESTIGATE_ADMISSION_FLAGS:
                reasons.append(f"{record.photo_id}:{flag}")
    if contract is not None:
        reasons.extend(contract.identity_flags)
    if reasons:
        return Routing(RoutingDecision.INVESTIGATE, tuple(dict.fromkeys(reasons)))

    if len(admitted) < min_admitted_photos:
        return Routing(RoutingDecision.RESUBMIT_PHOTOS, (f"admitted_photos={len(admitted)}<{min_admitted_photos}",))

    if model_call.refusal:
        return Routing(RoutingDecision.ADJUSTER_REVIEW, ("model_refusal",))

    if not model_call.called or contract is None or (model_call.error and model_call.parsed_output is None):
        detail = model_call.error or "model_not_called"
        return Routing(RoutingDecision.ADJUSTER_REVIEW, (f"model_unavailable:{detail}",))

    if not contract.valid_schema:
        return Routing(RoutingDecision.ADJUSTER_REVIEW, ("schema_violation",))

    blocking = [finding.code for finding in contract.findings if finding.severity == "block"]
    if blocking:
        if "insufficient_usable_photos" in blocking or "photo_id_set_mismatch" in blocking:
            return Routing(RoutingDecision.RESUBMIT_PHOTOS, tuple(dict.fromkeys(blocking)))
        return Routing(RoutingDecision.ADJUSTER_REVIEW, tuple(dict.fromkeys(blocking)))

    if contract.uncovered_reported_zones:
        return Routing(
            RoutingDecision.RESUBMIT_PHOTOS,
            tuple(f"reported_zone_not_observed:{zone}" for zone in contract.uncovered_reported_zones),
        )

    escalations = [finding.code for finding in contract.findings if finding.severity == "escalate"]
    if escalations:
        return Routing(RoutingDecision.ADJUSTER_REVIEW, tuple(dict.fromkeys(escalations)))

    if not contract.confirmed_damages:
        return Routing(RoutingDecision.ADJUSTER_REVIEW, ("no_confirmed_damage",))

    return Routing(RoutingDecision.FAST_TRACK, ("confirmed_damage_in_reported_zones_only",))
