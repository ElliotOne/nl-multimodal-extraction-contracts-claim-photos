"""Render a run report as a self-contained HTML page for human review."""

from __future__ import annotations

import base64
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .models import AdmissionDecision, Claim, RoutingDecision, RunReport

THUMB_SIZE = 420

DECISION_LABEL = {
    AdmissionDecision.ADMIT: ("Admitted", "ok"),
    AdmissionDecision.ADMIT_FLAGGED: ("Admitted with flags", "warn"),
    AdmissionDecision.REJECT: ("Rejected", "bad"),
}

ROUTING_LABEL = {
    RoutingDecision.FAST_TRACK: ("Fast track", "ok", "Clean claim: confirmed damage only in reported zones, identity consistent."),
    RoutingDecision.ADJUSTER_REVIEW: ("Adjuster review", "warn", "A person must look before anything is paid."),
    RoutingDecision.RESUBMIT_PHOTOS: ("Resubmit photos", "info", "The evidence is not sufficient yet. Ask the claimant for more or better photos."),
    RoutingDecision.INVESTIGATE: ("Investigate", "bad", "An integrity signal fired. Route to the fraud or special investigations queue."),
}

REASON_TEXT = {
    "blur_detected": "image is too blurry",
    "resolution_too_low": "image is too small",
    "underexposed": "image is too dark",
    "overexposed": "image is too bright",
    "format_not_allowed": "file format not allowed",
    "file_too_large": "file exceeds the size limit",
    "file_missing": "file not found",
    "unreadable_image": "file could not be decoded",
    "capture_time_missing": "no capture date in the file (common after messaging apps)",
    "capture_time_outside_window": "capture date is outside the incident window",
    "editing_software_detected": "file was saved by an image editor",
    "gps_metadata_present": "GPS location found and stripped before upload",
    "orientation_normalised": "photo was stored rotated and has been turned upright",
}

STAGE_HELP = [
    ("1. Admission", "Code opens each photo and measures format, size, sharpness, exposure, duplicates, and EXIF. Bad photos stop here."),
    ("2. Prepare and budget", "Admitted photos are turned upright, shrunk, stripped of metadata, and priced in image tokens against a budget."),
    ("3. Foundry vision model", "Only budgeted photos are sent. The model must answer in a strict JSON shape. It is not told the plate number."),
    ("4. Contract gate", "Code checks the answer: right photo ids, confidence in range, plate and colour match, reported versus observed damage zones."),
    ("5. Routing", "A fixed table turns the findings into one decision. The model never decides this."),
]


def render_html(report: RunReport, claim: Claim) -> str:
    photos_by_id = {photo.photo_id: photo for photo in claim.photos}
    data = report.to_dict()
    routing_label, routing_tone, routing_help = ROUTING_LABEL[report.routing.decision]

    parts: list[str] = []
    parts.append(_head(f"Claim photo gate: {claim.claim_id}"))
    parts.append('<body><main class="wrap">')
    parts.append(_header(report, claim))
    parts.append(_routing_banner(routing_label, routing_tone, routing_help, report.routing.reasons))
    parts.append(_pipeline_strip(report))
    parts.append(_photos_section(report, photos_by_id))
    parts.append(_budget_section(report))
    parts.append(_model_section(report))
    parts.append(_contract_section(report))
    parts.append(_raw_section(data))
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_html_report(report: RunReport, claim: Claim, json_report_path: Path) -> Path:
    target = json_report_path.with_suffix(".html")
    target.write_text(render_html(report, claim), encoding="utf-8")
    return target


# --- sections ---------------------------------------------------------------


def _head(title: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#0f1420; --card:#171d2b; --line:#2a3345; --text:#e5e7eb; --muted:#9aa3b5;
  --ok:#10b981; --warn:#f59e0b; --bad:#e32a5b; --info:#6953f9; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, Segoe UI, Roboto, sans-serif }}
.wrap {{ max-width:1180px; margin:0 auto; padding:24px 16px 64px }}
h1 {{ font-size:26px; margin:0 0 4px }} h2 {{ font-size:20px; margin:32px 0 12px; border-bottom:1px solid var(--line); padding-bottom:6px }}
h3 {{ font-size:16px; margin:0 0 6px }}
.muted {{ color:var(--muted) }} .small {{ font-size:13px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fill, minmax(300px,1fr)) }}
.banner {{ border-radius:14px; padding:20px 24px; margin:18px 0; border:2px solid }}
.banner h2 {{ border:0; margin:0 0 6px; padding:0; font-size:28px }}
.tone-ok {{ border-color:var(--ok) }} .tone-warn {{ border-color:var(--warn) }} .tone-bad {{ border-color:var(--bad) }} .tone-info {{ border-color:var(--info) }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; letter-spacing:.3px }}
.badge-ok {{ background:rgba(16,185,129,.18); color:#6ee7b7 }} .badge-warn {{ background:rgba(245,158,11,.18); color:#fcd34d }}
.badge-bad {{ background:rgba(227,42,91,.18); color:#fda4af }} .badge-info {{ background:rgba(105,83,249,.2); color:#c4b5fd }}
.strip {{ display:grid; gap:10px; grid-template-columns:repeat(5,1fr) }}
.stage {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px }}
.stage b {{ display:block; margin-bottom:4px }} .stage .n {{ font-size:22px; font-weight:700 }}
img.thumb {{ width:100%; height:auto; border-radius:8px; border:1px solid var(--line); background:#000 }}
.photo-card.bad img.thumb {{ opacity:.45; filter:grayscale(1) }}
ul {{ margin:6px 0 0 18px; padding:0 }} li {{ margin:2px 0 }}
table {{ width:100%; border-collapse:collapse; font-size:14px }} th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top }}
th {{ color:var(--muted); font-weight:600 }}
.facts {{ display:grid; grid-template-columns:1fr 1fr; gap:2px 12px; font-size:13px; margin-top:8px }}
.facts span:first-child {{ color:var(--muted) }}
.bar {{ height:12px; background:#0b0f18; border:1px solid var(--line); border-radius:999px; overflow:hidden; margin:8px 0 }}
.bar i {{ display:block; height:100%; background:var(--info) }}
details {{ margin-top:12px }} summary {{ cursor:pointer; color:var(--muted) }}
pre {{ background:#0b0f18; border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; font-size:12px }}
@media (max-width:760px) {{ .strip {{ grid-template-columns:1fr }} }}
</style></head>"""


def _header(report: RunReport, claim: Claim) -> str:
    v = claim.vehicle
    zones = ", ".join(z.value for z in claim.reported_impact_zones) or "none"
    return f"""<header>
<h1>Claim photo gate report</h1>
<div class="muted">Claim <b>{html.escape(claim.claim_id)}</b> · run {html.escape(report.run_at_utc.strftime('%Y-%m-%d %H:%M UTC'))}</div>
<div class="card" style="margin-top:14px">
<div class="facts" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">
<div><span class="muted">Vehicle</span><br>{v.year} {html.escape(v.make)} {html.escape(v.model)}, {html.escape(v.color)}</div>
<div><span class="muted">Policy plate</span><br>{html.escape(v.plate)}</div>
<div><span class="muted">Incident</span><br>{claim.incident_at_utc.strftime('%Y-%m-%d %H:%M UTC')}</div>
<div><span class="muted">Reported impact zones</span><br>{html.escape(zones)}</div>
</div>
<p class="small muted" style="margin:12px 0 0">Claimant statement: {html.escape(claim.claimant_statement)}</p>
</div></header>"""


def _routing_banner(label: str, tone: str, help_text: str, reasons: tuple[str, ...]) -> str:
    items = "".join(f"<li><code>{html.escape(r)}</code></li>" for r in reasons)
    return f"""<section class="banner tone-{tone}">
<div class="muted small">Decision</div>
<h2>{html.escape(label)}</h2>
<div>{html.escape(help_text)}</div>
<div class="small muted" style="margin-top:8px">Because:</div><ul>{items}</ul>
</section>"""


def _pipeline_strip(report: RunReport) -> str:
    admitted = sum(1 for r in report.admission if r.is_admitted)
    rejected = len(report.admission) - admitted
    sent = len(report.budget.included_photo_ids)
    model = "called" if report.model_call.called else "not called"
    if report.model_call.error:
        model = "error"
    contract = "—"
    if report.contract is not None:
        contract = f"{len(report.contract.confirmed_damages)} confirmed / {len(report.contract.unconfirmed_damages)} unconfirmed"
    numbers = [
        f"{admitted} admitted, {rejected} rejected",
        f"{sent} sent · {report.budget.used_image_tokens}/{report.budget.max_image_tokens} tokens",
        model,
        contract,
        ROUTING_LABEL[report.routing.decision][0],
    ]
    cells = "".join(
        f'<div class="stage"><b>{html.escape(name)}</b><div class="n">{html.escape(num)}</div><div class="small muted">{html.escape(desc)}</div></div>'
        for (name, desc), num in zip(STAGE_HELP, numbers)
    )
    return f"<h2>What happened, stage by stage</h2><div class='strip'>{cells}</div>"


def _photos_section(report: RunReport, photos_by_id: dict[str, Any]) -> str:
    cards = []
    sent = set(report.budget.included_photo_ids)
    for record in report.admission:
        label, tone = DECISION_LABEL[record.decision]
        photo = photos_by_id.get(record.photo_id)
        thumb = _thumbnail(photo.path) if photo is not None else None
        img = f'<img class="thumb" alt="{html.escape(record.photo_id)}" src="data:image/jpeg;base64,{thumb}">' if thumb else '<div class="muted">(file not available)</div>'
        notes = [f"<li><b>Rejected:</b> {html.escape(_explain(r))}</li>" for r in record.reasons]
        notes += [f"<li><b>Flag:</b> {html.escape(_explain(f))}</li>" for f in record.flags]
        if record.is_admitted and record.photo_id not in sent:
            notes.append("<li><b>Not sent:</b> excluded by the image token budget</li>")
        facts = ""
        if record.facts is not None:
            f = record.facts
            captured = f.captured_at_utc.strftime("%Y-%m-%d %H:%M") if f.captured_at_utc else "missing"
            facts = f"""<div class="facts">
<span>Size</span><span>{f.width}×{f.height} px, {f.file_bytes // 1024} KB {html.escape(f.format)}</span>
<span>Sharpness</span><span>{f.sharpness_score:.0f}</span>
<span>Brightness</span><span>{f.mean_luminance:.0f} / 255</span>
<span>Captured</span><span>{html.escape(captured)}</span>
<span>Software</span><span>{html.escape(f.software or 'none')}</span>
<span>GPS in file</span><span>{'yes' if f.has_gps else 'no'}</span>
</div>"""
        cards.append(
            f"""<div class="card photo-card {tone}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
<h3>{html.escape(record.photo_id)} <span class="muted small">· declared {html.escape(record.declared_view.value)}</span></h3>
<span class="badge badge-{tone}">{html.escape(label)}</span></div>
{img}
<ul>{''.join(notes) or '<li class="muted">No issues found.</li>'}</ul>
{facts}
</div>"""
        )
    return f"<h2>Photos and admission decisions</h2><p class='muted small'>Each photo is measured by code before anything is uploaded. Greyed-out photos were never sent to the model.</p><div class='grid'>{''.join(cards)}</div>"


def _budget_section(report: RunReport) -> str:
    b = report.budget
    pct = min(100, round(100 * b.used_image_tokens / max(1, b.max_image_tokens)))
    excluded = "".join(f"<li>{html.escape(e['photo_id'])}: {html.escape(e['reason'])}</li>" for e in b.excluded) or "<li class='muted'>none</li>"
    return f"""<h2>Image token budget</h2><div class="card">
<div>Estimated <b>{b.used_image_tokens}</b> of <b>{b.max_image_tokens}</b> image tokens ({html.escape(b.method)} method, detail {html.escape(b.detail)})</div>
<div class="bar"><i style="width:{pct}%"></i></div>
<div class="small muted">Sent: {html.escape(', '.join(b.included_photo_ids) or 'none')}</div>
<div class="small muted" style="margin-top:6px">Excluded by budget:</div><ul>{excluded}</ul>
</div>"""


def _model_section(report: RunReport) -> str:
    m = report.model_call
    if not m.called:
        return f"<h2>Vision model</h2><div class='card'><span class='badge badge-info'>Not called</span> <span class='muted'>{html.escape(m.error or 'model disabled for this run')}</span></div>"
    if m.error:
        return f"<h2>Vision model</h2><div class='card'><span class='badge badge-bad'>Error</span> {html.escape(m.error)}</div>"
    if m.refusal:
        return f"<h2>Vision model</h2><div class='card'><span class='badge badge-warn'>Refusal</span> {html.escape(m.refusal)}</div>"
    out = m.parsed_output or {}
    usage = m.usage or {}
    reasoning = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) if isinstance(usage, dict) else None
    rows = []
    for p in out.get("photos", []):
        votes = m.agreement.get(p["photo_id"], {}) if m.agreement else {}
        damages = "".join(
            f"<li>{html.escape(d['part'].replace('_', ' '))}: {html.escape(d['damage_type'])}, {html.escape(d['severity'])} <span class='muted'>({d['confidence']:.2f}"
            + (f", {votes.get(d['part'], 1)}/{m.sample_count} samples" if m.sample_count > 1 else "") + ")</span></li>"
            for d in p.get("damages", [])
        ) or "<li class='muted'>none</li>"
        plate = p.get("plate_text") if p.get("plate_legible") else "not legible"
        issues = ", ".join(i for i in p.get("quality_issues", []) if i != "none") or "none"
        rows.append(
            f"<tr><td><b>{html.escape(p['photo_id'])}</b><br><span class='muted small'>{html.escape(p['observed_view'])}</span></td>"
            f"<td>{html.escape(str(plate))}</td><td>{html.escape(str(p.get('body_color') or '—'))}</td><td><ul style='margin:0 0 0 16px'>{damages}</ul></td><td>{html.escape(issues)}</td></tr>"
        )
    usage_line = ""
    if usage:
        usage_line = f"<div class='small muted' style='margin-top:10px'>Usage: {usage.get('prompt_tokens', '?')} prompt tokens, {usage.get('completion_tokens', '?')} completion tokens" + (f", {reasoning} of them reasoning" if reasoning is not None else "") + f". Estimated image tokens before the call: {report.budget.used_image_tokens}.</div>"
    return f"""<h2>What the vision model observed</h2>
<div class="card">
<div><span class="badge badge-ok">Called</span> <span class="muted small">{html.escape(m.source)} · {html.escape(m.model or '')}{(' · consensus of ' + str(m.sample_count) + ' samples') if m.sample_count > 1 else ''}</span>{(' <span class="badge badge-warn">' + html.escape(m.error) + '</span>') if m.error else ''}</div>
<table style="margin-top:10px"><thead><tr><th>Photo</th><th>Plate read</th><th>Colour</th><th>Damages (confidence)</th><th>Quality issues</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="facts" style="margin-top:12px;grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">
<div><span class="muted">Same vehicle across photos</span><br>{html.escape(str(out.get('same_vehicle_across_photos')))}</div>
<div><span class="muted">Overall severity</span><br>{html.escape(str(out.get('overall_severity')))}</div>
<div><span class="muted">Drivable</span><br>{html.escape(str(out.get('drivable_assessment')))}</div>
</div>
<p class="small" style="margin:10px 0 0">Model notes: {html.escape(str(out.get('notes', '')))}</p>
{usage_line}
</div>"""


def _contract_section(report: RunReport) -> str:
    c = report.contract
    if c is None:
        return "<h2>Contract checks</h2><div class='card muted'>Not run because the model was not called.</div>"
    tone_for = {"info": "ok", "warn": "warn", "escalate": "warn", "block": "bad"}
    rows = "".join(
        f"<tr><td><span class='badge badge-{tone_for.get(f.severity, 'info')}'>{html.escape(f.severity)}</span></td><td><code>{html.escape(f.code)}</code></td><td>{html.escape(f.detail)}</td><td>{html.escape(f.photo_id or '—')}</td></tr>"
        for f in c.findings
    ) or "<tr><td colspan='4' class='muted'>No findings.</td></tr>"
    zones = f"""<div class="facts" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr));margin-bottom:12px">
<div><span class="muted">Zones with confirmed damage</span><br>{html.escape(', '.join(c.observed_zones) or 'none')}</div>
<div><span class="muted">Reported zones not seen</span><br>{html.escape(', '.join(c.uncovered_reported_zones) or 'none')}</div>
<div><span class="muted">Damage in unreported zones</span><br>{html.escape(', '.join(c.unreported_observed_zones) or 'none')}</div>
<div><span class="muted">Identity flags</span><br>{html.escape(', '.join(c.identity_flags) or 'none')}</div>
</div>"""
    return f"""<h2>Contract checks on the model's answer</h2>
<div class="card">
<div class="small muted" style="margin-bottom:10px">Schema valid: <b>{'yes' if c.valid_schema else 'no'}</b> · confirmed damages: <b>{len(c.confirmed_damages)}</b> · below confidence threshold: <b>{len(c.unconfirmed_damages)}</b></div>
{zones}
<table><thead><tr><th>Severity</th><th>Check</th><th>Detail</th><th>Photo</th></tr></thead><tbody>{rows}</tbody></table>
</div>"""


def _raw_section(data: dict[str, Any]) -> str:
    return f"<details><summary>Raw JSON report</summary><pre>{html.escape(json.dumps(data, indent=2))}</pre></details>"


# --- helpers ----------------------------------------------------------------


def _explain(code: str) -> str:
    if code.startswith("duplicate_of:"):
        return f"same image as {code.split(':', 1)[1]} (resend)"
    return REASON_TEXT.get(code, code.replace("_", " "))


def _thumbnail(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image) or image
            rgb = oriented.convert("RGB")
            rgb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            rgb.save(buffer, format="JPEG", quality=78)
    except OSError:
        return None
    return base64.b64encode(buffer.getvalue()).decode("ascii")
