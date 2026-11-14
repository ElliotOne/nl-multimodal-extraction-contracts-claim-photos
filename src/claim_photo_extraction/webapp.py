"""Local web application for running the claim photo gate interactively.

Standard library only. Start with ``claim-photo-web`` and open http://127.0.0.1:8765.
The browser posts a claim record and photos as JSON; the server runs the same engine the
CLI uses against the configured Microsoft Foundry deployment and returns the rendered
review page plus the JSON report. Credentials come from the CPX_ environment variables.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import tempfile
import threading
import webbrowser
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .audit import write_report
from .claims import load_claim
from .config import AppConfig, load_config
from .engine import ClaimPhotoEngine
from .foundry import FoundryVisionClient, ReplayVisionClient, VisionModelClient
from .html_report import render_html, write_html_report
from .models import Claim, ClaimPhoto, DamageZone, DeclaredView, Vehicle

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLAIM = ROOT / "data" / "claims" / "CLM-2026-004471" / "claim.json"
SAMPLE_REPLAY = ROOT / "data" / "replay" / "CLM-2026-004471.response.json"
CONFIG_DRY = ROOT / "config" / "dry-run.json"
CONFIG_LIVE = ROOT / "config" / "foundry.json"

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class ClaimPhotoWebHandler(BaseHTTPRequestHandler):
    server_version = "ClaimPhotoGate/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
        elif self.path == "/api/sample":
            self._send_json(200, sample_payload())
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 200 * 1024 * 1024:
            self._send_json(413, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = run_from_payload(payload)
        except Exception as exc:  # the browser shows the message; nothing else is at stake locally
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, result)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"[web] {self.address_string()} {format % args}")

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(data).encode("utf-8"))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def sample_payload() -> dict[str, Any]:
    claim = load_claim(SAMPLE_CLAIM)
    photos = []
    for photo in claim.photos:
        photos.append(
            {
                "photo_id": photo.photo_id,
                "name": photo.path.name,
                "declared_view": photo.declared_view.value,
                "priority": photo.priority,
                "data": base64.b64encode(photo.path.read_bytes()).decode("ascii"),
            }
        )
    return {
        "claim": {
            "claim_id": claim.claim_id,
            "incident_at_utc": claim.incident_at_utc.isoformat().replace("+00:00", "Z"),
            "reported_at_utc": claim.reported_at_utc.isoformat().replace("+00:00", "Z"),
            "vehicle": {
                "year": claim.vehicle.year, "make": claim.vehicle.make, "model": claim.vehicle.model,
                "color": claim.vehicle.color, "plate": claim.vehicle.plate,
            },
            "reported_impact_zones": [zone.value for zone in claim.reported_impact_zones],
            "claimant_statement": claim.claimant_statement,
        },
        "photos": photos,
        "replay_available": SAMPLE_REPLAY.is_file(),
    }


def run_from_payload(payload: dict[str, Any], report_root: Path | None = None) -> dict[str, Any]:
    mode = str(payload.get("mode", "live"))
    claim_data = payload["claim"]
    photos_data = payload.get("photos", [])
    if not photos_data:
        raise ValueError("add at least one photo")

    workdir = Path(tempfile.mkdtemp(prefix="claim-photo-web-"))
    photos: list[ClaimPhoto] = []
    for index, item in enumerate(photos_data, start=1):
        photo_id = str(item.get("photo_id") or f"P{index}")
        if not SAFE_ID.match(photo_id):
            raise ValueError(f"invalid photo id {photo_id!r}")
        suffix = Path(str(item.get("name") or "photo.jpg")).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            suffix = ".bin"
        target = workdir / f"{photo_id}{suffix}"
        target.write_bytes(base64.b64decode(str(item["data"])))
        photos.append(
            ClaimPhoto(
                photo_id=photo_id,
                path=target,
                declared_view=DeclaredView(str(item.get("declared_view", "unknown"))),
                priority=int(item.get("priority", index * 10)),
            )
        )

    vehicle_data = claim_data["vehicle"]
    claim = Claim(
        claim_id=str(claim_data.get("claim_id") or "CLM-WEB"),
        incident_at_utc=_parse(claim_data["incident_at_utc"]),
        reported_at_utc=_parse(claim_data.get("reported_at_utc") or claim_data["incident_at_utc"]),
        vehicle=Vehicle(
            year=int(vehicle_data["year"]), make=str(vehicle_data["make"]), model=str(vehicle_data["model"]),
            color=str(vehicle_data["color"]), plate=str(vehicle_data["plate"]),
        ),
        reported_impact_zones=tuple(DamageZone(str(zone)) for zone in claim_data.get("reported_impact_zones", [])),
        claimant_statement=str(claim_data.get("claimant_statement", "")),
        photos=tuple(photos),
    )

    config, client = _build_runtime(mode, payload.get("foundry") or {})
    report = ClaimPhotoEngine(config, client).run(claim, now_utc=datetime.now(UTC))
    if report_root is None:
        report_root = config.report_directory if config.report_directory.is_absolute() else ROOT / config.report_directory
    report_path = write_report(report, report_root)
    write_html_report(report, claim, report_path)
    page = render_html(report, claim)
    body = page.split("<main class=\"wrap\">", 1)[1].rsplit("</main>", 1)[0]
    return {"html": body, "report": report.to_dict(), "mode": mode, "report_path": str(report_path)}


def _build_runtime(mode: str, foundry: dict[str, Any]) -> tuple[AppConfig, VisionModelClient | None]:
    if mode == "live":
        config = load_config(CONFIG_LIVE, validate=False)
        fc = config.foundry
        fc = replace(
            fc,
            enabled=True,
            base_url=str(foundry.get("base_url") or os.environ.get("CPX_FOUNDRY_BASE_URL") or fc.base_url),
            api_key=str(foundry.get("api_key") or os.environ.get("CPX_FOUNDRY_API_KEY") or fc.api_key),
            model=str(foundry.get("model") or os.environ.get("CPX_FOUNDRY_MODEL") or fc.model),
            image_detail=str(foundry.get("image_detail") or fc.image_detail),
        )
        config = replace(config, foundry=fc)
        config.validate()
        return config, FoundryVisionClient(config.foundry)

    config = load_config(CONFIG_DRY, validate=False).without_foundry()
    config.validate()
    if mode == "replay":
        return config, ReplayVisionClient(SAMPLE_REPLAY)
    return config, None


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the claim photo gate web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ClaimPhotoWebHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Claim photo gate web app running at {url}  (Ctrl+C to stop)")
    missing = [name for name in ("CPX_FOUNDRY_BASE_URL", "CPX_FOUNDRY_API_KEY", "CPX_FOUNDRY_MODEL") if not os.environ.get(name)]
    if missing:
        print("Warning: not set in this environment: " + ", ".join(missing) + ". Runs will fail until they are set (or config/foundry.json is filled in).")
    missing = [name for name in ("CPX_FOUNDRY_BASE_URL", "CPX_FOUNDRY_API_KEY", "CPX_FOUNDRY_MODEL") if not os.environ.get(name)]
    if missing:
        print("Warning: not set in this environment: " + ", ".join(missing) + ". Runs will fail until they are set (or config/foundry.json is filled in).")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


INDEX_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claim Photo Gate</title>
<style>
:root{--bg:#0f1420;--card:#171d2b;--line:#2a3345;--text:#e5e7eb;--muted:#9aa3b5;--ok:#10b981;--warn:#f59e0b;--bad:#e32a5b;--info:#6953f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}h3{font-size:16px;margin:0 0 6px}
.muted{color:var(--muted)}.small{font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.form{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
label{display:block;font-size:13px;color:var(--muted);margin-bottom:4px}
input,select,textarea{width:100%;background:#0b0f18;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font:inherit}
textarea{min-height:70px}
button{background:var(--info);color:#fff;border:0;border-radius:8px;padding:10px 16px;font:600 15px system-ui;cursor:pointer}
button.secondary{background:#263044}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
.zones label{display:inline-flex;align-items:center;gap:6px;margin:0 12px 6px 0;color:var(--text)}
.zones input{width:auto}
.photo{display:grid;grid-template-columns:120px 1fr;gap:10px;align-items:start;background:#0b0f18;border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:8px}
.photo img{width:120px;height:90px;object-fit:cover;border-radius:6px;background:#000}
.banner{border-radius:14px;padding:20px 24px;margin:18px 0;border:2px solid}.banner h2{border:0;margin:0 0 6px;padding:0;font-size:28px}
.tone-ok{border-color:var(--ok)}.tone-warn{border-color:var(--warn)}.tone-bad{border-color:var(--bad)}.tone-info{border-color:var(--info)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
.badge-ok{background:rgba(16,185,129,.18);color:#6ee7b7}.badge-warn{background:rgba(245,158,11,.18);color:#fcd34d}.badge-bad{background:rgba(227,42,91,.18);color:#fda4af}.badge-info{background:rgba(105,83,249,.2);color:#c4b5fd}
.strip{display:grid;gap:10px;grid-template-columns:repeat(5,1fr)}.stage{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.stage b{display:block;margin-bottom:4px}.stage .n{font-size:22px;font-weight:700}
img.thumb{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#000}.photo-card.bad img.thumb{opacity:.45;filter:grayscale(1)}
ul{margin:6px 0 0 18px;padding:0}li{margin:2px 0}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.facts{display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:13px;margin-top:8px}.facts span:first-child{color:var(--muted)}
.bar{height:12px;background:#0b0f18;border:1px solid var(--line);border-radius:999px;overflow:hidden;margin:8px 0}.bar i{display:block;height:100%;background:var(--info)}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted)}pre{background:#0b0f18;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font-size:12px}
#status{margin-left:8px}.err{color:#fda4af}
@media (max-width:760px){.strip{grid-template-columns:1fr}.photo{grid-template-columns:1fr}}
</style></head><body><main class="wrap">
<h1>Claim Photo Gate</h1>
<div class="muted">Upload claim photos, run the deterministic gates and your Foundry vision model, and see every decision.</div>

<h2>1. Claim</h2>
<div class="card">
<div class="form">
<div><label>Claim id</label><input id="claim_id" value="CLM-WEB-0001"></div>
<div><label>Incident time (UTC)</label><input id="incident" type="datetime-local"></div>
<div><label>Year</label><input id="year" type="number" value="2014"></div>
<div><label>Make</label><input id="make" value="Suzuki"></div>
<div><label>Model</label><input id="model" value="Swift"></div>
<div><label>Colour</label><input id="color" value="blue"></div>
<div><label>Policy plate</label><input id="plate" value="LM64 LVG"></div>
</div>
<div style="margin-top:12px"><label>Reported impact zones</label>
<div class="zones" id="zones"></div></div>
<div style="margin-top:8px"><label>Claimant statement</label><textarea id="statement"></textarea></div>
<div class="row"><button class="secondary" id="loadSample">Load sample claim and photos</button><span class="muted small">Two real Commons photos of a damaged Suzuki plus derived variants.</span></div>
</div>

<h2>2. Photos</h2>
<div class="card">
<input type="file" id="files" multiple accept="image/*">
<div id="photos"></div>
<div class="row"><button class="secondary" id="clearPhotos">Remove all photos</button></div>
</div>

<h2>3. Run</h2>
<div class="card">
<p class="small muted" style="margin:0 0 10px">Runs the deterministic gates, then calls your Microsoft Foundry vision deployment. The server reads <code>CPX_FOUNDRY_BASE_URL</code>, <code>CPX_FOUNDRY_API_KEY</code>, and <code>CPX_FOUNDRY_MODEL</code> from its environment.</p>
<div class="row"><button id="run">Run claim photo gate</button><span id="status" class="muted small"></span></div>
</div>

<div id="result"></div>
</main>
<script>
const ZONES=["front","rear","left","right","roof","glass","interior","undercarriage"];
const VIEWS=["front","rear","left_side","right_side","front_left","front_right","rear_left","rear_right","interior","closeup","unknown"];
let photos=[];
const $=id=>document.getElementById(id);
function init(){
  $("zones").innerHTML=ZONES.map(z=>`<label><input type="checkbox" value="${z}" ${z==="front"||z==="left"?"checked":""}> ${z}</label>`).join("");
  const d=new Date(); d.setMinutes(d.getMinutes()-d.getTimezoneOffset()); $("incident").value=d.toISOString().slice(0,16);
  $("files").addEventListener("change",e=>addFiles(e.target.files));
  $("clearPhotos").onclick=()=>{photos=[];renderPhotos();};
  $("loadSample").onclick=loadSample; $("run").onclick=run;
}
function addFiles(list){
  [...list].forEach(f=>{const r=new FileReader();r.onload=()=>{photos.push({photo_id:"P"+(photos.length+1),name:f.name,declared_view:"unknown",priority:(photos.length+1)*10,data:r.result.split(",")[1],mime:f.type||"image/jpeg"});renderPhotos();};r.readAsDataURL(f);});
  $("files").value="";
}
function renderPhotos(){
  $("photos").innerHTML=photos.map((p,i)=>`<div class="photo"><img src="data:${p.mime};base64,${p.data}" alt="${p.photo_id}">
  <div><div class="form"><div><label>Photo id</label><input value="${p.photo_id}" onchange="photos[${i}].photo_id=this.value"></div>
  <div><label>Declared view</label><select onchange="photos[${i}].declared_view=this.value">${VIEWS.map(v=>`<option ${v===p.declared_view?"selected":""}>${v}</option>`).join("")}</select></div>
  <div><label>Priority (lower first)</label><input type="number" value="${p.priority}" onchange="photos[${i}].priority=+this.value"></div>
  <div><label>File</label><div class="small muted" style="padding-top:8px">${p.name}</div></div></div>
  <div class="row"><button class="secondary" onclick="photos.splice(${i},1);renderPhotos()">Remove</button></div></div></div>`).join("")||`<p class="muted small" style="margin:10px 0 0">No photos yet. Choose files or load the sample claim.</p>`;
}
async function loadSample(){
  $("status").textContent="loading sample...";
  const s=await (await fetch("/api/sample")).json();
  $("claim_id").value=s.claim.claim_id; $("incident").value=s.claim.incident_at_utc.slice(0,16);
  $("year").value=s.claim.vehicle.year; $("make").value=s.claim.vehicle.make; $("model").value=s.claim.vehicle.model; $("color").value=s.claim.vehicle.color; $("plate").value=s.claim.vehicle.plate;
  document.querySelectorAll("#zones input").forEach(c=>c.checked=s.claim.reported_impact_zones.includes(c.value));
  $("statement").value=s.claim.claimant_statement;
  photos=s.photos.map(p=>({...p,mime:"image/jpeg"})); renderPhotos();
  $("status").textContent="sample loaded";
}
async function run(){
  const body={mode:"live",claim:{claim_id:$("claim_id").value,incident_at_utc:$("incident").value+":00Z",reported_at_utc:$("incident").value+":00Z",
    vehicle:{year:+$("year").value,make:$("make").value,model:$("model").value,color:$("color").value,plate:$("plate").value},
    reported_impact_zones:[...document.querySelectorAll("#zones input:checked")].map(c=>c.value),claimant_statement:$("statement").value},
    photos:photos.map(({mime,...p})=>p)};
  $("run").disabled=true; $("status").textContent="running gates and calling Foundry...";
  try{
    const r=await fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const data=await r.json();
    if(!r.ok){throw new Error(data.error||r.statusText);}
    $("result").innerHTML=data.html; $("status").textContent="done"; $("result").scrollIntoView({behavior:"smooth"});
  }catch(e){$("status").innerHTML='<span class="err">'+e.message+'</span>';}
  $("run").disabled=false;
}
init();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
