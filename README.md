# nl-multimodal-extraction-contracts

An educational Python project showing how to put deterministic gates around a
Microsoft Foundry vision model when the input is a set of photos rather than text.
The same gates apply to any workflow that accepts photos from people: receipts,
product returns, property inspections, damage claims.

The worked example is motor insurance first notice of loss. A claimant uploads photos of
a damaged vehicle. The system decides which photos are fit to send, what the model
is allowed to say about them, how its structured output is checked against the
policy record, and where the claim goes next. The model observes; code decides.

## Overview

This repository demonstrates a small production-shaped control loop:

1. Load runtime configuration from JSON with `CPX_` environment overrides.
2. Load a claim record with its vehicle details, reported impact zones, and photos.
3. Admit or reject each photo with deterministic image checks before any upload.
4. Normalise orientation, bound resolution, strip EXIF, and estimate image tokens.
5. Pack admitted photos into a per-claim image token budget in priority order.
6. Call a Microsoft Foundry vision deployment with a strict JSON schema response, several times, and keep the consensus.
7. Validate the structured output and cross-check it against the claim record.
8. Route the claim through a fixed table: fast track, adjuster review, resubmit, investigate.
9. Persist a JSON report and an HTML review page with every admission, budget, model, contract, and routing decision.

The vision model supplies observations. The application owns admission, budget,
identity checks, confidence thresholds, zone coverage, and routing.

## What This Project Demonstrates

- Deterministic photo admission: format, size, resolution, exposure, blur, duplicates
- EXIF read by code for capture time, editing software, GPS, and orientation
- Metadata stripped before upload so GPS and device data never reach the model endpoint
- Image token estimation using the documented patch and tile formulas
- A per-claim image budget and the documented 10-image, 20 MB Foundry limits
- Strict structured outputs through the OpenAI-compatible chat completions endpoint
- A contract gate that enforces what the schema cannot: confidence ranges, photo id sets,
  plate and colour agreement with the policy record, reported versus observed damage zones
- Majority-vote consensus across several model samples, with per-part agreement counts
- Corroboration rules before an unreported-zone finding can escalate
- A routing table where integrity flags win over evidence and evidence wins over convenience
- A replay client so the full pipeline runs in tests and demos without credentials
- A local web app for running claims interactively with your own photos

## Prerequisites

- Python 3.11 or later
- A Microsoft Foundry resource with a deployed vision-capable model that supports
  structured outputs, such as `gpt-5.4-mini` (the default, verified live), `gpt-5-mini`, `gpt-5`, `gpt-4.1`, or `gpt-4.1-mini`
- A Foundry OpenAI-compatible endpoint such as
  `https://<resource>.services.ai.azure.com/openai/v1/`
  or an Azure OpenAI v1 endpoint `https://<resource>.openai.azure.com/openai/v1/`
- An API key for the endpoint

## Quick Start

Create the environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the deterministic gates only. No model call is made:

```powershell
claim-photo-gate --config config/dry-run.json --claim data/claims/CLM-2026-004471/claim.json --skip-foundry --now 2026-11-12T09:00:00Z
```

Run the whole pipeline against the replay fixture. Still no network access:

```powershell
claim-photo-gate --config config/dry-run.json --claim data/claims/CLM-2026-004471/claim.json --replay data/replay/CLM-2026-004471.response.json --now 2026-11-12T09:05:00Z --save-sent reports/sent
```

Run with Microsoft Foundry:

```powershell
$env:CPX_FOUNDRY_BASE_URL="https://<resource>.services.ai.azure.com/openai/v1/"
$env:CPX_FOUNDRY_API_KEY="<foundry-api-key>"
$env:CPX_FOUNDRY_MODEL="<your-vision-deployment-name>"

claim-photo-gate --config config/foundry.json --claim data/claims/CLM-2026-004471/claim.json --save-sent reports/sent
```

Each run writes a JSON report under `reports/` and an HTML review page beside it
with the same name. Open the `.html` file in a browser to see every photo, why it
was admitted or rejected, the token budget, what the model observed, the contract
checks, and the routing decision with its reasons. Pass `--no-html` to skip it.
With `--save-sent`, the exact JPEG bytes sent to the model are written as well so a
reviewer can see what the model saw.

Run the web app instead of the CLI:

```powershell
claim-photo-web
```

It opens http://127.0.0.1:8765 in your browser. Load the sample claim or upload your own
photos, fill in the claim record, and press run. Every run goes through the deterministic
gates and then calls your Microsoft Foundry deployment, using the `CPX_FOUNDRY_BASE_URL`,
`CPX_FOUNDRY_API_KEY`, and `CPX_FOUNDRY_MODEL` variables of the server process (set them
in the same terminal before starting the app). The page shows every photo with its
admission decision, the token budget, the model's observations, the contract checks, and
the routing decision. Each run is also written to `reports/` as JSON and HTML.

The sample photos are derived from two real photographs of a crash-damaged Suzuki Swift
published on Wikimedia Commons under CC BY-SA 4.0. See `data/source-photos/ATTRIBUTION.md`.
`scripts/make_sample_photos.py` regenerates the six-photo upload set from them: a wide
shot, a blurred close-up, a resent duplicate, a front view with no EXIF, a rotated close-up
with GPS data, and a thumbnail.

## Configuration

The live Foundry config is explicit:

```json
{
  "foundry": {
    "enabled": true,
    "base_url": "https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/",
    "api_key": "replace-me",
    "model": "gpt-5.4-mini",
    "image_detail": "high",
    "timeout_seconds": 90,
    "max_output_tokens": 4000,
    "reasoning_effort": "medium",
    "samples": 3
  },
  "admission": {
    "allowed_formats": [
      "JPEG",
      "PNG",
      "WEBP"
    ],
    "max_file_megabytes": 20,
    "min_shortest_side_px": 640,
    "min_sharpness_score": 60.0,
    "min_mean_luminance": 40.0,
    "max_mean_luminance": 215.0,
    "duplicate_hamming_threshold": 6,
    "max_days_before_incident": 1,
    "max_days_after_incident": 14,
    "flag_editing_software": true
  },
  "preparation": {
    "max_longest_side_px": 1536,
    "jpeg_quality": 85,
    "token_method": "patch",
    "patch_multiplier": 1.2,
    "patch_budget": 2500,
    "tile_base_tokens": 70,
    "tile_tokens": 140,
    "max_image_tokens_per_claim": 9000,
    "max_images_per_request": 10
  },
  "contract": {
    "min_damage_confidence": 0.6,
    "min_admitted_photos": 2,
    "require_plate_match_when_legible": true,
    "require_color_match": true,
    "min_sample_agreement": 2,
    "unreported_zone_min_confidence": 0.8,
    "unreported_zone_min_severity": "moderate"
  },
  "report_directory": "reports"
}
```

Environment overrides:

```text
CPX_FOUNDRY_ENABLED
CPX_FOUNDRY_BASE_URL
CPX_FOUNDRY_API_KEY
CPX_FOUNDRY_MODEL
CPX_IMAGE_DETAIL
CPX_REASONING_EFFORT
CPX_SAMPLES
CPX_MAX_IMAGE_TOKENS
CPX_TOKEN_METHOD
CPX_PATCH_MULTIPLIER
```

The app accepts a Foundry `/openai/v1/` endpoint, a Foundry project endpoint, or an
Azure OpenAI endpoint and resolves it to the OpenAI-compatible base URL before calling
`chat/completions`.

Token estimation follows the model family. `patch` with a multiplier of `1.2`
and a budget of `2500` patches matches `gpt-5.4-mini` at high detail; `gpt-5-mini`
uses the same multiplier and `gpt-4.1-mini` uses `1.62` with a `6144` budget.
`tile` with base `70` and `140` per tile matches `gpt-5` and `gpt-5.1`; `gpt-4.1`
and `gpt-4o` use `85` and `170`. The estimate is a budgeting tool. The report also
records the usage block the endpoint returns so the two can be compared. Live
`gpt-5.4-mini` runs on the sample claim billed about 3,372 prompt tokens per call,
text included, against an estimate of 5,192 image tokens, so treat the estimate as
an upper bound and calibrate from usage.

## How It Works

### Admission

`ImageAdmissionGate` opens each file with Pillow and records facts: format, pixel
size, file size, a Laplacian-variance sharpness score, mean luminance, a 64-bit
difference hash, EXIF capture time, EXIF software tag, GPS presence, and orientation.

Rejections are hard: `format_not_allowed`, `file_too_large`, `resolution_too_low`,
`underexposed`, `overexposed`, `blur_detected`, `duplicate_of:<photo_id>`,
`file_missing`, `unreadable_image`. Duplicates are resolved in priority order so the
claimant's primary photo survives and the resend is dropped.

Flags admit the photo but carry meaning downstream: `capture_time_missing`,
`capture_time_outside_window`, `editing_software_detected`, `gps_metadata_present`,
`orientation_normalised`. Two of them are integrity signals that route the claim to
investigation regardless of what the model says.

### Preparation and budget

`prepare_image` applies the EXIF orientation, converts to RGB, bounds the longest side,
and re-encodes as JPEG without any EXIF block. `estimate_image_tokens` applies the
documented patch or tile formula to the bounded dimensions. `budget_images` packs
admitted photos by priority until the per-claim image token budget or the per-request
image limit is reached and records every exclusion.

### Model call

`FoundryVisionClient` sends one chat completion with a system prompt, a short claim
context, and one text label plus one `image_url` data URI per photo. The policy plate
and the claimant's identity are deliberately absent from the prompt. The response format
is a strict `json_schema`. Reasoning deployments receive `max_completion_tokens` and
`reasoning_effort`; classic chat deployments receive `max_tokens` and `temperature`.

`ReplayVisionClient` reads a saved chat completion body instead. The checked-in fixture
is hand-authored and labelled as such; it describes the sample photos plausibly so the
contract and routing gates can be exercised offline.

### Consensus

A vision model asked the same question twice will not always name the same parts. With
`foundry.samples` above 1 the engine sends the identical request that many times and
`aggregate_samples` keeps only the parts that at least `contract.min_sample_agreement`
samples reported, averaging their confidence. Severity takes the strict majority when
there is one and otherwise the median, rounding up on an even split, so one sample's
`severe` cannot outvote a `minor` and a `moderate`. The damage type is taken only from samples that agree on the
chosen severity, so type and severity always come from the same observations. Plate readings need the same agreement on the normalised registration, so two samples
reading different plates cannot pass one off as agreed; quality issues are voted the same way; vehicle consistency needs every
sample to agree. The aggregated answer has exactly the schema shape, and per-part vote
counts are kept beside it so the report can show `2/3 samples`. Usage is summed across
samples. One failed sample is tolerated and noted; if every sample fails the claim routes
to an adjuster as model unavailable.

### Contract

`ObservationContractGate` validates the output against the same schema, then checks the
things a schema cannot express: the returned photo ids must equal the sent photo ids, the
confidence must be within 0 and 1, a legible plate must match the policy plate, the body
colour must be compatible with the policy colour, damages below the confidence threshold
are unconfirmed, and reported impact zones are compared with observed zones. Damage in a
zone the claimant did not report escalates only when it is corroborated: at least
`unreported_zone_min_severity` with confidence at or above `unreported_zone_min_confidence`,
or the same part seen in two photos. An uncorroborated minor finding is recorded as a
warning instead, so a single sample's stray scratch does not change the route.

### Routing

`route` is a fixed table. Integrity flags from admission or identity flags from the
contract go to `investigate`. Too few admitted photos or an uncovered reported zone go to
`resubmit_photos`. Model errors, refusals, schema violations, blocking findings,
escalations such as airbag deployment or unreported damage, and claims with no confirmed
damage go to `adjuster_review`. Only clean, confirmed, consistent claims go to `fast_track`.

## Project Structure

```text
config/
  dry-run.json
  foundry.json
data/
  claims/CLM-2026-004471/claim.json
  claims/CLM-2026-004471/photos/*.jpg
  replay/CLM-2026-004471.response.json
  source-photos/                 CC BY-SA 4.0 originals and ATTRIBUTION.md
scripts/
  make_sample_photos.py
src/claim_photo_extraction/
  admission.py     deterministic image facts and admission gate
  audit.py         JSON report writer
  claims.py        claim record loader
  cli.py           command-line entry point
  config.py        typed configuration and environment overrides
  consensus.py     majority-vote aggregation across model samples
  contract.py      schema validation and claim cross-checks
  engine.py        pipeline orchestration
  foundry.py       Foundry vision client and replay client
  html_report.py   self-contained HTML review page renderer
  models.py        dataclasses and enums
  prepare.py       orientation, resize, EXIF strip, token estimation, budgeting
  routing.py       deterministic routing table
  samples.py       sample photo set derivation from the source photos
  schema.py        strict JSON schema, prompt, and message builder
  webapp.py        local web app (standard library HTTP server + single page UI)
tests/
```

## Control Guarantees

- No image bytes leave the process until the admission gate admits the photo.
- EXIF, GPS, and device metadata are read by code and stripped before upload.
- The per-request image count and file size stay within the documented Foundry limits.
- The model is never given the policy plate, so a plate match is a check, not an echo.
- Structured output is validated twice: by the endpoint's strict schema and by local code.
- Confidence thresholds and photo id sets are enforced in code, not in the prompt.
- Integrity flags override model findings in routing.
- The model never fast-tracks a claim; only the routing table does.
- A finding seen by one sample out of three does not reach the contract gate.

## Tests

```powershell
python -m pytest
```

The tests cover configuration and endpoint resolution, every admission outcome, image
preparation and token estimation against documented examples, schema strictness, the
contract cross-checks, consensus aggregation and corroboration, the routing table, request
payload shape with an injected transport, an end-to-end replay run, the HTML review page,
and the web app's run handler. No test requires Foundry credentials or network
access.

## License

MIT. See `LICENSE`.

## Contributing

Open an issue or pull request with a focused change and a test that demonstrates it.
