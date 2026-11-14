from __future__ import annotations

import argparse
from pathlib import Path

from .audit import write_report
from .claims import load_claim, parse_utc
from .config import load_config
from .engine import ClaimPhotoEngine
from .foundry import FoundryVisionClient, ReplayVisionClient, VisionModelClient
from .html_report import write_html_report
from .models import AdmissionDecision


def main() -> int:
    parser = argparse.ArgumentParser(description="Run claim photo admission, extraction, and contract gates.")
    parser.add_argument("--config", type=Path, default=Path("config/foundry.json"))
    parser.add_argument("--claim", type=Path, default=Path("data/claims/CLM-2026-004471/claim.json"))
    parser.add_argument("--replay", type=Path, help="Recorded chat completion body to replay instead of calling Foundry.")
    parser.add_argument("--skip-foundry", action="store_true", help="Stop after admission and budgeting.")
    parser.add_argument("--now", help="ISO-8601 UTC timestamp for deterministic reports.")
    parser.add_argument("--save-sent", type=Path, help="Directory to write the exact JPEG bytes sent to the model.")
    parser.add_argument("--no-html", action="store_true", help="Skip the HTML review page written beside the JSON report.")
    args = parser.parse_args()

    live = not args.skip_foundry and args.replay is None
    config = load_config(args.config, validate=live)
    if not live:
        config = config.without_foundry()
        config.validate()

    client: VisionModelClient | None
    if args.replay is not None:
        client = ReplayVisionClient(args.replay)
    elif live:
        client = FoundryVisionClient(config.foundry)
    else:
        client = None

    claim = load_claim(args.claim)
    now = parse_utc(args.now) if args.now else None
    report = ClaimPhotoEngine(config, client).run(claim, now_utc=now, sent_copies_dir=args.save_sent)
    report_path = write_report(report, config.report_directory)
    html_path = None if args.no_html else write_html_report(report, claim, report_path)

    admitted = [item for item in report.admission if item.is_admitted]
    flagged = [item for item in admitted if item.decision is AdmissionDecision.ADMIT_FLAGGED]
    rejected = [item for item in report.admission if item.decision is AdmissionDecision.REJECT]

    print(f"Claim: {report.claim_id}")
    print(f"Photos admitted: {len(admitted)} (flagged: {len(flagged)})")
    print(f"Photos rejected: {len(rejected)}")
    print(f"Images sent: {len(report.budget.included_photo_ids)} using {report.budget.used_image_tokens}/{report.budget.max_image_tokens} est. image tokens")
    print(f"Model called: {str(report.model_call.called).lower()} ({report.model_call.source})")
    if report.model_call.error:
        print(f"Model note: {report.model_call.error}")
    if report.contract is not None:
        print(f"Confirmed damages: {len(report.contract.confirmed_damages)} (unconfirmed: {len(report.contract.unconfirmed_damages)})")
    print(f"Routing: {report.routing.decision.value} <- {', '.join(report.routing.reasons)}")
    print(f"Report: {report_path}")
    if html_path is not None:
        print(f"Review page: {html_path}")

    if report.model_call.enabled and report.model_call.error:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
