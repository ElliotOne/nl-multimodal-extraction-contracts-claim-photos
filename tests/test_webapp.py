from __future__ import annotations

from claim_photo_extraction.webapp import run_from_payload, sample_payload


def test_sample_payload_carries_claim_and_photos() -> None:
    sample = sample_payload()
    assert sample["claim"]["vehicle"]["plate"] == "LM64 LVG"
    assert [photo["photo_id"] for photo in sample["photos"]] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert all(photo["data"] for photo in sample["photos"])


def test_run_from_payload_dry_and_replay(tmp_path) -> None:
    sample = sample_payload()
    dry = run_from_payload({"mode": "dry", "claim": sample["claim"], "photos": sample["photos"]}, report_root=tmp_path)
    assert dry["report"]["model_call"]["called"] is False
    assert "Photos and admission decisions" in dry["html"]

    replay = run_from_payload({"mode": "replay", "claim": sample["claim"], "photos": sample["photos"]}, report_root=tmp_path)
    assert replay["report"]["routing"]["decision"] == "fast_track"
    assert "Fast track" in replay["html"]


def test_run_from_payload_rejects_bad_ids(tmp_path) -> None:
    sample = sample_payload()
    sample["photos"][0]["photo_id"] = "../etc"
    try:
        run_from_payload({"mode": "dry", "claim": sample["claim"], "photos": sample["photos"]}, report_root=tmp_path)
    except ValueError as exc:
        assert "invalid photo id" in str(exc)
    else:
        raise AssertionError("expected ValueError")
