from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import AnalysisResult, GroupSummary, HearingMetadata, HearingResponse, PoliticalDirection


def save_analysis(cache_dir: Path, key: str, result: AnalysisResult) -> None:
    payload = asdict(result)
    target = cache_dir / f"{key}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_analysis(cache_dir: Path, key: str) -> AnalysisResult | None:
    target = cache_dir / f"{key}.json"
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    metadata = HearingMetadata(**payload["metadata"])
    responses = [HearingResponse(**item) for item in payload["responses"]]
    groups = [GroupSummary(**item) for item in payload["groups"]]
    political_directions = [PoliticalDirection(**item) for item in payload.get("political_directions", [])]
    return AnalysisResult(
        metadata=metadata,
        responses=responses,
        groups=groups,
        political_directions=political_directions,
        similarity_pairs=payload["similarity_pairs"],
        stance_counts=payload["stance_counts"],
        source_breakdown=payload["source_breakdown"],
        errors=payload["errors"],
    )
