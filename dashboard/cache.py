from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import (
    ActorAssessment,
    AnalysisResult,
    CustomerAdvice,
    GroupSummary,
    HearingMetadata,
    HearingResponse,
    PoliticalDirection,
    PositionBlock,
    ThemeCard,
)


def save_analysis(cache_dir: Path, key: str, result: AnalysisResult) -> None:
    payload = asdict(result)
    target = cache_dir / f"{key}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_analysis(cache_dir: Path, key: str) -> AnalysisResult | None:
    target = cache_dir / f"{key}.json"
    if not target.exists():
        return None

    payload = json.loads(target.read_text(encoding="utf-8"))
    try:
        metadata = HearingMetadata(**payload["metadata"])
        responses = [HearingResponse(**item) for item in payload["responses"]]
        groups = [GroupSummary(**item) for item in payload.get("groups", [])]
        political_directions = [PoliticalDirection(**item) for item in payload.get("political_directions", [])]
        actor_assessments = [ActorAssessment(**item) for item in payload.get("actor_assessments", [])]
        position_matrix = [PositionBlock(**item) for item in payload.get("position_matrix", [])]
        theme_cards = [ThemeCard(**item) for item in payload.get("theme_cards", [])]
        customer_advice = CustomerAdvice(**payload["customer_advice"])
    except Exception:
        return None

    return AnalysisResult(
        metadata=metadata,
        responses=responses,
        groups=groups,
        political_directions=political_directions,
        actor_assessments=actor_assessments,
        position_matrix=position_matrix,
        theme_cards=theme_cards,
        summary_points=payload.get("summary_points", []),
        political_main_picture=payload.get("political_main_picture", ""),
        argument_map=payload.get("argument_map", []),
        possible_allies=payload.get("possible_allies", []),
        possible_opponents=payload.get("possible_opponents", []),
        recommended_follow_up=payload.get("recommended_follow_up", []),
        executive_summary=payload.get("executive_summary", ""),
        customer_advice=customer_advice,
        similarity_pairs=payload.get("similarity_pairs", []),
        stance_counts=payload.get("stance_counts", {}),
        source_breakdown=payload.get("source_breakdown", {}),
        errors=payload.get("errors", []),
    )
