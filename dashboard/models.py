from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HearingResponse:
    actor: str
    actor_type: str
    source_url: str
    source_kind: str
    text: str = ""
    response_date: str | None = None
    response_type: str | None = None
    title: str | None = None
    source_file_url: str | None = None
    errors: list[str] = field(default_factory=list)

    def as_search_text(self) -> str:
        parts = [self.actor, self.title or "", self.text]
        return " ".join(part for part in parts if part).strip()


@dataclass
class HearingMetadata:
    title: str
    organization: str
    published_date: str | None
    deadline: str | None
    status: str | None
    source_url: str


@dataclass
class GroupSummary:
    group_id: int
    label: str
    stance: str
    member_count: int
    actor_types: list[str]
    top_terms: list[str]
    actor_names: list[str]
    summary_points: list[str]
    similarity_examples: list[dict[str, Any]]


@dataclass
class PoliticalDirection:
    direction_id: int
    title: str
    stance: str
    member_count: int
    themes: list[str]
    supported_by: list[str]
    description: str
    evidence_terms: list[str]


@dataclass
class AnalysisResult:
    metadata: HearingMetadata
    responses: list[HearingResponse]
    groups: list[GroupSummary]
    political_directions: list[PoliticalDirection]
    similarity_pairs: list[dict[str, Any]]
    stance_counts: dict[str, int]
    source_breakdown: dict[str, int]
    errors: list[str]
