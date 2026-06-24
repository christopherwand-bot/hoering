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
class PositionBlock:
    label: str
    actor_count: int
    political_weight: int
    key_actors: list[str]


@dataclass
class ActorAssessment:
    actor: str
    actor_type: str
    normalized_actor_type: str
    primary_position: str
    tags: list[str]
    main_argument: str
    concrete_request: str
    relevance_for_client: str
    relationship_to_client: str
    short_quote: str
    source_url: str
    political_weight: int
    power_label: str
    themes: list[str]


@dataclass
class ThemeCard:
    theme: str
    supports: list[str]
    disagrees: list[str]
    typical_arguments: list[str]
    political_risk: str
    client_message: str


@dataclass
class CustomerAdvice:
    customer_type: str
    top_priority_actors: list[str]
    actors_to_contact: list[str]
    suggested_arguments: list[str]
    likely_allies: list[str]
    likely_opponents: list[str]
    objections_to_prepare: list[str]
    recommended_actions: list[str]


@dataclass
class AnalysisResult:
    metadata: HearingMetadata
    responses: list[HearingResponse]
    groups: list[GroupSummary]
    political_directions: list[PoliticalDirection]
    actor_assessments: list[ActorAssessment]
    position_matrix: list[PositionBlock]
    theme_cards: list[ThemeCard]
    summary_points: list[str]
    political_main_picture: str
    argument_map: list[str]
    possible_allies: list[str]
    possible_opponents: list[str]
    recommended_follow_up: list[str]
    executive_summary: str
    customer_advice: CustomerAdvice
    similarity_pairs: list[dict[str, Any]]
    stance_counts: dict[str, int]
    source_breakdown: dict[str, int]
    errors: list[str]
