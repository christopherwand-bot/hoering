from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import AnalysisResult, GroupSummary, HearingMetadata, HearingResponse, PoliticalDirection

NORWEGIAN_STOPWORDS = {
    "at", "av", "blir", "ble", "da", "de", "den", "denne", "dette", "du", "eg", "ein",
    "eit", "eller", "en", "er", "et", "etter", "for", "fra", "frå", "ha", "har", "her",
    "hos", "hun", "hva", "hvor", "i", "ikke", "inn", "ja", "kan", "kom", "med", "me",
    "men", "mot", "må", "ni", "no", "nok", "nå", "og", "om", "opp", "oss", "over", "på",
    "seg", "si", "sin", "skal", "slik", "som", "til", "under", "ut", "var", "ved", "vi",
    "vil", "vore", "vår", "vårt", "ynskjer", "ønsker", "det", "dei", "dei", "dei", "sitt",
    "sine", "høyring", "horing", "høring", "forskrift", "cruiseavgift", "kommunal",
    "forslag", "departementet", "nærings", "fiskeridepartementet", "innspel", "høringssvar",
}
STOPWORDS = set(word.lower() for word in ENGLISH_STOP_WORDS).union(NORWEGIAN_STOPWORDS)

STANCE_RULES = {
    "støtter": ["støttar", "støtter", "positive til", "er positive til", "slutter seg til"],
    "kritiske": ["fraråder", "går imot", "avviser", "er negative til", "motsetter", "bør ikke"],
    "betinget": ["forutsetter", "forutsetning", "dersom", "samtidig", "men", "bør vurderes", "må"],
}

POLITICAL_SIGNAL_SETS = {
    "lokalt selvstyre": ["kommune", "kommunal", "lokal", "lokalsamfunn", "selvstyre", "handlingsrom"],
    "næringsvennlig linje": ["næring", "reiseliv", "verdiskaping", "konkurranse", "bedrift", "besøkende"],
    "miljøstyring": ["miljø", "natur", "utslipp", "bærekraft", "forvaltning", "besøksforvaltning"],
    "avgiftsmoderasjon": ["moderat", "lavere", "avgiftsnivå", "forutsigbar", "belastning", "kostnad"],
    "fellesskapsfinansiering": ["fellesgode", "infrastruktur", "finansiering", "bidra", "avgift", "inntekter"],
    "regelklarhet": ["forskrift", "rammer", "hjemmel", "presisering", "administrativ", "regelverk"],
    "regional fordeling": ["fylkeskommune", "region", "distrikt", "kyst", "havner", "fordeling"],
    "arbeidsplasser": ["arbeidsplasser", "sysselsetting", "arbeid", "ansatte", "sjømannsforbund", "maskinistforbund"],
}


def analyze_hearing(metadata: HearingMetadata, responses: list[HearingResponse], errors: list[str]) -> AnalysisResult:
    usable = [response for response in responses if response.text]
    if not usable:
        return AnalysisResult(
            metadata=metadata,
            responses=responses,
            groups=[],
            political_directions=[],
            similarity_pairs=[],
            stance_counts={},
            source_breakdown=Counter(response.source_kind for response in responses),
            errors=errors or ["Ingen svar kunne analyseres."],
        )

    documents = [response.as_search_text() for response in usable]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
        max_features=2500,
        token_pattern=r"(?u)\b[a-zA-ZæøåÆØÅ][a-zA-ZæøåÆØÅ\-]{1,}\b",
    )
    matrix = vectorizer.fit_transform(documents)
    features = np.array(vectorizer.get_feature_names_out())

    cluster_count = _choose_cluster_count(len(usable))
    labels = np.zeros(len(usable), dtype=int)
    if len(usable) > 1:
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(matrix)
        centroids = kmeans.cluster_centers_
    else:
        centroids = np.asarray(matrix.mean(axis=0))

    similarity = cosine_similarity(matrix)
    groups = _build_groups(usable, labels, features, matrix, centroids, similarity)
    political_directions = _build_political_directions(groups, usable)
    similarity_pairs = _top_similarity_pairs(usable, similarity)
    stance_counts = Counter(group.stance for group in groups)

    return AnalysisResult(
        metadata=metadata,
        responses=responses,
        groups=groups,
        political_directions=political_directions,
        similarity_pairs=similarity_pairs,
        stance_counts=dict(stance_counts),
        source_breakdown=dict(Counter(response.source_kind for response in responses)),
        errors=errors,
    )


def _choose_cluster_count(doc_count: int) -> int:
    if doc_count <= 3:
        return max(1, doc_count)
    return max(2, min(8, int(math.sqrt(doc_count))))


def _build_groups(
    responses: list[HearingResponse],
    labels: np.ndarray,
    features: np.ndarray,
    matrix,
    centroids: np.ndarray,
    similarity: np.ndarray,
) -> list[GroupSummary]:
    grouped: defaultdict[int, list[tuple[int, HearingResponse]]] = defaultdict(list)
    for index, response in enumerate(responses):
        grouped[int(labels[index])].append((index, response))

    groups: list[GroupSummary] = []
    for group_id, members in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        member_indexes = [index for index, _ in members]
        actor_names = [response.actor for _, response in members]
        actor_types = sorted({response.actor_type for _, response in members})
        term_scores = _group_term_scores(group_id, member_indexes, matrix, centroids)
        top_term_indexes = np.argsort(term_scores)[::-1][:6]
        top_terms = [features[index] for index in top_term_indexes if term_scores[index] > 0]
        stance = _detect_group_stance([response for _, response in members])
        summary_points = _summarize_group([response for _, response in members], top_terms)
        label = _label_group(top_terms, stance)
        examples = _group_similarity_examples(member_indexes, responses, similarity)
        groups.append(
            GroupSummary(
                group_id=group_id + 1,
                label=label,
                stance=stance,
                member_count=len(members),
                actor_types=actor_types,
                top_terms=top_terms,
                actor_names=actor_names,
                summary_points=summary_points,
                similarity_examples=examples,
            )
        )
    return groups


def _group_term_scores(group_id: int, member_indexes: list[int], matrix, centroids: np.ndarray) -> np.ndarray:
    if len(member_indexes) == 1:
        return np.asarray(matrix[member_indexes[0]].todense()).ravel()
    if centroids.ndim == 1:
        return np.asarray(centroids).ravel()
    return centroids[group_id]


def _detect_group_stance(responses: list[HearingResponse]) -> str:
    counts = Counter()
    for response in responses:
        text = response.text.lower()
        for stance, phrases in STANCE_RULES.items():
            if any(phrase in text for phrase in phrases):
                counts[stance] += 1
    if not counts:
        return "uklar"
    return counts.most_common(1)[0][0]


def _summarize_group(responses: list[HearingResponse], top_terms: list[str]) -> list[str]:
    combined = " ".join(response.text for response in responses)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", combined) if len(sentence.strip()) > 40]
    ranked: list[tuple[int, str]] = []
    for sentence in sentences:
        score = sum(1 for term in top_terms if term and term.lower() in sentence.lower())
        if score:
            ranked.append((score, sentence))
    top_sentences = [sentence for _, sentence in sorted(ranked, key=lambda item: item[0], reverse=True)[:3]]
    if top_sentences:
        return [_trim_sentence(sentence) for sentence in top_sentences]

    fallback = []
    for response in responses[:3]:
        snippet = _trim_sentence(response.text)
        if snippet:
            fallback.append(snippet)
    return fallback


def _label_group(top_terms: list[str], stance: str) -> str:
    if not top_terms:
        return f"Gruppe med {stance} vurdering"
    return f"{', '.join(top_terms[:3]).title()} ({stance})"


def _group_similarity_examples(member_indexes: list[int], responses: list[HearingResponse], similarity: np.ndarray) -> list[dict[str, str | float]]:
    examples: list[dict[str, str | float]] = []
    for left in member_indexes:
        for right in member_indexes:
            if left >= right:
                continue
            examples.append(
                {
                    "left": responses[left].actor,
                    "right": responses[right].actor,
                    "score": round(float(similarity[left][right]) * 100, 1),
                }
            )
    return sorted(examples, key=lambda item: item["score"], reverse=True)[:3]


def _top_similarity_pairs(responses: list[HearingResponse], similarity: np.ndarray) -> list[dict[str, str | float]]:
    pairs: list[dict[str, str | float]] = []
    for left in range(len(responses)):
        for right in range(left + 1, len(responses)):
            pairs.append(
                {
                    "left": responses[left].actor,
                    "right": responses[right].actor,
                    "score": round(float(similarity[left][right]) * 100, 1),
                }
            )
    return sorted(pairs, key=lambda item: item["score"], reverse=True)[:10]


def _build_political_directions(groups: list[GroupSummary], responses: list[HearingResponse]) -> list[PoliticalDirection]:
    directions: list[PoliticalDirection] = []
    response_lookup = {response.actor: response for response in responses}

    for direction_id, group in enumerate(groups, start=1):
        member_text = " ".join(response_lookup[name].as_search_text() for name in group.actor_names if name in response_lookup)
        themes = _infer_political_themes(member_text, group.top_terms)
        title = _build_direction_title(themes, group.stance)
        description = _build_direction_description(group, themes)
        directions.append(
            PoliticalDirection(
                direction_id=direction_id,
                title=title,
                stance=group.stance,
                member_count=group.member_count,
                themes=themes,
                supported_by=group.actor_names[:8],
                description=description,
                evidence_terms=group.top_terms[:6],
            )
        )

    return directions


def _infer_political_themes(text: str, top_terms: list[str]) -> list[str]:
    lowered = f"{text.lower()} {' '.join(term.lower() for term in top_terms)}"
    scores: list[tuple[int, str]] = []
    for label, keywords in POLITICAL_SIGNAL_SETS.items():
        score = sum(lowered.count(keyword.lower()) for keyword in keywords)
        if score:
            scores.append((score, label))

    if not scores:
        return [term.title() for term in top_terms[:2]] or ["Uspesifisert retning"]

    ordered = [label for _, label in sorted(scores, key=lambda item: item[0], reverse=True)[:3]]
    return ordered


def _build_direction_title(themes: list[str], stance: str) -> str:
    lead = " + ".join(theme.title() for theme in themes[:2]) if themes else "Uspesifisert retning"
    stance_map = {
        "støtter": "med støtte til hovedgrepet",
        "kritiske": "med motstand mot hovedgrepet",
        "betinget": "med betinget støtte",
        "uklar": "med uklar støtte",
    }
    return f"{lead} {stance_map.get(stance, '')}".strip()


def _build_direction_description(group: GroupSummary, themes: list[str]) -> str:
    theme_text = ", ".join(themes[:3]) if themes else "flere ulike hensyn"
    point = group.summary_points[0] if group.summary_points else "Gruppen peker i samme retning, men med ulike begrunnelser."
    return (
        f"Denne retningen samler {group.member_count} svar og vektlegger særlig {theme_text}. "
        f"Typisk argumentasjon i gruppen: {point}"
    )


def _trim_sentence(text: str, max_length: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "…"
