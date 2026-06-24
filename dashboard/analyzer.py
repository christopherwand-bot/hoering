from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

NORWEGIAN_STOPWORDS = {
    "at", "av", "blir", "ble", "da", "de", "den", "denne", "dette", "du", "eg", "ein",
    "eit", "eller", "en", "er", "et", "etter", "for", "fra", "frå", "ha", "har", "her",
    "hos", "hun", "hva", "hvor", "i", "ikke", "inn", "ja", "kan", "kom", "med", "me",
    "men", "mot", "må", "ni", "no", "nok", "nå", "og", "om", "opp", "oss", "over", "på",
    "seg", "si", "sin", "skal", "slik", "som", "til", "under", "ut", "var", "ved", "vi",
    "vil", "vore", "vår", "vårt", "ynskjer", "ønsker", "det", "dei", "høyring", "horing",
    "høring", "forskrift", "forslaget", "cruiseavgift", "forslag", "departementet",
    "nærings", "fiskeridepartementet", "innspel", "høringssvar", "svar", "sak", "dato",
}
STOPWORDS = set(word.lower() for word in ENGLISH_STOP_WORDS).union(NORWEGIAN_STOPWORDS)

POSITION_RULES = {
    "Motsetter seg forslaget": ["motsetter", "går imot", "avviser", "fraråder", "bør ikke vedtas", "sterkt kritisk"],
    "Ønsker strengere regulering": ["strengere", "skjerpes", "høyere avgift", "sterkere regulering", "bør økes"],
    "Ønsker mildere regulering": ["lavere avgift", "moderat avgift", "mildere", "reduseres", "forutsigbar", "bør settes lavere"],
    "Støtter med forbehold": ["støtter", "samtidig", "forutsetter", "dersom", "men", "bør vurderes", "må presiseres"],
    "Støtter forslaget": ["støtter", "positive til", "slutter seg til", "stiller seg bak"],
}

TAG_RULES = {
    "Støtter forslaget": ["støtter", "positive til", "slutter seg til"],
    "Støtter med forbehold": ["forutsetter", "dersom", "men", "må presiseres", "bør vurderes"],
    "Motsetter seg forslaget": ["motsetter", "går imot", "avviser", "fraråder"],
    "Ønsker strengere regulering": ["strengere", "høyere avgift", "skjerpes", "sterkere regulering"],
    "Ønsker mildere regulering": ["lavere avgift", "moderat", "forutsigbar", "mildere"],
    "Opptatt av lokal finansiering": ["vertskommune", "lokalsamfunn", "lokal", "fellesgode", "infrastruktur", "kommunen"],
    "Opptatt av næringshensyn": ["næring", "reiseliv", "verdiskaping", "konkurranse", "kostnad", "bedrift"],
    "Opptatt av miljø/klima": ["miljø", "natur", "utslipp", "bærekraft", "klima", "besøksforvaltning"],
    "Opptatt av juridisk uklarhet": ["hjemmel", "uklar", "presisering", "forskrift", "regelverk", "administrativ"],
}

THEME_DEFINITIONS = {
    "Inntektene bør tilfalle vertskommunen": ["vertskommune", "lokalsamfunn", "kommunen", "lokal finansiering", "fellesgode", "infrastruktur"],
    "Avgiften må være moderat og forutsigbar": ["moderat", "forutsigbar", "lavere avgift", "kostnad", "avgiftsnivå"],
    "Reiselivet må skjermes": ["reiseliv", "verdiskaping", "konkurranse", "bedrift", "næring"],
    "Reglene må bli tydeligere": ["hjemmel", "presisering", "forskrift", "regelverk", "administrativ"],
    "Miljøbelastningen må håndteres strengere": ["miljø", "natur", "utslipp", "bærekraft", "klima"],
}

ACTOR_TYPE_MAP = {
    "kommune": "kommune",
    "fylkeskommune": "kommune",
    "departement": "statlig organ",
    "annen offentlig etat": "statlig organ",
    "bruker- og interesseorganisasjon": "ngo",
    "annen frivillig organisasjon": "ngo",
    "arbeidsgiverorganisasjon": "bransjeorganisasjon",
    "arbeidstakerorganisasjon": "bransjeorganisasjon",
    "privat virksomhet": "næringsliv",
    "privatperson": "privatperson",
}

CLIENT_PRIORITY_RULES = {
    "cruiseaktør": {"allies": {"Opptatt av næringshensyn", "Ønsker mildere regulering"}, "opponents": {"Ønsker strengere regulering"}},
    "kommune": {"allies": {"Opptatt av lokal finansiering", "Støtter forslaget"}, "opponents": {"Ønsker mildere regulering"}},
    "hotell/reiseliv": {"allies": {"Opptatt av næringshensyn", "Ønsker mildere regulering"}, "opponents": {"Ønsker strengere regulering"}},
    "miljøorganisasjon": {"allies": {"Ønsker strengere regulering", "Opptatt av miljø/klima"}, "opponents": {"Ønsker mildere regulering"}},
    "departement": {"allies": {"Støtter forslaget", "Opptatt av juridisk uklarhet"}, "opponents": {"Motsetter seg forslaget"}},
    "havn": {"allies": {"Opptatt av lokal finansiering", "Støtter med forbehold"}, "opponents": {"Motsetter seg forslaget"}},
    "bransjeorganisasjon": {"allies": {"Opptatt av næringshensyn", "Ønsker mildere regulering"}, "opponents": {"Ønsker strengere regulering"}},
}


def analyze_hearing(
    metadata: HearingMetadata,
    responses: list[HearingResponse],
    errors: list[str],
    customer_type: str = "",
) -> AnalysisResult:
    usable = [response for response in responses if response.text]
    if not usable:
        empty_customer = _build_customer_advice(customer_type, [], [], [])
        return AnalysisResult(
            metadata=metadata,
            responses=responses,
            groups=[],
            political_directions=[],
            actor_assessments=[],
            position_matrix=[],
            theme_cards=[],
            summary_points=["Ingen høringssvar kunne analyseres ennå."],
            political_main_picture="Det finnes ikke nok lesbare svar til å tegne et politisk hovedbilde.",
            argument_map=[],
            possible_allies=[],
            possible_opponents=[],
            recommended_follow_up=["Prøv oppdatering på nytt eller kontroller kildegrunnlaget."],
            executive_summary="Det finnes foreløpig ikke nok lesbare svar til å gi kunden et godt råd.",
            customer_advice=empty_customer,
            similarity_pairs=[],
            stance_counts={},
            source_breakdown=dict(Counter(response.source_kind for response in responses)),
            errors=errors or ["Ingen svar kunne analyseres."],
        )

    vectorizer, matrix, features = _vectorize(usable)
    similarity = cosine_similarity(matrix)
    groups = _build_groups(usable, features, matrix, similarity)
    political_directions = _build_political_directions(groups, usable)
    actor_assessments = _build_actor_assessments(usable, similarity, customer_type)
    position_matrix = _build_position_matrix(actor_assessments)
    theme_cards = _build_theme_cards(actor_assessments)
    summary_points = _build_summary_points(actor_assessments, theme_cards, metadata)
    political_main_picture = _build_political_main_picture(actor_assessments, theme_cards, metadata)
    argument_map = _build_argument_map(actor_assessments)
    possible_allies = [actor.actor for actor in actor_assessments if actor.relationship_to_client == "mulig alliert"][:10]
    possible_opponents = [actor.actor for actor in actor_assessments if actor.relationship_to_client == "motstander"][:10]
    recommended_follow_up = _build_recommended_follow_up(actor_assessments, theme_cards, customer_type)
    executive_summary = _build_executive_summary(metadata, actor_assessments, theme_cards, customer_type, recommended_follow_up)
    customer_advice = _build_customer_advice(customer_type, actor_assessments, theme_cards, recommended_follow_up)
    stance_counts = Counter(actor.primary_position for actor in actor_assessments)

    return AnalysisResult(
        metadata=metadata,
        responses=responses,
        groups=groups,
        political_directions=political_directions,
        actor_assessments=actor_assessments,
        position_matrix=position_matrix,
        theme_cards=theme_cards,
        summary_points=summary_points,
        political_main_picture=political_main_picture,
        argument_map=argument_map,
        possible_allies=possible_allies,
        possible_opponents=possible_opponents,
        recommended_follow_up=recommended_follow_up,
        executive_summary=executive_summary,
        customer_advice=customer_advice,
        similarity_pairs=_top_similarity_pairs(usable, similarity),
        stance_counts=dict(stance_counts),
        source_breakdown=dict(Counter(response.source_kind for response in responses)),
        errors=errors,
    )


def _vectorize(responses: list[HearingResponse]):
    documents = [response.as_search_text() for response in responses]
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
    return vectorizer, matrix, features


def _build_groups(
    responses: list[HearingResponse],
    features: np.ndarray,
    matrix,
    similarity: np.ndarray,
) -> list[GroupSummary]:
    if len(responses) == 1:
        labels = np.array([0])
        centroids = np.asarray(matrix.mean(axis=0))
    else:
        cluster_count = max(2, min(8, int(math.sqrt(len(responses)))))
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(matrix)
        centroids = kmeans.cluster_centers_

    grouped: defaultdict[int, list[tuple[int, HearingResponse]]] = defaultdict(list)
    for index, response in enumerate(responses):
        grouped[int(labels[index])].append((index, response))

    groups: list[GroupSummary] = []
    for group_id, members in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        indexes = [index for index, _ in members]
        actor_names = [response.actor for _, response in members]
        actor_types = sorted({response.actor_type for _, response in members})
        term_scores = centroids[group_id] if centroids.ndim > 1 else np.asarray(centroids).ravel()
        top_term_indexes = np.argsort(term_scores)[::-1][:6]
        top_terms = [features[index] for index in top_term_indexes if term_scores[index] > 0]
        stance = _detect_group_stance([response for _, response in members])
        summary_points = _summarize_group([response for _, response in members], top_terms)
        groups.append(
            GroupSummary(
                group_id=group_id + 1,
                label=_friendly_group_label(top_terms, stance),
                stance=stance,
                member_count=len(members),
                actor_types=actor_types,
                top_terms=top_terms,
                actor_names=actor_names,
                summary_points=summary_points,
                similarity_examples=_group_similarity_examples(indexes, responses, similarity),
            )
        )
    return groups


def _friendly_group_label(top_terms: list[str], stance: str) -> str:
    if top_terms:
        return f"Felles spor: {', '.join(top_terms[:3])}"
    return f"Felles spor: {stance}"


def _detect_group_stance(responses: list[HearingResponse]) -> str:
    counts = Counter(_detect_primary_position(response.text) for response in responses)
    return counts.most_common(1)[0][0]


def _summarize_group(responses: list[HearingResponse], top_terms: list[str]) -> list[str]:
    sentences = []
    for response in responses:
        sentences.extend(_sentences(response.text))
    ranked: list[tuple[int, str]] = []
    for sentence in sentences:
        score = sum(1 for term in top_terms if term.lower() in sentence.lower())
        if score:
            ranked.append((score, sentence))
    if ranked:
        return [_trim_sentence(sentence) for _, sentence in sorted(ranked, key=lambda item: item[0], reverse=True)[:3]]
    return [_trim_sentence(response.text) for response in responses[:3] if response.text]


def _build_political_directions(groups: list[GroupSummary], responses: list[HearingResponse]) -> list[PoliticalDirection]:
    lookup = {response.actor: response for response in responses}
    directions: list[PoliticalDirection] = []
    for index, group in enumerate(groups, start=1):
        text = " ".join(lookup[name].as_search_text() for name in group.actor_names if name in lookup)
        themes = _infer_themes_from_text(text)
        directions.append(
            PoliticalDirection(
                direction_id=index,
                title=_direction_title(themes, group.stance),
                stance=group.stance,
                member_count=group.member_count,
                themes=themes[:3],
                supported_by=group.actor_names[:6],
                description=f"Aktørene i denne blokken samler seg rundt {', '.join(themes[:2]).lower() if themes else 'et felles syn'} og uttrykker hovedsakelig {group.stance.lower()}.",
                evidence_terms=group.top_terms[:6],
            )
        )
    return directions


def _build_actor_assessments(
    responses: list[HearingResponse],
    similarity: np.ndarray,
    customer_type: str,
) -> list[ActorAssessment]:
    assessments: list[ActorAssessment] = []
    for index, response in enumerate(responses):
        normalized_type = _normalize_actor_type(response.actor_type)
        tags = _detect_tags(response.text)
        primary_position = _pick_primary_position(tags, response.text)
        themes = _infer_themes_from_text(response.text)
        main_argument = _extract_main_argument(response.text, themes)
        concrete_request = _extract_concrete_request(response.text)
        political_weight = _political_weight(response.actor, normalized_type)
        relevance = _client_relevance(customer_type, tags, normalized_type, political_weight)
        relationship = _relationship_to_client(customer_type, tags, primary_position)
        quote = _extract_short_quote(response.text)
        assessments.append(
            ActorAssessment(
                actor=response.actor,
                actor_type=response.actor_type,
                normalized_actor_type=normalized_type,
                primary_position=primary_position,
                tags=tags,
                main_argument=main_argument,
                concrete_request=concrete_request,
                relevance_for_client=relevance,
                relationship_to_client=relationship,
                short_quote=quote,
                source_url=response.source_url,
                political_weight=political_weight,
                power_label=_power_label(political_weight),
                themes=themes,
            )
        )

    assessments.sort(key=lambda item: (-item.political_weight, _relevance_rank(item.relevance_for_client), item.actor))
    return assessments


def _build_position_matrix(actor_assessments: list[ActorAssessment]) -> list[PositionBlock]:
    buckets: defaultdict[str, list[ActorAssessment]] = defaultdict(list)
    for actor in actor_assessments:
        buckets[actor.primary_position].append(actor)

    blocks = []
    for label, members in sorted(buckets.items(), key=lambda item: sum(actor.political_weight for actor in item[1]), reverse=True):
        blocks.append(
            PositionBlock(
                label=label,
                actor_count=len(members),
                political_weight=sum(actor.political_weight for actor in members),
                key_actors=[actor.actor for actor in members[:5]],
            )
        )
    return blocks


def _build_theme_cards(actor_assessments: list[ActorAssessment]) -> list[ThemeCard]:
    cards: list[ThemeCard] = []
    for theme, keywords in THEME_DEFINITIONS.items():
        supporters = []
        opponents = []
        arguments = []
        for actor in actor_assessments:
            text = " ".join(actor.themes + actor.tags + [actor.main_argument]).lower()
            if any(keyword.lower() in text for keyword in keywords):
                if actor.primary_position == "Motsetter seg forslaget":
                    opponents.append(actor.actor)
                else:
                    supporters.append(actor.actor)
                if actor.main_argument and actor.main_argument not in arguments:
                    arguments.append(actor.main_argument)

        if supporters or opponents:
            cards.append(
                ThemeCard(
                    theme=theme,
                    supports=supporters[:8],
                    disagrees=opponents[:8],
                    typical_arguments=arguments[:3],
                    political_risk=_theme_risk(theme, supporters, opponents),
                    client_message=_theme_message(theme),
                )
            )
    return cards[:6]


def _build_summary_points(
    actor_assessments: list[ActorAssessment],
    theme_cards: list[ThemeCard],
    metadata: HearingMetadata,
) -> list[str]:
    positions = Counter(actor.primary_position for actor in actor_assessments)
    heavyweights = [actor.actor for actor in actor_assessments if actor.political_weight >= 4][:4]
    themes = [card.theme for card in theme_cards[:2]]
    points = [
        f"Flest aktører ligger i kategorien «{positions.most_common(1)[0][0]}».",
        f"Saken dreier seg særlig om {', '.join(theme.lower() for theme in themes)}." if themes else "Svarene peker i flere retninger uten ett klart hovedtema.",
        f"De viktigste aktørene å følge tett er {', '.join(heavyweights)}." if heavyweights else "Det finnes foreløpig ingen tydelig tungvekter i materialet.",
        f"Høringen hos {metadata.organization} utløser både støtte og tydelige forbehold, ikke bare ja/nei.",
        "Kundens handlingsrom avhenger mest av hvem som eier lokal finansiering, næringshensyn og juridisk klarhet.",
    ]
    return points[:5]


def _build_political_main_picture(
    actor_assessments: list[ActorAssessment],
    theme_cards: list[ThemeCard],
    metadata: HearingMetadata,
) -> str:
    top_positions = Counter(actor.primary_position for actor in actor_assessments).most_common(2)
    dominant_themes = ", ".join(card.theme.lower() for card in theme_cards[:2]) if theme_cards else "flere konkurrerende hensyn"
    position_text = " og ".join(position.lower() for position, _ in top_positions)
    return (
        f"Dette handler egentlig ikke bare om selve forskriften, men om hvem som skal bære kostnaden og hvem som skal få gevinsten. "
        f"I denne høringen er hovedbildet preget av {position_text}, samtidig som aktørene samler seg rundt {dominant_themes}. "
        f"For {metadata.organization} betyr det at saken må håndteres som både fordelingspolitikk, næringspolitikk og styringsspørsmål."
    )


def _build_argument_map(actor_assessments: list[ActorAssessment]) -> list[str]:
    counter = Counter()
    for actor in actor_assessments:
        for tag in actor.tags:
            if tag.startswith("Opptatt av"):
                counter[tag] += actor.political_weight
    return [f"{tag}: samlet tyngde {weight}" for tag, weight in counter.most_common(6)]


def _build_recommended_follow_up(
    actor_assessments: list[ActorAssessment],
    theme_cards: list[ThemeCard],
    customer_type: str,
) -> list[str]:
    client = customer_type or "kunden"
    top_relevant = [actor.actor for actor in actor_assessments if actor.relevance_for_client == "høy"][:4]
    strongest_theme = theme_cards[0].theme if theme_cards else "hovedlinjen i høringen"
    actions = [
        f"Kontakt {', '.join(top_relevant)} først for å teste budskapet mot de mest relevante aktørene for {client}." if top_relevant else f"Identifiser de mest relevante aktørene for {client} før neste møte.",
        f"Bygg hovedbudskapet rundt temaet «{strongest_theme}» siden dette ser ut til å samle flest aktører.",
        "Forbered ett budskap for allierte og ett eget svar på de tyngste innvendingene fra motstanderne.",
        "Prioriter aktører med høy politisk tyngde før brede kontaktflater mot mindre avsendere.",
    ]
    return actions


def _build_executive_summary(
    metadata: HearingMetadata,
    actor_assessments: list[ActorAssessment],
    theme_cards: list[ThemeCard],
    customer_type: str,
    recommended_follow_up: list[str],
) -> str:
    top_position, top_count = Counter(actor.primary_position for actor in actor_assessments).most_common(1)[0]
    top_theme = theme_cards[0].theme if theme_cards else "flere ulike hensyn"
    allierte = [actor.actor for actor in actor_assessments if actor.relationship_to_client == "mulig alliert"][:3]
    motstandere = [actor.actor for actor in actor_assessments if actor.relationship_to_client == "motstander"][:3]
    client = customer_type or "kunden"
    return (
        f"Høringen om «{metadata.title}» viser at saken først og fremst dreier seg om {top_theme.lower()}. "
        f"Det største tyngdepunktet i materialet ligger nå i kategorien «{top_position}», med {top_count} aktører som trekker i den retningen. "
        f"For {client} betyr det at det politiske rommet trolig åpner seg best gjennom argumenter om lokal nytte, praktisk gjennomføring og tydelige rammer. "
        f"Mulige allierte er {', '.join(allierte) if allierte else 'ingen tydelige allierte ennå'}, mens {', '.join(motstandere) if motstandere else 'det foreløpig ikke finnes én samlet motblokk'} bør behandles som de viktigste innvendingene. "
        f"Anbefalt neste steg er å {recommended_follow_up[0].lower() if recommended_follow_up else 'teste budskapet raskt mot nøkkelaktører'}."
    )


def _build_customer_advice(
    customer_type: str,
    actor_assessments: list[ActorAssessment],
    theme_cards: list[ThemeCard],
    recommended_follow_up: list[str],
) -> CustomerAdvice:
    client = customer_type or "ukjent kundeprofil"
    return CustomerAdvice(
        customer_type=client,
        top_priority_actors=[actor.actor for actor in actor_assessments if actor.relevance_for_client == "høy"][:6],
        actors_to_contact=[actor.actor for actor in actor_assessments if actor.relationship_to_client != "nøytral"][:6],
        suggested_arguments=[card.client_message for card in theme_cards[:3]],
        likely_allies=[actor.actor for actor in actor_assessments if actor.relationship_to_client == "mulig alliert"][:6],
        likely_opponents=[actor.actor for actor in actor_assessments if actor.relationship_to_client == "motstander"][:6],
        objections_to_prepare=[actor.main_argument for actor in actor_assessments if actor.relationship_to_client == "motstander"][:4],
        recommended_actions=recommended_follow_up[:4],
    )


def _detect_primary_position(text: str) -> str:
    lowered = text.lower()
    for label, phrases in POSITION_RULES.items():
        if any(phrase in lowered for phrase in phrases):
            return label
    return "Uklart/ikke klassifiserbart"


def _detect_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = [label for label, phrases in TAG_RULES.items() if any(phrase in lowered for phrase in phrases)]
    return tags or ["Uklart/ikke klassifiserbart"]


def _pick_primary_position(tags: list[str], text: str) -> str:
    for candidate in [
        "Motsetter seg forslaget",
        "Ønsker strengere regulering",
        "Ønsker mildere regulering",
        "Støtter med forbehold",
        "Støtter forslaget",
    ]:
        if candidate in tags:
            return candidate
    return _detect_primary_position(text)


def _normalize_actor_type(actor_type: str) -> str:
    return ACTOR_TYPE_MAP.get(actor_type.strip().lower(), "annet")


def _infer_themes_from_text(text: str) -> list[str]:
    lowered = text.lower()
    scores = []
    for theme, keywords in THEME_DEFINITIONS.items():
        score = sum(lowered.count(keyword.lower()) for keyword in keywords)
        if score:
            scores.append((score, theme))
    return [theme for _, theme in sorted(scores, key=lambda item: item[0], reverse=True)[:3]]


def _extract_main_argument(text: str, themes: list[str]) -> str:
    sentences = _sentences(text)
    for sentence in sentences:
        if any(theme_word.lower() in sentence.lower() for theme in themes for theme_word in theme.split()):
            return _trim_sentence(sentence)
    return _trim_sentence(sentences[0]) if sentences else "Ingen tydelig hovedbegrunnelse funnet."


def _extract_concrete_request(text: str) -> str:
    requests = [
        sentence for sentence in _sentences(text)
        if any(marker in sentence.lower() for marker in ["bør", "må", "ønsker", "ber om", "foreslår", "forutsetter"])
    ]
    return _trim_sentence(requests[0]) if requests else "Ikke tydelig konkretisert i svaret."


def _extract_short_quote(text: str) -> str:
    sentences = _sentences(text)
    return _trim_sentence(sentences[0], max_length=180) if sentences else "Ingen sitatlinje tilgjengelig."


def _political_weight(actor: str, normalized_type: str) -> int:
    actor_lower = actor.lower()
    if normalized_type == "statlig organ":
        return 5
    if normalized_type == "kommune" and any(keyword in actor_lower for keyword in ["ål", "stranda", "luster", "tromsø", "trondheim", "stavanger", "oslo", "kristiansand", "ål esund", "ålesund"]):
        return 5
    if normalized_type in {"bransjeorganisasjon", "ngo"}:
        return 4
    if normalized_type == "kommune":
        return 4
    if normalized_type == "næringsliv":
        return 3
    if normalized_type == "privatperson":
        return 1
    return 2


def _power_label(weight: int) -> str:
    return {5: "svært høy", 4: "høy", 3: "middels", 2: "lav", 1: "svært lav"}.get(weight, "lav")


def _client_relevance(customer_type: str, tags: list[str], normalized_type: str, weight: int) -> str:
    if not customer_type:
        return "middels" if weight >= 3 else "lav"
    rules = CLIENT_PRIORITY_RULES.get(customer_type.lower())
    if rules and any(tag in rules["allies"] or tag in rules["opponents"] for tag in tags):
        return "høy" if weight >= 3 else "middels"
    if normalized_type in {"statlig organ", "bransjeorganisasjon"} and weight >= 4:
        return "høy"
    return "middels" if weight >= 3 else "lav"


def _relationship_to_client(customer_type: str, tags: list[str], primary_position: str) -> str:
    if not customer_type:
        return "nøytral"
    rules = CLIENT_PRIORITY_RULES.get(customer_type.lower())
    if not rules:
        return "nøytral"
    if any(tag in rules["allies"] for tag in tags):
        return "mulig alliert"
    if primary_position == "Motsetter seg forslaget" or any(tag in rules["opponents"] for tag in tags):
        return "motstander"
    return "nøytral"


def _theme_risk(theme: str, supporters: list[str], opponents: list[str]) -> str:
    if len(opponents) > len(supporters):
        return f"Høy risiko: flere tunge innvendinger enn tydelig støtte rundt temaet «{theme}»."
    if opponents:
        return f"Middels risiko: temaet samler støtte, men møter relevante forbehold."
    return f"Lavere risiko: temaet samler foreløpig mest støtte."


def _theme_message(theme: str) -> str:
    messages = {
        "Inntektene bør tilfalle vertskommunen": "Avgiften må oppleves som lokal verdiskaping, ikke som statlig skattlegging.",
        "Avgiften må være moderat og forutsigbar": "Reglene må gi forutsigbarhet og ikke undergrave investeringer og trafikkgrunnlag.",
        "Reiselivet må skjermes": "Forslaget bør vise at næringsaktivitet og lokalt bidrag kan balanseres.",
        "Reglene må bli tydeligere": "Kunden bør etterspørre klare kriterier, hjemler og enkel praktisering.",
        "Miljøbelastningen må håndteres strengere": "Kunden bør vise hvordan tiltak kan gi faktisk miljøeffekt og lokal legitimitet.",
    }
    return messages.get(theme, "Budskapet bør knyttes til lokal nytte, praktisk gjennomføring og politisk legitimitet.")


def _direction_title(themes: list[str], stance: str) -> str:
    if not themes:
        return stance
    lead = " / ".join(themes[:2])
    return f"{lead} ({stance})"


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


def _group_similarity_examples(indexes: list[int], responses: list[HearingResponse], similarity: np.ndarray) -> list[dict[str, str | float]]:
    examples = []
    for left in indexes:
        for right in indexes:
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


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence.strip()) > 20]


def _trim_sentence(text: str, max_length: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "…"


def _relevance_rank(label: str) -> int:
    return {"høy": 0, "middels": 1, "lav": 2}.get(label, 3)
