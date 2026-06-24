from __future__ import annotations

import csv
import io
from urllib.parse import quote

from flask import Blueprint, Response, current_app, render_template, request

from .analyzer import analyze_hearing
from .cache import load_analysis, save_analysis
from .scraper import HearingScraper

dashboard_bp = Blueprint("dashboard", __name__)
DEFAULT_HEARING_URL = "https://www.regjeringen.no/no/dokumenter/horing-av-forskrift-om-cruiseavgift/id3151840/?showSvar=true&consterm=&page=1&isFilterOpen=true"
DEFAULT_CUSTOMER_TYPE = "cruiseaktør"


def _get_result(hearing_url: str, customer_type: str, force_refresh: bool):
    cache_dir = current_app.config["CACHE_DIR"]
    scraper = HearingScraper()
    cache_key = scraper.cache_key(f"lobby-v3::{hearing_url}::{customer_type.lower().strip()}")
    result = None if force_refresh else load_analysis(cache_dir, cache_key)
    if result is None:
        metadata, responses, errors = scraper.scrape(hearing_url)
        result = analyze_hearing(metadata, responses, errors, customer_type=customer_type)
        save_analysis(cache_dir, cache_key, result)
    return result


@dashboard_bp.route("/", methods=["GET", "POST"])
def index():
    hearing_url = request.form.get("hearing_url") or request.args.get("hearing_url") or DEFAULT_HEARING_URL
    customer_type = request.form.get("customer_type") or request.args.get("customer_type") or DEFAULT_CUSTOMER_TYPE
    force_refresh = request.form.get("force_refresh") == "1"
    should_analyze = request.method == "POST" or bool(request.args.get("hearing_url"))
    result = None
    fatal_error = None

    if should_analyze:
        try:
            result = _get_result(hearing_url, customer_type, force_refresh)
        except Exception as exc:
            fatal_error = f"Analysen kunne ikke fullføres akkurat nå: {exc}"

    return render_template(
        "index.html",
        hearing_url=hearing_url,
        customer_type=customer_type,
        result=result,
        fatal_error=fatal_error,
        total_responses=len(result.responses) if result else 0,
        analyzed_responses=sum(1 for response in result.responses if response.text) if result else 0,
        mailto_link=_mailto_link(result, customer_type) if result else "",
    )


@dashboard_bp.route("/export/actors.csv")
def export_actors_csv():
    hearing_url = request.args.get("hearing_url", DEFAULT_HEARING_URL)
    customer_type = request.args.get("customer_type", DEFAULT_CUSTOMER_TYPE)
    result = _get_result(hearing_url, customer_type, force_refresh=False)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "Aktør",
        "Aktørtype",
        "Standpunkt",
        "Viktigste argument",
        "Hva de ber om konkret",
        "Relevans for kunden",
        "Mulig alliert/motstander/nøytral",
        "Kort sitat",
        "Lenke til originalt svar",
    ])
    for actor in result.actor_assessments:
        writer.writerow([
            actor.actor,
            actor.normalized_actor_type,
            actor.primary_position,
            actor.main_argument,
            actor.concrete_request,
            actor.relevance_for_client,
            actor.relationship_to_client,
            actor.short_quote,
            actor.source_url,
        ])

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=aktortabell.csv"},
    )


@dashboard_bp.route("/export/brief")
def export_brief():
    hearing_url = request.args.get("hearing_url", DEFAULT_HEARING_URL)
    customer_type = request.args.get("customer_type", DEFAULT_CUSTOMER_TYPE)
    result = _get_result(hearing_url, customer_type, force_refresh=False)
    return render_template("brief.html", result=result, customer_type=customer_type)


@dashboard_bp.route("/export/meeting-note.txt")
def export_meeting_note():
    hearing_url = request.args.get("hearing_url", DEFAULT_HEARING_URL)
    customer_type = request.args.get("customer_type", DEFAULT_CUSTOMER_TYPE)
    result = _get_result(hearing_url, customer_type, force_refresh=False)
    lines = [
        f"Kundeprofil: {customer_type}",
        f"Høring: {result.metadata.title}",
        "",
        "Kort oppsummering:",
        *[f"- {point}" for point in result.summary_points],
        "",
        "Anbefalt oppfølging:",
        *[f"- {point}" for point in result.recommended_follow_up],
    ]
    return Response(
        "\n".join(lines),
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=motenotat.txt"},
    )


def _mailto_link(result, customer_type: str) -> str:
    subject = quote(f"Kundebrief: {result.metadata.title}")
    body = quote(
        f"Kundeprofil: {customer_type}\n\n"
        f"Executive summary:\n{result.executive_summary}\n\n"
        f"Anbefalt oppfølging:\n- " + "\n- ".join(result.recommended_follow_up[:4])
    )
    return f"mailto:?subject={subject}&body={body}"
