from __future__ import annotations

from flask import Blueprint, current_app, render_template, request

from .analyzer import analyze_hearing
from .cache import load_analysis, save_analysis
from .scraper import HearingScraper

dashboard_bp = Blueprint("dashboard", __name__)
DEFAULT_HEARING_URL = "https://www.regjeringen.no/no/dokumenter/horing-av-forskrift-om-cruiseavgift/id3151840/?showSvar=true&consterm=&page=1&isFilterOpen=true"


@dashboard_bp.route("/", methods=["GET", "POST"])
def index():
    hearing_url = request.form.get("hearing_url") or request.args.get("hearing_url") or DEFAULT_HEARING_URL
    force_refresh = request.form.get("force_refresh") == "1"
    should_analyze = request.method == "POST" or bool(request.args.get("hearing_url"))
    result = None
    fatal_error = None

    if should_analyze:
        cache_dir = current_app.config["CACHE_DIR"]
        scraper = HearingScraper()
        cache_key = scraper.cache_key(hearing_url)

        result = None if force_refresh else load_analysis(cache_dir, cache_key)
        if result is None:
            try:
                metadata, responses, errors = scraper.scrape(hearing_url)
                result = analyze_hearing(metadata, responses, errors)
                save_analysis(cache_dir, cache_key, result)
            except Exception as exc:
                fatal_error = f"Analysen kunne ikke fullføres akkurat nå: {exc}"

    return render_template(
        "index.html",
        hearing_url=hearing_url,
        result=result,
        fatal_error=fatal_error,
        total_responses=len(result.responses) if result else 0,
        analyzed_responses=sum(1 for response in result.responses if response.text) if result else 0,
    )
