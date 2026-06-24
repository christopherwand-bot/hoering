from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import HearingMetadata, HearingResponse

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept-Language": "nb-NO,nb;q=0.9,en;q=0.8",
}


class HearingScraper:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(BASE_HEADERS)
        cookie_handle = tempfile.NamedTemporaryFile(prefix="hearing-cookies-", suffix=".txt", delete=False)
        cookie_handle.close()
        self.cookie_path = Path(cookie_handle.name)

    def scrape(self, hearing_url: str) -> tuple[HearingMetadata, list[HearingResponse], list[str]]:
        html = self._get_html(hearing_url, browser_like=False)
        soup = BeautifulSoup(html, "html.parser")
        metadata = self._parse_metadata(soup, hearing_url)
        responses = self._parse_response_index(soup, hearing_url)
        errors: list[str] = []

        for response in responses:
            try:
                self._populate_response(response, hearing_url)
            except Exception as exc:  # pragma: no cover - robustness around remote site
                response.errors.append(str(exc))
                errors.append(f"{response.actor}: {exc}")

        return metadata, responses, errors

    def _get(self, url: str, referer: str | None = None) -> requests.Response:
        headers = {}
        if referer:
            headers["Referer"] = referer
        response = self.session.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        return response

    def _get_html(self, url: str, referer: str | None = None, browser_like: bool = True) -> str:
        for attempt in range(2):
            command = [
                "curl",
                "-sS",
                "-L",
                "--http1.1",
                "--max-time",
                "25",
                "-b",
                str(self.cookie_path),
                "-c",
                str(self.cookie_path),
                url,
            ]
            if browser_like:
                command[6:6] = ["-A", BASE_HEADERS["User-Agent"]]
            if referer:
                command.extend(["-e", referer])
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if not self._is_unavailable_html(result.stdout):
                time.sleep(0.1)
                return result.stdout
            time.sleep(0.4 + (attempt * 0.6))
        return result.stdout

    def _parse_metadata(self, soup: BeautifulSoup, hearing_url: str) -> HearingMetadata:
        title = self._text(soup.select_one("h1")) or "Ukjent høring"
        organization = self._text(soup.select_one("a[href*='/dep/']")) or "Ukjent avsender"
        pairs = self._extract_label_values(soup)
        return HearingMetadata(
            title=title,
            organization=organization,
            published_date=pairs.get("Dato"),
            deadline=pairs.get("Høringsfrist"),
            status=pairs.get("Status"),
            source_url=hearing_url,
        )

    def _parse_response_index(self, soup: BeautifulSoup, hearing_url: str) -> list[HearingResponse]:
        items: list[HearingResponse] = []
        for li in soup.select("[data-horingssvar-list] > li"):
            anchor = li.select_one("a[href]")
            if not anchor:
                continue
            href = urljoin(hearing_url, anchor["href"])
            kind = "pdf" if ".pdf" in href.lower() else "html"
            items.append(
                HearingResponse(
                    actor=self._clean_whitespace(anchor.get_text(" ", strip=True)),
                    actor_type=(li.get("data-instans") or "Ukjent").strip(),
                    source_url=href,
                    source_kind=kind,
                    source_file_url=href if kind == "pdf" else None,
                )
            )
        return items

    def _populate_response(self, response: HearingResponse, hearing_url: str) -> None:
        if response.source_kind == "pdf":
            response.text = self._extract_pdf_text(response.source_url, hearing_url)
            response.title = response.actor
            return

        response_page = self._compose_response_url(hearing_url, response.source_url)
        document = self._get_html(response_page, referer=hearing_url)
        if self._is_unavailable_html(document):
            raise RuntimeError("Enkeltsvaret var midlertidig utilgjengelig fra regjeringen.no")
        soup = BeautifulSoup(document, "html.parser")
        response.title = self._text(soup.select_one("h1")) or response.actor
        meta = self._extract_label_values(soup)
        response.response_date = meta.get("Dato")
        response.response_type = meta.get("Svartype")
        response.text = self._extract_article_text(soup)

    def _compose_response_url(self, hearing_url: str, response_url: str) -> str:
        hearing_parts = urlparse(hearing_url)
        query = parse_qs(urlparse(response_url).query)
        uid = query.get("uid", [""])[0]
        if not uid:
            return response_url
        clean_base = f"{hearing_parts.scheme}://{hearing_parts.netloc}{hearing_parts.path}"
        return f"{clean_base}?showSvar=true&uid={uid}"

    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        body = soup.select_one(".article-body")
        if body:
            return self._clean_whitespace(body.get_text("\n", strip=True))
        main = soup.select_one("main")
        return self._clean_whitespace(main.get_text("\n", strip=True) if main else "")

    def _extract_pdf_text(self, pdf_url: str, hearing_url: str) -> str:
        response = self._get(pdf_url, referer=hearing_url)
        reader = PdfReader(BytesIO(response.content))
        text_parts: list[str] = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return self._clean_whitespace("\n".join(text_parts))

    def _extract_label_values(self, soup: BeautifulSoup) -> dict[str, str]:
        pairs: dict[str, str] = {}
        text = soup.get_text("\n", strip=True)
        for label in ["Dato", "Status", "Høringsfrist", "Svartype"]:
            match = re.search(rf"{label}:\s*([^\n]+)", text)
            if match:
                pairs[label] = self._clean_whitespace(match.group(1))
        return pairs

    @staticmethod
    def _is_unavailable_html(document: str) -> bool:
        markers = [
            "Tjenesten er midlertidig utilgjengelig",
            "Service temporarily unavailable",
            "Beklager, siden finnes ikke",
        ]
        return any(marker in document for marker in markers)

    @staticmethod
    def cache_key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    @staticmethod
    def _text(node: BeautifulSoup | None) -> str:
        if not node:
            return ""
        return HearingScraper._clean_whitespace(node.get_text(" ", strip=True))

    @staticmethod
    def _clean_whitespace(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()
