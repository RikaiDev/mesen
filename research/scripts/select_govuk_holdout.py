"""Freeze public GOV.UK help routes before model or page inspection."""

import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "holdout" / "govuk-help-source.html"
OUTPUT = ROOT / "artifacts" / "holdout" / "govuk-route-selection.json"
BASE = "https://www.gov.uk/help"
SEED = "mesen-uiux-blind-govuk-v1-2026-09-28"


class Links(HTMLParser):
    """Collect navigation URLs without using page judgments."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def main() -> None:
    """Select ten distinct same-site help pages by deterministic hash rank."""
    if OUTPUT.exists():
        raise FileExistsError("GOV.UK route selection is already sealed")
    source = SOURCE.read_bytes()
    links = Links()
    links.feed(source.decode("utf-8"))
    routes = {
        urldefrag(urljoin(BASE, href)).url
        for href in links.hrefs
        if urlsplit(urljoin(BASE, href)).netloc == "www.gov.uk"
        and urlsplit(urljoin(BASE, href)).path.startswith("/help/")
    }
    candidates = sorted(routes)
    if len(candidates) < 10:
        raise ValueError("GOV.UK help navigation has fewer than ten linked pages")
    selected = sorted(
        candidates,
        key=lambda route: hashlib.sha256(f"{SEED}:{route}".encode()).hexdigest(),
    )[:10]
    packet = {
        "schema": "mesen.public-route-selection.v1",
        "site_family": "govuk-help",
        "source": BASE,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "selection_seed": SEED,
        "selection_method": "ten smallest SHA256(seed:url) among linked /help/ pages",
        "excluded_from_training": True,
        "routes": selected,
        "selected_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"routes": len(selected), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
