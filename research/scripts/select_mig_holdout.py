"""Freeze unseen MIG guideline pages from public navigation links."""

import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "holdout" / "mig-lab-source.html"
OUTPUT = ROOT / "artifacts" / "holdout" / "mig-route-selection.json"
BASE = "https://rikaidev.github.io/MIG/zh/lab.html"
SEED = "mesen-uiux-blind-mig-v1-2026-09-28"


class Links(HTMLParser):
    """Collect ordinary anchor destinations without rendering page content."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def main() -> None:
    """Select ten distinct pages by fixed hash rank before capture."""
    if OUTPUT.exists():
        raise FileExistsError("MIG route selection is already sealed")
    source = SOURCE.read_bytes()
    links = Links()
    links.feed(source.decode("utf-8"))
    routes = {
        urldefrag(urljoin(BASE, href)).url
        for href in links.hrefs
        if urlsplit(urljoin(BASE, href)).path.startswith("/MIG/zh/")
    }
    candidates = sorted(route for route in routes if route != BASE and route.endswith(".html"))
    if len(candidates) < 10:
        raise ValueError("MIG navigation has fewer than ten unseen guideline pages")
    selected = sorted(
        candidates,
        key=lambda route: hashlib.sha256(f"{SEED}:{route}".encode()).hexdigest(),
    )[:10]
    packet = {
        "schema": "mesen.public-route-selection.v1",
        "site_family": "mig-guidelines",
        "source": BASE,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "selection_seed": SEED,
        "selection_method": "ten smallest SHA256(seed:url) among linked zh HTML pages other than lab.html",
        "excluded_from_training": True,
        "routes": selected,
        "selected_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"routes": len(selected), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
