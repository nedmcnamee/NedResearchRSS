"""Render docs/papers.json into a single-file docs/index.html dashboard."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from dateutil import parser as dateparser
from jinja2 import Environment, FileSystemLoader, select_autoescape

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "src" / "templates"
PAPERS_JSON = REPO_ROOT / "docs" / "papers.json"
RATINGS_JSON = REPO_ROOT / "data" / "ratings.json"
CONFIG_YAML = REPO_ROOT / "config.yaml"
OUT_HTML = REPO_ROOT / "docs" / "index.html"
DISPLAY_TZ = ZoneInfo("Australia/Melbourne")


def main() -> int:
    if not PAPERS_JSON.exists():
        print(
            f"error: {PAPERS_JSON} not found. Run `python src/fetch_and_score.py` first.",
            file=sys.stderr,
        )
        return 1

    with open(PAPERS_JSON) as f:
        data = json.load(f)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("dashboard.html.j2")

    updated_at_raw = data.get("updated_at") or datetime.now(timezone.utc).isoformat()
    try:
        updated_dt = dateparser.parse(updated_at_raw)
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
        updated_local = updated_dt.astimezone(DISPLAY_TZ)
        # %Z resolves to AEST or AEDT depending on daylight saving
        updated_human = updated_local.strftime("%Y-%m-%d %H:%M %Z (Melbourne)")
    except Exception:
        updated_human = updated_at_raw

    papers_json_inline = json.dumps(data["papers"], ensure_ascii=False)
    # Guard against accidental </script> in abstracts breaking the inline block.
    papers_json_inline = papers_json_inline.replace("</", "<\\/")

    # --- Admin config (passphrase hash, repo) ---
    admin_hash = ""
    admin_repo = ""
    try:
        with open(CONFIG_YAML) as cf:
            cfg = yaml.safe_load(cf)
        admin_cfg = cfg.get("admin") or {}
        admin_hash = admin_cfg.get("passphrase_sha256") or ""
        admin_repo = admin_cfg.get("repo") or ""
    except Exception:
        pass

    # --- Existing ratings (for cross-device merge) ---
    ratings_json_inline = "{}"
    if RATINGS_JSON.exists():
        try:
            with open(RATINGS_JSON) as rf:
                ratings_data = json.load(rf)
            ratings_json_inline = json.dumps(
                ratings_data.get("ratings") or {}, ensure_ascii=False
            ).replace("</", "<\\/")
        except Exception:
            pass

    html = template.render(
        updated_human=updated_human,
        count=data.get("count", 0),
        tier_counts=data.get("tier_counts", {"high": 0, "medium": 0, "low": 0}),
        library_size=data.get("library_size", 0),
        feeds=data.get("feeds", []),
        papers_json=papers_json_inline,
        admin_hash=admin_hash,
        admin_repo=admin_repo,
        ratings_json=ratings_json_inline,
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"wrote {OUT_HTML} ({len(html)//1024} KB, {data.get('count', 0)} papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
