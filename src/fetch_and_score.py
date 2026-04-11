"""Fetch RSS feeds, dedupe against the Paperpile library, score, and emit docs/papers.json.

Run order:
    python src/build_reference.py      # one-time (or on library change)
    python src/fetch_and_score.py      # daily
    python src/generate_dashboard.py

Scoring:
    nn_score  = clip((max_cosine - low) / (high - low), 0, 1) * 100
    kw_score  = min(#tier1*30 + #tier2*15 + #tier3*7, 100)
    final     = 0.7 * nn_score + 0.3 * kw_score
    tier      = high (>=60) | medium (>=30) | low (<30)
"""

from __future__ import annotations

import html
import json
import re
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import numpy as np
import yaml
from dateutil import parser as dateparser

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = REPO_ROOT / "data" / "knowledgebase.md"


# ---------- helpers ----------


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_kb_keywords() -> dict[str, list[str]]:
    """Parse the 'Keywords for Paper Matching' section of data/knowledgebase.md.

    Returns {"tier1": [...], "tier2": [...], "tier3": [...]} with any tiers
    not mentioned mapped to empty lists. If the KB file is missing or the
    section isn't found, returns empty dict.
    """
    if not KB_PATH.exists():
        return {}
    try:
        text = KB_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    # Find the keywords section
    m = re.search(r"##\s*Keywords for Paper Matching(.*?)(?:\n##\s|\Z)", text, re.DOTALL)
    if not m:
        return {}
    section = m.group(1)
    tiers: dict[str, list[str]] = {}
    for tier_num, tier_key in [("1", "tier1"), ("2", "tier2"), ("3", "tier3")]:
        pat = rf"###\s*Tier\s*{tier_num}[^\n]*\n([^#]*?)(?:\n###|\Z)"
        tm = re.search(pat, section, re.DOTALL)
        if tm:
            body = tm.group(1).strip()
            words = [w.strip() for w in body.split(",") if w.strip()]
            tiers[tier_key] = words
    return tiers


def normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_html(raw: str) -> str:
    """Minimal HTML → plain text: unescape entities and drop tags."""
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_ARXIV_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.IGNORECASE)
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def extract_doi(entry, link: str) -> str | None:
    """Pull a DOI out of a feedparser entry or its link."""
    # prism:doi is common on Nature/Cell feeds
    doi = entry.get("prism_doi") or entry.get("dc_identifier") or entry.get("id")
    if isinstance(doi, str):
        m = _DOI_RE.search(doi)
        if m:
            return m.group(0).lower()
    # else try the link
    if link:
        m = _DOI_RE.search(link)
        if m:
            return m.group(0).lower()
    return None


def extract_arxiv_id(entry, link: str) -> str | None:
    for field in ("id", "link", "arxiv_id"):
        val = entry.get(field) or ""
        if isinstance(val, str):
            m = _ARXIV_RE.search(val)
            if m:
                return m.group(1).lower()
    if link:
        m = _ARXIV_RE.search(link)
        if m:
            return m.group(1).lower()
    return None


def parse_entry_date(entry) -> datetime | None:
    for field in ("published", "updated", "created"):
        raw = entry.get(field)
        if not raw:
            continue
        try:
            dt = dateparser.parse(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue
    # feedparser also exposes *_parsed struct_time fields
    for field in ("published_parsed", "updated_parsed"):
        st = entry.get(field)
        if st:
            try:
                return datetime(*st[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def format_authors(entry) -> list[str]:
    authors_field = entry.get("authors") or []
    names: list[str] = []
    for a in authors_field:
        if isinstance(a, dict):
            name = a.get("name") or ""
        else:
            name = str(a)
        name = name.strip()
        if name:
            names.append(name)
    if not names:
        author = entry.get("author")
        if isinstance(author, str) and author.strip():
            names = [p.strip() for p in re.split(r"[,;]", author) if p.strip()]
    return names


def fetch_feed(feed: dict, user_agent: str, timeout: int) -> list[dict]:
    """Pull one feed and normalise its entries."""
    name = feed["name"]
    url = feed["url"]
    print(f"  fetching: {name}  ({url})")
    # feedparser 6 accepts request_headers but no timeout arg; set socket default
    socket.setdefaulttimeout(timeout)
    try:
        parsed = feedparser.parse(url, request_headers={"User-Agent": user_agent})
    except Exception as e:
        print(f"    ! error: {e}", file=sys.stderr)
        return []
    if parsed.bozo and not parsed.entries:
        exc = parsed.bozo_exception
        print(f"    ! parse failed: {exc}", file=sys.stderr)
        return []
    out: list[dict] = []
    for entry in parsed.entries:
        title = strip_html(entry.get("title") or "")
        if not title:
            continue
        summary = strip_html(entry.get("summary") or entry.get("description") or "")
        link = entry.get("link") or ""
        published = parse_entry_date(entry)
        out.append(
            {
                "title": title,
                "abstract": summary,
                "authors": format_authors(entry),
                "url": link,
                "journal": name,
                "published": published.isoformat() if published else None,
                "published_dt": published,
                "doi": extract_doi(entry, link),
                "arxiv_id": extract_arxiv_id(entry, link),
                "title_norm": normalize_title(title),
            }
        )
    print(f"    got {len(out)} entries")
    return out


# ---------- dedupe / scoring ----------


def build_library_keys(reference_meta: dict) -> tuple[set[str], set[str], set[str]]:
    dois: set[str] = set()
    arxiv_ids: set[str] = set()
    titles: set[str] = set()
    for p in reference_meta.get("papers", []):
        if p.get("doi"):
            dois.add(p["doi"])
        if p.get("arxiv_id"):
            arxiv_ids.add(p["arxiv_id"])
        if p.get("title_norm"):
            titles.add(p["title_norm"])
    return dois, arxiv_ids, titles


def is_duplicate(paper: dict, dois: set[str], arxiv_ids: set[str], titles: set[str]) -> bool:
    if paper.get("doi") and paper["doi"] in dois:
        return True
    if paper.get("arxiv_id") and paper["arxiv_id"] in arxiv_ids:
        return True
    if paper.get("title_norm") and paper["title_norm"] in titles:
        return True
    return False


def compute_recency_weights(
    reference_papers: list[dict],
    half_life_years: float,
    floor: float,
    now: datetime | None = None,
) -> np.ndarray:
    """Per-library-paper weight in [floor, 1.0] based on when the user added it."""
    if now is None:
        now = datetime.now(timezone.utc)
    weights = np.empty(len(reference_papers), dtype=np.float32)
    for i, p in enumerate(reference_papers):
        ts = p.get("added_ts")
        if not ts:
            weights[i] = floor
            continue
        try:
            added_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, OSError):
            weights[i] = floor
            continue
        years_old = (now - added_dt).total_seconds() / (365.25 * 86400)
        w = 0.5 ** (years_old / half_life_years)
        weights[i] = max(w, floor)
    return weights


def score_papers(
    papers: list[dict],
    reference_vectors: np.ndarray,
    reference_meta: dict,
    config: dict,
) -> list[dict]:
    if not papers:
        return []

    scoring = config["scoring"]
    low = float(scoring["nn_sim_low"])
    high = float(scoring["nn_sim_high"])
    nn_w = float(scoring["weights"]["nn"])
    kw_w = float(scoring["weights"]["keyword"])
    high_tier = float(scoring["tiers"]["high"])
    medium_tier = float(scoring["tiers"]["medium"])
    half_life = float(scoring.get("recency_half_life_years", 5))
    floor = float(scoring.get("recency_floor", 0.2))

    profile = config["research_profile"]
    cfg_t1 = [k.lower() for k in profile.get("tier1_keywords") or []]
    cfg_t2 = [k.lower() for k in profile.get("tier2_keywords") or []]
    cfg_t3 = [k.lower() for k in profile.get("tier3_keywords") or []]

    # Merge in keywords from data/knowledgebase.md if present (union).
    kb_kw = load_kb_keywords()
    kb_t1 = [k.lower() for k in kb_kw.get("tier1", [])]
    kb_t2 = [k.lower() for k in kb_kw.get("tier2", [])]
    kb_t3 = [k.lower() for k in kb_kw.get("tier3", [])]

    def _union(a: list[str], b: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for kw in a + b:
            if kw and kw not in seen:
                seen.add(kw)
                out.append(kw)
        return out

    tier1 = _union(cfg_t1, kb_t1)
    tier2 = _union(cfg_t2, kb_t2)
    tier3 = _union(cfg_t3, kb_t3)
    if kb_kw:
        print(
            f"  keyword tiers merged with KB: "
            f"tier1={len(tier1)} (+{len(kb_t1)})  "
            f"tier2={len(tier2)} (+{len(kb_t2)})  "
            f"tier3={len(tier3)} (+{len(kb_t3)})"
        )

    reference_papers = reference_meta.get("papers", [])
    recency_weights = compute_recency_weights(reference_papers, half_life, floor)
    print(
        f"recency weights: half_life={half_life}y floor={floor}  "
        f"min={recency_weights.min():.3f}  median={float(np.median(recency_weights)):.3f}  "
        f"max={recency_weights.max():.3f}"
    )

    # --- embedding score ---
    from sentence_transformers import SentenceTransformer

    print(f"loading embedding model: {scoring['model']}")
    model = SentenceTransformer(scoring["model"])
    texts = [f"{p['title']}. {p['abstract']}" if p["abstract"] else p["title"] for p in papers]
    print(f"encoding {len(texts)} candidate papers...")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    # cosine similarity == dot product since both sides are L2-normalised
    sims = vectors @ reference_vectors.T  # [n_papers, n_reference]
    weighted_sims = sims * recency_weights[None, :]
    max_weighted = weighted_sims.max(axis=1)
    argmax = weighted_sims.argmax(axis=1)
    nn_scores = np.clip((max_weighted - low) / max(high - low, 1e-9), 0.0, 1.0) * 100.0

    # --- keyword score ---
    scored: list[dict] = []
    for i, paper in enumerate(papers):
        text = f"{paper['title']} {paper['abstract']}".lower()
        matched: list[str] = []
        kw_points = 0
        for kw in tier1:
            if kw and kw in text:
                matched.append(kw)
                kw_points += 30
        for kw in tier2:
            if kw and kw in text:
                matched.append(kw)
                kw_points += 15
        for kw in tier3:
            if kw and kw in text:
                matched.append(kw)
                kw_points += 7
        kw_score = min(kw_points, 100)

        nn_score = float(nn_scores[i])
        final = round(nn_w * nn_score + kw_w * kw_score, 2)
        if final >= high_tier:
            tier = "high"
        elif final >= medium_tier:
            tier = "medium"
        else:
            tier = "low"

        # Top library match diagnostics
        top_idx = int(argmax[i])
        top_ref = reference_papers[top_idx] if 0 <= top_idx < len(reference_papers) else {}
        top_added_year: int | None = None
        if top_ref.get("added_ts"):
            try:
                top_added_year = datetime.fromtimestamp(
                    float(top_ref["added_ts"]), tz=timezone.utc
                ).year
            except (ValueError, OSError):
                top_added_year = None

        scored.append(
            {
                "id": paper.get("doi") or paper.get("arxiv_id") or paper.get("url") or paper["title"],
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["abstract"],
                "url": paper["url"],
                "journal": paper["journal"],
                "published": paper["published"],
                "doi": paper.get("doi"),
                "arxiv_id": paper.get("arxiv_id"),
                "final_score": final,
                "embedding_score": round(nn_score, 2),
                "keyword_score": round(float(kw_score), 2),
                "max_cosine_raw": round(float(sims[i, top_idx]), 4),
                "max_cosine_weighted": round(float(max_weighted[i]), 4),
                "recency_weight_used": round(float(recency_weights[top_idx]), 3),
                "top_match_added_year": top_added_year,
                "top_match_title": top_ref.get("title", "")[:120],
                "tier": tier,
                "matched_keywords": matched,
            }
        )

    scored.sort(key=lambda p: p["final_score"], reverse=True)
    return scored


# ---------- main ----------


def main() -> int:
    cfg = load_config()

    ref_path = REPO_ROOT / cfg["library"]["cache"]
    meta_path = REPO_ROOT / cfg["library"]["meta"]
    if not ref_path.exists() or not meta_path.exists():
        print(
            "error: reference index not found. Run `python src/build_reference.py` first.",
            file=sys.stderr,
        )
        return 1

    print(f"loading reference index from {ref_path}")
    reference_vectors = np.load(ref_path)["vectors"]
    with open(meta_path) as f:
        reference_meta = json.load(f)
    print(f"  {reference_vectors.shape[0]} reference papers")

    dois, arxiv_ids, titles = build_library_keys(reference_meta)

    fetch_cfg = cfg["fetch"]
    ua = fetch_cfg.get("user_agent", "SimonResearchRSS/0.1")
    timeout = int(fetch_cfg.get("request_timeout", 30))
    lookback_days = int(fetch_cfg.get("lookback_days", 14))
    max_per_feed = int(fetch_cfg.get("max_papers_per_feed", 100))
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    print(f"\nfetching {len(cfg['feeds'])} feeds (lookback: {lookback_days}d)")
    all_papers: list[dict] = []
    for feed in cfg["feeds"]:
        entries = fetch_feed(feed, user_agent=ua, timeout=timeout)
        # trim to recent + cap
        filtered: list[dict] = []
        for e in entries:
            dt = e.pop("published_dt", None)
            if dt is not None and dt < cutoff:
                continue
            filtered.append(e)
            if len(filtered) >= max_per_feed:
                break
        all_papers.extend(filtered)
        time.sleep(0.2)  # polite pause between hosts
    print(f"\ncollected {len(all_papers)} papers across all feeds")

    # internal dedupe (same paper from multiple feeds)
    seen_keys: set[str] = set()
    unique: list[dict] = []
    for p in all_papers:
        key = (
            (p.get("doi") and f"doi:{p['doi']}")
            or (p.get("arxiv_id") and f"arxiv:{p['arxiv_id']}")
            or (p.get("title_norm") and f"title:{p['title_norm']}")
            or (p.get("url") and f"url:{p['url']}")
        )
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(p)
    print(f"  {len(unique)} unique after cross-feed dedupe")

    # library dedupe
    fresh = [p for p in unique if not is_duplicate(p, dois, arxiv_ids, titles)]
    print(f"  {len(fresh)} after dedupe against Paperpile library")

    # score
    scored = score_papers(fresh, reference_vectors, reference_meta, cfg)

    # optional second-stage LLM rerank (no-op when disabled or API key missing)
    from llm_rerank import rerank
    scored = rerank(scored, cfg)

    # write output
    out_path = REPO_ROOT / "docs" / "papers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for p in scored:
        tier_counts[p["tier"]] += 1
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(scored),
        "tier_counts": tier_counts,
        "feeds": [f["name"] for f in cfg["feeds"]],
        "lookback_days": lookback_days,
        "library_size": reference_vectors.shape[0],
        "papers": scored,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")
    print(f"  tier counts: {tier_counts}")
    if scored:
        top = scored[0]
        print(f"  top paper: {top['final_score']:.1f}  {top['title'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
