"""Optional second-stage LLM rerank for shortlisted papers.

Usage from `fetch_and_score.py`:

    from llm_rerank import rerank
    scored = rerank(scored, config)   # mutates each dict to add llm_score / llm_reason

The function is a no-op (with a warning) when:
  - `config["scoring"]["llm_rerank"]["enabled"]` is false, OR
  - `ANTHROPIC_API_KEY` env var is missing.

Caching: results are persisted to `data/llm_cache.json` keyed by `(model, paper_id)`.
The cache is reset whenever the configured model differs from the cached model.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- cache I/O ----------


def _load_cache(cache_path: Path, model: str) -> dict[str, dict]:
    """Load existing cache for the given model. Reset on model mismatch."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("model") != model:
        print(
            f"  llm_rerank: cached model={data.get('model')!r} differs from "
            f"config model={model!r} — resetting cache"
        )
        return {}
    return dict(data.get("entries") or {})


def _save_cache(cache_path: Path, model: str, entries: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "entries": entries}
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------- prompt + parsing ----------


_SYSTEM_PROMPT = (
    "You are filtering academic papers for a researcher. Score each paper from 0 "
    "(irrelevant) to 100 (must-read) based on how well it matches the researcher's "
    "stated interests, then give a single concise sentence justifying the score. "
    "Always respond as a JSON object with keys 'score' (integer 0-100) and 'reason' "
    "(string). No prose outside the JSON."
)


def _build_user_prompt(profile_brief: str, paper: dict) -> str:
    abstract = (paper.get("abstract") or "").strip()
    if len(abstract) > 1800:
        abstract = abstract[:1800].rstrip() + "…"
    return (
        f"Researcher interests:\n{profile_brief.strip()}\n\n"
        f"Paper title: {paper.get('title', '').strip()}\n"
        f"Paper abstract: {abstract or '(no abstract available)'}\n\n"
        "Respond with JSON only."
    )


def _parse_response(text: str) -> dict | None:
    """Parse the LLM's JSON response into {score, reason}."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences if the model added them.
    if text.startswith("```"):
        text = text.strip("`")
        # remove leading 'json\n'
        if text.lower().startswith("json"):
            text = text[4:].lstrip("\n").lstrip()
    # Find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    score = obj.get("score")
    reason = obj.get("reason")
    if not isinstance(score, (int, float)) or not isinstance(reason, str):
        return None
    return {"score": max(0, min(100, int(round(score)))), "reason": reason.strip()}


# ---------- async dispatch ----------


async def _score_one(client, model: str, system: str, user: str) -> dict | None:
    try:
        msg = await client.messages.create(
            model=model,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        print(f"  llm_rerank: API error: {e}", file=sys.stderr)
        return None
    text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return _parse_response("".join(text_blocks))


async def _score_many(
    papers_to_score: list[dict],
    model: str,
    profile_brief: str,
    max_concurrent: int,
) -> dict[str, dict]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(max_concurrent)
    results: dict[str, dict] = {}

    async def worker(paper: dict) -> None:
        async with sem:
            user_prompt = _build_user_prompt(profile_brief, paper)
            parsed = await _score_one(client, model, _SYSTEM_PROMPT, user_prompt)
            if parsed is not None:
                parsed["model"] = model
                parsed["scored_at"] = datetime.now(timezone.utc).isoformat()
                results[paper["id"]] = parsed

    await asyncio.gather(*(worker(p) for p in papers_to_score))
    return results


# ---------- public entrypoint ----------


def rerank(scored: list[dict], config: dict) -> list[dict]:
    """Stage 2: re-score shortlisted papers with the Anthropic API.

    Mutates each paper dict above the threshold to add `llm_score` and `llm_reason`.
    Recomputes `final_score` and `tier` from the blend. Returns the same list,
    re-sorted by the new final_score.
    """
    rcfg = (config.get("scoring") or {}).get("llm_rerank") or {}
    if not rcfg.get("enabled"):
        print("  llm_rerank: disabled in config; skipping")
        return scored

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "  llm_rerank: ANTHROPIC_API_KEY not set; skipping LLM rerank "
            "(set the env var locally or add as a GitHub secret to enable)",
            file=sys.stderr,
        )
        return scored

    model = rcfg.get("model", "claude-sonnet-4-6")
    threshold = float(rcfg.get("apply_to_min_score", 30))
    blend_w = float(rcfg.get("blend_weight", 0.5))
    cache_path = REPO_ROOT / rcfg.get("cache_path", "data/llm_cache.json")
    max_concurrent = int(rcfg.get("max_concurrent", 5))
    profile_brief = (
        rcfg.get("profile_brief")
        or ", ".join((config.get("research_profile") or {}).get("tier1_keywords") or [])
    )

    cache = _load_cache(cache_path, model)
    shortlisted = [p for p in scored if p.get("final_score", 0) >= threshold]
    if not shortlisted:
        print(f"  llm_rerank: no papers above threshold ({threshold}); skipping")
        return scored

    cached = [p for p in shortlisted if p["id"] in cache]
    fresh = [p for p in shortlisted if p["id"] not in cache]
    print(
        f"  llm_rerank: {len(shortlisted)} candidates above {threshold:.0f}  "
        f"({len(cached)} cached, {len(fresh)} new API calls)  model={model}"
    )

    if fresh:
        new_results = asyncio.run(_score_many(fresh, model, profile_brief, max_concurrent))
        cache.update(new_results)
        _save_cache(cache_path, model, cache)
        print(
            f"  llm_rerank: scored {len(new_results)}/{len(fresh)} new papers; "
            f"cache now has {len(cache)} entries"
        )

    # Apply scores back to the shortlisted papers
    applied = 0
    for p in shortlisted:
        entry = cache.get(p["id"])
        if not entry:
            p["llm_score"] = None
            p["llm_reason"] = None
            continue
        p["llm_score"] = float(entry["score"])
        p["llm_reason"] = entry.get("reason")
        p["stage1_score"] = p["final_score"]
        p["final_score"] = round(blend_w * p["stage1_score"] + (1 - blend_w) * p["llm_score"], 2)
        applied += 1

    # Re-tier and re-sort
    tiers = (config.get("scoring") or {}).get("tiers") or {}
    high_tier = float(tiers.get("high", 60))
    medium_tier = float(tiers.get("medium", 30))
    for p in scored:
        s = p["final_score"]
        p["tier"] = "high" if s >= high_tier else ("medium" if s >= medium_tier else "low")
    scored.sort(key=lambda p: p["final_score"], reverse=True)
    print(f"  llm_rerank: applied LLM scores to {applied} papers; pipeline complete")
    return scored
