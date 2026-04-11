"""Optional second-stage LLM rerank for shortlisted papers.

Usage from `fetch_and_score.py`:

    from llm_rerank import rerank
    scored = rerank(scored, config)   # mutates each dict to add llm_score

The function is a no-op (with a warning) when:
  - `config["scoring"]["llm_rerank"]["enabled"]` is false, OR
  - `ANTHROPIC_API_KEY` env var is missing.

Caching: results are persisted to `data/llm_cache.json` keyed by `(model, paper_id)`.
The cache is reset whenever the configured model OR knowledgebase content changes.

Privacy: the LLM produces a one-sentence reason alongside its score, but the
reason text is grounded in the (private) knowledgebase and could leak
unpublished research ideas. Reasons are kept in the local cache file only and
are NOT propagated onto paper dicts (so they never reach the public papers.json
or dashboard). The cache itself is gitignored and persisted across CI runs via
GitHub Actions cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = REPO_ROOT / "data" / "knowledgebase.md"


def load_knowledgebase() -> tuple[str | None, str | None]:
    """Return (kb_text, kb_sha256) if data/knowledgebase.md exists, else (None, None).

    The KB gives the LLM a rich understanding of the researcher's focus, projects,
    methods, and collaborators — far more context than the short `profile_brief`.
    """
    if not KB_PATH.exists():
        return None, None
    try:
        text = KB_PATH.read_text(encoding="utf-8")
    except OSError:
        return None, None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text.strip(), digest[:16]


# ---------- cache I/O ----------


def _load_cache(cache_path: Path, model: str, kb_hash: str | None) -> dict[str, dict]:
    """Load existing cache. Reset on model OR KB content mismatch.

    KB content is baked into the system prompt, so any KB edit invalidates
    existing scores — we want fresh judgment under the new context.
    """
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
    if data.get("kb_hash") != kb_hash:
        print(
            f"  llm_rerank: cached kb_hash={data.get('kb_hash')!r} differs from "
            f"current kb_hash={kb_hash!r} — resetting cache"
        )
        return {}
    return dict(data.get("entries") or {})


def _save_cache(
    cache_path: Path,
    model: str,
    kb_hash: str | None,
    entries: dict[str, dict],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "kb_hash": kb_hash, "entries": entries}
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------- prompt + parsing ----------


_SYSTEM_PROMPT_TEMPLATE = (
    "You are filtering academic papers for a researcher. Score each paper from 0 "
    "(irrelevant) to 100 (must-read) based on how well it matches the researcher's "
    "active work, methods, and interests as described below. Prioritise direct "
    "overlap with their listed core research areas and active projects over "
    "generic topical similarity. Reward papers that would be genuinely useful to "
    "their current projects; penalise tangential matches even if they share "
    "buzzwords. Give a single concise sentence justifying the score. Always "
    "respond as a JSON object with keys 'score' (integer 0-100) and 'reason' "
    "(string). No prose outside the JSON.\n\n"
    "--- RESEARCHER PROFILE ---\n"
    "{profile}\n"
    "--- END PROFILE ---"
)


def _build_system_prompt(profile: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(profile=profile.strip())


def _build_user_prompt(paper: dict) -> str:
    abstract = (paper.get("abstract") or "").strip()
    if len(abstract) > 1800:
        abstract = abstract[:1800].rstrip() + "…"
    return (
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
    system_prompt: str,
    max_concurrent: int,
) -> dict[str, dict]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(max_concurrent)
    results: dict[str, dict] = {}

    async def worker(paper: dict) -> None:
        async with sem:
            user_prompt = _build_user_prompt(paper)
            parsed = await _score_one(client, model, system_prompt, user_prompt)
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

    # Prefer data/knowledgebase.md (rich KB) over the short profile_brief.
    kb_text, kb_hash = load_knowledgebase()
    if kb_text:
        profile = kb_text
        print(f"  llm_rerank: using knowledgebase.md ({len(kb_text)} chars, hash={kb_hash})")
    else:
        profile = (
            rcfg.get("profile_brief")
            or ", ".join((config.get("research_profile") or {}).get("tier1_keywords") or [])
        )
        print("  llm_rerank: no data/knowledgebase.md found; using profile_brief from config")

    system_prompt = _build_system_prompt(profile)

    cache = _load_cache(cache_path, model, kb_hash)
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
        new_results = asyncio.run(
            _score_many(fresh, model, system_prompt, max_concurrent)
        )
        cache.update(new_results)
        _save_cache(cache_path, model, kb_hash, cache)
        print(
            f"  llm_rerank: scored {len(new_results)}/{len(fresh)} new papers; "
            f"cache now has {len(cache)} entries"
        )

    # Apply scores back to the shortlisted papers.
    # Note: we deliberately do NOT copy `reason` onto the paper dict — reasons
    # are KB-grounded and may leak unpublished research details. They live only
    # in data/llm_cache.json (gitignored, restored from Actions cache in CI).
    applied = 0
    for p in shortlisted:
        entry = cache.get(p["id"])
        if not entry:
            p["llm_score"] = None
            continue
        p["llm_score"] = float(entry["score"])
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
