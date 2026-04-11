"""Encode the Paperpile library into a reference index for scoring.

Reads data/library.json, encodes each paper's (title + abstract) with
sentence-transformers, and saves:
  - data/reference.npz          (float32 vectors, shape [N, 384])
  - data/reference_meta.json    (doi, arxiv_id, title_norm, labels, year,
                                 journal, title for each row + a sentinel
                                 hash of library.json so CI can skip rebuild)

Skips rebuild if library.json's sha256 and the model name match the stored
sentinel; otherwise re-encodes from scratch.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

# Bump when the meta schema changes so old caches get rebuilt automatically.
META_VERSION = 2


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — used for fallback dedupe."""
    t = title.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_ids(entry: dict) -> tuple[str | None, str | None]:
    """Return (doi, arxiv_id) from a Paperpile entry, both lowercased/normalised."""
    doi = (entry.get("doi") or "").strip().lower() or None
    arxiv_id: str | None = None
    for item in entry.get("id_list") or []:
        if not isinstance(item, str):
            continue
        if item.startswith("doi:") and not doi:
            doi = item.removeprefix("doi:").strip().lower() or None
        elif item.startswith("arxivid:"):
            arxiv_id = item.removeprefix("arxivid:").strip().lower() or None
    return doi, arxiv_id


def prepare_entries(library: list[dict]) -> list[dict]:
    """Keep papers with non-empty title AND abstract; build metadata rows."""
    kept: list[dict] = []
    for entry in library:
        title = (entry.get("title") or "").strip()
        abstract = (entry.get("abstract") or "").strip()
        if not title or not abstract:
            continue
        doi, arxiv_id = extract_ids(entry)
        published = entry.get("published") or {}
        # `created` is a Unix timestamp (float) marking when the user added the
        # paper to Paperpile — used downstream for recency-weighted scoring.
        created = entry.get("created")
        added_ts: float | None = None
        if isinstance(created, (int, float)):
            added_ts = float(created)
        kept.append(
            {
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "title_norm": normalize_title(title),
                "labels": entry.get("labelsNamed") or [],
                "year": (published.get("year") or "").strip() if isinstance(published, dict) else "",
                "journal": (entry.get("journal") or "").strip(),
                "added_ts": added_ts,
            }
        )
    return kept


def load_cached_sentinel(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            data = json.load(f)
        return {
            "library_sha256": data.get("library_sha256"),
            "model": data.get("model"),
            "meta_version": data.get("meta_version"),
            "count": len(data.get("papers") or []),
        }
    except Exception:
        return None


def main() -> int:
    cfg = load_config()
    library_path = REPO_ROOT / cfg["library"]["path"]
    cache_path = REPO_ROOT / cfg["library"]["cache"]
    meta_path = REPO_ROOT / cfg["library"]["meta"]
    model_name = cfg["scoring"]["model"]

    if not library_path.exists():
        print(f"error: library file not found at {library_path}", file=sys.stderr)
        return 1

    library_hash = sha256_file(library_path)
    cached = load_cached_sentinel(meta_path)
    if (
        cached
        and cached.get("library_sha256") == library_hash
        and cached.get("model") == model_name
        and cached.get("meta_version") == META_VERSION
        and cache_path.exists()
    ):
        print(
            f"reference index up to date ({cached['count']} papers, "
            f"sha256={library_hash[:12]}, meta_version={META_VERSION}); skipping rebuild"
        )
        return 0

    print(f"loading library from {library_path}")
    with open(library_path) as f:
        library = json.load(f)
    print(f"  {len(library)} total entries")

    papers = prepare_entries(library)
    print(f"  {len(papers)} papers with title + abstract")
    if not papers:
        print("error: no usable papers in library", file=sys.stderr)
        return 1

    print(f"loading embedding model: {model_name}")
    from sentence_transformers import SentenceTransformer  # heavy import, keep local

    model = SentenceTransformer(model_name)
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    print(f"encoding {len(texts)} papers...")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  vectors shape: {vectors.shape}, dtype: {vectors.dtype}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, vectors=vectors)
    print(f"wrote {cache_path}")

    meta = {
        "library_sha256": library_hash,
        "model": model_name,
        "meta_version": META_VERSION,
        "papers": papers,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"wrote {meta_path}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
