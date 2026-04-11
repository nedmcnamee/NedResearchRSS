# SimonResearchRSS

A personalised academic paper feed that runs daily via GitHub Actions, scores incoming papers against your Paperpile library + a hand-tuned keyword profile, and publishes a static dashboard to GitHub Pages. Zero API cost — all scoring runs on CPU with local sentence embeddings.

## How it works

1. **Reference index** — `data/library.json` (your Paperpile export, 991 papers, 898 with abstracts) is encoded once with `sentence-transformers/all-MiniLM-L6-v2` and cached to `data/reference.npz`. The "added to Paperpile" timestamp is captured for recency weighting.
2. **Fetch** — daily, pull RSS feeds listed in `config.yaml` (Nature/Cell family journals + bioRxiv subject feeds).
3. **Dedupe** — drop any paper already in your library (matched by DOI, arXiv ID, or normalised title).
4. **Score** — for each candidate paper:
   - `embedding_score` = **recency-weighted** max cosine similarity to your library, mapped to 0–100. Each library paper's contribution is multiplied by `max(0.5 ** (years_since_added / half_life), floor)`, so recent additions count more than old ones.
   - `keyword_score`  = weighted substring match against Tier1 (30 pts) / Tier2 (15) / Tier3 (7) keywords, capped at 100
   - `final_score`    = 0.7 × embedding_score + 0.3 × keyword_score
   - Tier: high (≥60), medium (≥30), low (<30)
5. **(Optional) LLM rerank** — when enabled, papers above the medium tier are re-scored by the Anthropic API for cleaner ranking and a one-sentence justification. Cached by paper ID. See "Optional: LLM second-stage scoring" below.
6. **Source weighting** — each paper's final score is multiplied by its feed's `weight` (default 1.0). Noisy sources like arXiv use weights of 0.75–0.85, so their papers need a higher raw score to reach must-read tier — this surfaces the transferable gems from the ML firehose without drowning the top tier in preprints.
7. **Render** — write `docs/papers.json` and a self-contained `docs/index.html` with tier filter, text search, and read/unread tracking (localStorage).
8. **Deploy** — commit changes to main and publish `docs/` via GitHub Pages.

The dashboard supports marking papers as read (✓ button on each card) and hiding read papers ("Hide read" checkbox). Read state is stored in your browser's localStorage and survives daily updates because paper IDs are stable.

## Setup

1. **Push this repo to GitHub** (private is fine for Pages too, if you have Pro).
2. **Enable Pages**: Settings → Pages → Source: **GitHub Actions**.
3. **Trigger manually**: Actions → *Update Research Feed* → Run workflow.
4. Dashboard appears at `https://<username>.github.io/<repo-name>/`.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the reference index (one-time, or whenever library.json changes)
python src/build_reference.py

# Pull feeds, score, render
python src/fetch_and_score.py
python src/generate_dashboard.py

# Preview the dashboard
open docs/index.html
```

First run downloads the ~90 MB sentence-transformers model into `~/.cache/huggingface`. Subsequent runs reuse it.

## Customising

All knobs live in `config.yaml`:

- **Research profile keywords** — `research_profile.tier1_keywords` (30 pts each), `tier2_keywords` (15 pts), `tier3_keywords` (7 pts). Used for the keyword score component.
- **Scoring thresholds** — `scoring`:
  - `nn_sim_low` / `nn_sim_high`: weighted-cosine range that maps to 0–100. Raise the floor if irrelevant papers score too high.
  - `weights.nn` / `weights.keyword`: blend between semantic similarity and keyword matching.
  - `tiers.high` / `tiers.medium`: tier cutoffs.
- **Recency weighting** — `scoring.recency_half_life_years` (default 5) and `scoring.recency_floor` (default 0.2). Library papers added more recently count more; older ones decay exponentially with this half-life and never fall below the floor.
- **Lookback window** — `fetch.lookback_days` (default 14).

### Managing feeds

Feeds live in `config.yaml` → `feeds:` block. Each entry is one line with optional fields for source weighting:

```yaml
feeds:
  - {name: Nature, url: "https://www.nature.com/nature.rss"}
  - {name: arXiv cs.LG, url: "...", weight: 0.75, max_papers: 40}
```

| Field | Purpose |
|---|---|
| `name` | Display label on the dashboard |
| `url` | RSS/Atom URL |
| `weight` (optional, default `1.0`) | Multiplier applied to the paper's final score before tier assignment. Lower values demote noisy sources so they need a higher raw score to reach must-read. |
| `max_papers` (optional, default `fetch.max_papers_per_feed`) | Per-feed entry cap. Useful for high-volume feeds like arXiv. |

To **add a feed**: append a new line. The pipeline picks up changes on the next workflow run — no code changes needed.

**Currently fetching 18 feeds**:

| # | Feed | URL | Weight | Cap |
|---|---|---|---|---|
| 1 | Nature | `https://www.nature.com/nature.rss` | 1.0 | 100 |
| 2 | Nature Methods | `https://www.nature.com/nmeth.rss` | 1.0 | 100 |
| 3 | Nature Biotechnology | `https://www.nature.com/nbt.rss` | 1.0 | 100 |
| 4 | Nature Medicine | `https://www.nature.com/nm.rss` | 1.0 | 100 |
| 5 | Nature Communications | `https://www.nature.com/ncomms.rss` | 1.0 | 100 |
| 6 | Nature Machine Intelligence | `https://www.nature.com/natmachintell.rss` | 1.0 | 100 |
| 7 | Nature Cancer | `https://www.nature.com/natcancer.rss` | 1.0 | 100 |
| 8 | Nature Reviews Cancer | `https://www.nature.com/nrc.rss` | 1.0 | 100 |
| 9 | Cell | `https://www.cell.com/cell/inpress.rss` | 1.0 | 100 |
| 10 | Cancer Cell | `https://www.cell.com/cancer-cell/inpress.rss` | 1.0 | 100 |
| 11 | Molecular Cell | `https://www.cell.com/molecular-cell/inpress.rss` | 1.0 | 100 |
| 12 | bioRxiv Cancer Biology | `https://connect.biorxiv.org/biorxiv_xml.php?subject=cancer_biology` | 1.0 | 100 |
| 13 | bioRxiv Bioinformatics | `https://connect.biorxiv.org/biorxiv_xml.php?subject=bioinformatics` | 1.0 | 100 |
| 14 | bioRxiv Systems Biology | `https://connect.biorxiv.org/biorxiv_xml.php?subject=systems_biology` | 1.0 | 100 |
| 15 | bioRxiv Biochemistry | `https://connect.biorxiv.org/biorxiv_xml.php?subject=biochemistry` | 1.0 | 100 |
| 16 | arXiv cs.LG (ML) | `http://export.arxiv.org/api/query?...cs.LG...` | **0.75** | 40 |
| 17 | arXiv cs.CV (vision, pathology) | `http://export.arxiv.org/api/query?...cs.CV...` | **0.75** | 30 |
| 18 | arXiv q-bio.QM (comp biology) | `http://export.arxiv.org/api/query?...q-bio.QM...` | **0.85** | 40 |

**About arXiv**: arXiv is the firehose for cutting-edge ML and computational techniques, but it's not peer-reviewed and very high-volume. The three feeds above use arXiv's query API (the old `/rss/` endpoint returns zero entries — it's dead). Weights of 0.75–0.85 mean an arXiv paper needs a raw score of ~59–67 to clear the must-read threshold of 50, vs 50 for a journal paper. This surfaces the genuinely transferable gems (foundation model, multi-omics, vision-language for biology, etc.) without drowning the must-read tier in ML preprints.

**The query API format** lets you filter by any arXiv category:
```
http://export.arxiv.org/api/query?search_query=cat:<category>&sortBy=submittedDate&sortOrder=descending&max_results=40
```
Useful categories: `cs.LG` (ML), `cs.CV` (vision), `cs.AI` (AI broadly), `stat.ML` (statistical ML), `q-bio.QM` (quantitative methods in biology), `q-bio.GN` (genomics), `q-bio.BM` (biomolecules). You can also combine with `AND`/`OR`: `cat:cs.LG+AND+cat:q-bio.QM` for papers cross-listed between ML and comp biology.

**Suggested additional feeds** you may want:
- Mol. Cell. Proteomics, Bioinformatics, Nucleic Acids Res. (URLs change periodically — check the publisher's site)
- arXiv stat.ML — overlap with cs.LG but sometimes has unique papers
- OpenReview (ICLR/NeurIPS) — no RSS, but arXiv cs.LG already catches most of these

**If a feed returns 0 entries**: the URL has likely moved. `fetch_and_score.py` logs `got N entries` per feed every run, so dead feeds are easy to spot. Visit the publisher's site, find the new RSS link, update `config.yaml`, push.

### Optional: LLM second-stage scoring

The default pipeline (embeddings + keywords + recency) scores papers cheaply on CPU. You can optionally enable a second pass that re-scores shortlisted papers (those above the medium tier) using the Anthropic API. This produces a much cleaner medium-tier ranking plus a one-sentence justification per paper, at very low cost (~$3.60/month with Sonnet 4.6 default; ~$1.20/month with Haiku 4.5). Cached by paper ID across runs.

**One-time setup**:

1. Get an API key from https://console.anthropic.com/settings/keys
2. Add it to GitHub secrets:
   - Repo → **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-...`
3. Edit `config.yaml`:
   ```yaml
   scoring:
     llm_rerank:
       enabled: true                # flip this on
       model: claude-sonnet-4-6     # or claude-haiku-4-5 for ~3x cheaper
   ```
4. (Optional) Tune `scoring.llm_rerank.profile_brief` — a 2–3 sentence summary of your interests that the LLM uses as context. Ignored if you set up a private knowledgebase (next section).
5. Push and trigger the workflow manually.

**For local runs**: `export ANTHROPIC_API_KEY=sk-ant-...` before running `python src/fetch_and_score.py`.

### Private research knowledgebase (optional, sensitive content)

For much better LLM judgment, you can supply a rich personal research knowledgebase (active projects, methods, collaborators, keyword tiers, unpublished ideas, grant strategy). This is kept **out of the repo** because it may contain sensitive content — it's gitignored and restored into the CI runner from an encrypted GitHub Secret at workflow time.

**How it works**:
- The file lives locally at `data/knowledgebase.md` (gitignored — never committed).
- CI restores it from the `KB_CONTENT` secret into the runner's filesystem before `fetch_and_score.py` runs.
- The pipeline uses the KB as the LLM system prompt (replacing the short `profile_brief`) and parses its "Keywords for Paper Matching" section for tier1/2/3 keyword scoring.
- When the KB changes, its sha256 invalidates the LLM cache, so scores are recomputed under the new context.
- When `KB_CONTENT` is empty, the pipeline falls back gracefully to `profile_brief` from `config.yaml`.

**First-time setup**:

1. Create `data/knowledgebase.md` locally with sections like:
   ```
   # Personal Research Knowledgebase
   ## Research Identity
   ...
   ## Core Research Areas
   ### 1. ...
   ## Keywords for Paper Matching
   ### Tier 1: Direct match to active work
   keyword1, keyword2, ...
   ### Tier 2: Core methods and adjacent domains
   ...
   ### Tier 3: Broader interest
   ...
   ```
2. Push its content into the `KB_CONTENT` secret:
   ```bash
   gh secret set KB_CONTENT < data/knowledgebase.md
   ```
   Or via the GitHub web UI: Settings → Secrets and variables → Actions → New repository secret → name `KB_CONTENT`, paste the file contents.
3. Trigger the workflow; it will restore the KB from the secret and use it for scoring.

**Updating the KB**: edit `data/knowledgebase.md` locally, then resync:
```bash
gh secret set KB_CONTENT < data/knowledgebase.md
```
Next workflow run will pick up the change automatically (and invalidate the LLM cache).

**Privacy note**: GitHub Secrets are encrypted at rest, never shown in logs (even if your workflow tries to `echo` them), and only accessible to workflows running on branches you control. For a public repo, this is a reasonable balance between CI automation and content privacy.

If the API key is missing or `enabled: false`, the pipeline gracefully skips the LLM stage and produces a valid `papers.json` with the embedding+keyword scores.

## Refreshing your Paperpile library

1. Re-export your library from Paperpile as JSON.
2. Replace `data/library.json` with the new file.
3. Commit and push. The next workflow run will detect the hash change and rebuild `data/reference.npz`.

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | Feeds, keyword tiers, scoring parameters |
| `data/library.json` | Paperpile export (source of truth for your interests) |
| `data/reference.npz` | Cached library embeddings (auto-rebuilt on library change) |
| `data/reference_meta.json` | DOIs/titles/labels for dedupe + cache sentinel |
| `src/build_reference.py` | Encodes library → reference index |
| `src/fetch_and_score.py` | Daily pipeline: fetch → dedupe → score |
| `src/generate_dashboard.py` | Renders `docs/papers.json` → `docs/index.html` |
| `src/templates/dashboard.html.j2` | Jinja2 HTML template |
| `docs/` | GitHub Pages root (`papers.json` + `index.html`) |
| `.github/workflows/update-feed.yml` | Daily cron + Pages deploy |

## Notes

- RSS URLs for Nature and Cell journals change periodically. If a feed starts returning 0 entries, verify its URL at the publisher's site. `fetch_and_score.py` logs each feed's entry count on every run.
- The first real run is a good time to tune `nn_sim_low` / `nn_sim_high` — check the tier distribution in `docs/papers.json` and adjust if the "must-read" tier is empty or over-stuffed.
- Ideas for future iterations are in `SUMMARY.md` → "Possible extensions" (Slack webhook, read/unread tracking, LLM digest).
