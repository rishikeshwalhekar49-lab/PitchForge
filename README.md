# PitchForge — Startup Pitch Builder

Turn a raw business idea into a **fundable, fully editable 10-slide investor pitch** in minutes. Each slide is grounded in real reference pitch decks you upload, and the deck is checked the way a VC would check it.

```
idea + audience + industry ──► retrieve similar slides from indexed decks ──► draft (Gemini Flash / offline templates)
                                                                               │
          .pptx / .md export ◄── live edit + re-score ◄── investor checks (Gemini Pro red-team + rule engines)
```

## The 10-slide framework
Problem · Solution · Market Size (TAM/SAM/SOM) · Business Model · Competitive Landscape · Go-To-Market · Team · Financial Projections · Traction · Funding Ask

## Reference-deck pipeline (built for 10+ PDFs at a time)
| Stage | What it does |
|---|---|
| **Dedupe** | SHA-256 of each file's bytes, checked *before* parsing, so re-uploads are free and ingest can be retried safely |
| **Parallel parse** | `ProcessPoolExecutor` across CPU cores. One corrupt or encrypted PDF is reported as failed and never breaks the batch |
| **Redact** | Emails, phone numbers and API-key-shaped secrets are removed before anything is stored |
| **Classify** | Each page is labeled as one of the 10 slide types using keyword scores (titles weighted 3x) plus where the page sits in the deck |
| **Store** | SQLite (WAL), one transaction per document |
| **Index** | TF-IDF (1–2-grams) vector index, rebuilt lazily only when the corpus changes |
| **Use** | Retrieves the top matching reference slides for each slide (with citations), per-slide word-count benchmarks, and the slide order decks most often use |

Measured: 12 decks indexed in about 3.5 s on a laptop; re-ingesting the same 12 takes 0.00 s.

## What's new here
1. **Deck DNA**: learns the slide order the reference decks actually use (mean normalized position), per industry when at least 3 decks match.
2. **Market Math Guard**: reads the $ values for TAM/SAM/SOM and flags impossible funnels (SAM > TAM) and unrealistic capture (SOM > 20% of SAM).
3. **Cross-Slide Lie Detector**: checks that ask ÷ burn matches the stated runway, that use-of-funds adds up to 100%, and that pre-revenue traction doesn't sit next to $1M+ year-1 revenue.
4. **Evidence Gap Map**: counts every `[ASSUMPTION]` placeholder per slide, which gives you your diligence to-do list.
5. **Corpus-benchmarked Readiness Score**: scores specificity and length against the uploaded decks, not a fixed rubric. It re-scores live after every edit.
6. **VC Red-Team**: a Pro-tier model asks the toughest questions about your 3 weakest slides and names the single biggest risk that could sink the round.
7. **Citations on every slide**: shows which reference deck and page shaped each slide.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add GEMINI_API_KEY
python cli.py make-samples      # 12 synthetic reference decks (or use your own PDFs)
python cli.py ingest sample_decks
uvicorn app.main:app --reload   # open http://localhost:8000
```
CLI only: `python cli.py generate --idea "..." --audience "..." --industry fintech --out deck.pptx`

Docker: `docker build -t pitchforge . && docker run -p 8080:8080 -v $PWD/data:/data pitchforge`

## API
| Method | Path | |
|---|---|---|
| POST | `/api/references` | multipart upload of many PDFs → per-file report |
| GET | `/api/references` | indexed decks + Deck DNA |
| GET | `/api/search?q=&slide_type=` | semantic search over reference slides |
| POST | `/api/decks` | `{idea, audience, industry, company?}` → deck + analysis |
| PATCH | `/api/decks/{id}/slides/{key}` | edit a slide → re-analyzed deck |
| GET | `/api/decks/{id}/export.pptx` / `.md` | editable exports (with speaker notes) |

## Tests
`pytest -q` covers bulk ingest, idempotency, bad files, generation, edit + re-score, pptx export, and the consistency checks.

## Roadmap
OCR for scanned decks (Document AI / Tesseract) · Gemini / Vertex embeddings behind the same `Store.search` interface · BigQuery vector search for multi-tenant scale · slide design extraction from images.
