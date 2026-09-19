"""FastAPI app: ingest reference decks, generate, edit, analyze and export pitch frameworks."""
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from . import analysis, export, generator
from .ingest import ingest_many
from .schema import SLIDE_KEYS
from .store import Store

app = FastAPI(title="PitchForge", version="1.0")
store = Store()
STATIC = Path(__file__).parent.parent / "static"
MAX_PDF_MB = 50


class IdeaIn(BaseModel):
    idea: str = Field(min_length=10)
    audience: str = Field(min_length=2)
    industry: str = Field(min_length=2)
    company: str | None = None
    stage: str | None = "seed"


class SlideEdit(BaseModel):
    headline: str | None = None
    bullets: list[str] | None = None
    speaker_notes: str | None = None


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.post("/api/references")
async def upload_references(files: list[UploadFile] = File(...)):
    good, bad = [], []
    for f in files:
        data = await f.read()
        if len(data) > MAX_PDF_MB * 1024 * 1024:
            bad.append({"filename": f.filename, "status": "failed", "error": f"exceeds {MAX_PDF_MB}MB"})
        elif not data.startswith(b"%PDF-"):
            bad.append({"filename": f.filename, "status": "failed", "error": "not a PDF"})
        else:
            good.append((f.filename, data))
    return {"results": bad + ingest_many(store, good, workers=os.cpu_count() or 4)}


@app.get("/api/references")
def list_references():
    return {"documents": store.docs(), "dna": analysis.deck_dna(store)}


@app.get("/api/search")
def search(q: str, slide_type: str | None = None, k: int = 5):
    return {"results": store.search(q, slide_type, k)}


@app.post("/api/decks")
def create_deck(inp: IdeaIn):
    deck = generator.generate(store, inp.model_dump())
    deck["analysis"] = analysis.analyze(store, deck)
    store.save_deck(deck["id"], deck["input"], deck["slides"], deck["analysis"])
    return deck


def _load(deck_id):
    deck = store.get_deck(deck_id)
    if not deck:
        raise HTTPException(404, "deck not found")
    return deck


@app.get("/api/decks/{deck_id}")
def get_deck(deck_id: str):
    return _load(deck_id)


@app.patch("/api/decks/{deck_id}/slides/{key}")
def edit_slide(deck_id: str, key: str, edit: SlideEdit):
    if key not in SLIDE_KEYS:
        raise HTTPException(400, "unknown slide")
    deck = _load(deck_id)
    slide = next(s for s in deck["slides"] if s["key"] == key)
    slide.update({k: v for k, v in edit.model_dump().items() if v is not None})
    deck["analysis"] = analysis.analyze(store, deck)  # live re-scoring after every edit
    store.save_deck(deck_id, deck["input"], deck["slides"], deck["analysis"])
    return deck


@app.get("/api/decks/{deck_id}/export.pptx")
def export_pptx(deck_id: str):
    return Response(export.to_pptx(_load(deck_id)),
                    media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    headers={"Content-Disposition": f'attachment; filename="pitch-{deck_id}.pptx"'})


@app.get("/api/decks/{deck_id}/export.md", response_class=PlainTextResponse)
def export_md(deck_id: str):
    return export.to_markdown(_load(deck_id))


@app.get("/healthz")
def health():
    return {"ok": True, "decks_indexed": len(store.docs())}
