import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pf")
    os.environ["PITCHFORGE_DB"] = str(tmp / "t.db")
    os.environ.pop("GEMINI_API_KEY", None); os.environ.pop("GOOGLE_API_KEY", None)
    subprocess.run([sys.executable, "cli.py", "make-samples", str(tmp / "decks")], cwd=ROOT, check=True)
    import importlib
    import app.store, app.main
    importlib.reload(app.store); importlib.reload(app.main)
    from fastapi.testclient import TestClient
    c = TestClient(app.main.app)
    c.decks = sorted((tmp / "decks").glob("*.pdf"))
    return c


def _upload(c, paths):
    return c.post("/api/references", files=[("files", (p.name, p.read_bytes(), "application/pdf")) for p in paths]).json()


def test_bulk_ingest_and_idempotency(client):
    r = _upload(client, client.decks)["results"]
    assert sum(x["status"] == "indexed" for x in r) == 12
    assert all(len(x["slides_detected"]) >= 8 for x in r)  # classifier finds most of the 10 slides
    again = _upload(client, client.decks[:3])["results"]
    assert all(x["status"] == "duplicate" for x in again)


def test_bad_files_do_not_break_batch(client):
    r = client.post("/api/references", files=[("files", ("x.pdf", b"not a pdf", "application/pdf")),
                                              ("files", ("y.pdf", b"%PDF-1.4 garbage", "application/pdf"))]).json()
    assert [x["status"] for x in r["results"]] == ["failed", "failed"]


def test_generate_edit_export(client):
    d = client.post("/api/decks", json={"idea": "Instant payments reconciliation for SMBs",
                                        "audience": "SMB finance teams", "industry": "fintech"}).json()
    assert [s["key"] for s in d["slides"]][0] == "problem" and len(d["slides"]) == 10
    assert any(s["references"] for s in d["slides"])  # grounded on corpus
    before = d["analysis"]["readiness"]["per_slide"]["market"]["score"]
    e = client.patch(f"/api/decks/{d['id']}/slides/market", json={
        "headline": "A $40B market", "bullets": ["TAM $40B", "SAM $6B", "SOM $900M"]}).json()
    assert e["analysis"]["market_math"]["values"]["SOM"] == 9e8
    assert e["analysis"]["readiness"]["per_slide"]["market"]["score"] > before
    assert client.get(f"/api/decks/{d['id']}/export.pptx").content[:2] == b"PK"


def test_consistency_catches_runway_lie():
    from app.analysis import consistency
    slides = [{"key": k, "headline": "", "bullets": []} for k in
              ["problem", "solution", "market", "business_model", "competition", "gtm", "team", "financials", "traction", "ask"]]
    slides[7]["bullets"] = ["Monthly burn: $100k"]
    slides[9]["bullets"] = ["Raising $1M", "Runway: 24 months", "60% product, 30% GTM"]
    issues = consistency(slides)
    assert any("10 months" in i for i in issues) and any("90%" in i for i in issues)
