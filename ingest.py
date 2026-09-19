"""Parallel PDF ingestion pipeline: hash -> dedupe -> extract -> redact -> classify -> index."""
import hashlib
import io
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from pypdf import PdfReader

from .schema import KEYWORDS, SLIDE_KEYS

INDUSTRIES = {
    "fintech": ["payment", "bank", "lending", "fintech", "card", "credit"],
    "healthtech": ["patient", "health", "clinic", "medical", "doctor", "care"],
    "edtech": ["student", "learning", "school", "education", "course", "teacher"],
    "saas": ["saas", "b2b", "workflow", "enterprise", "dashboard", "api"],
    "marketplace": ["marketplace", "buyers", "sellers", "gmv", "listing", "hosts"],
    "climate": ["carbon", "climate", "energy", "solar", "emission", "ev"],
    "consumer": ["consumer", "app", "social", "creator", "brand", "d2c"],
}

_REDACT = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[PHONE]"),
    (re.compile(r"\b(?:sk|AKIA|AIza)[A-Za-z0-9_\-]{12,}\b"), "[SECRET]"),
]


def redact(text: str) -> str:
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    return text


def classify_page(text: str, position: float) -> tuple[str, float]:
    """Keyword scoring with a positional prior (problem early, ask late). Returns (slide_type, confidence)."""
    t = text.lower()
    head = t[:160]  # slide titles live at the top: weigh them 3x
    scores = {}
    for i, key in enumerate(SLIDE_KEYS):
        s = sum(t.count(kw) + 2 * head.count(kw) for kw in KEYWORDS[key])
        expected = i / (len(SLIDE_KEYS) - 1)
        scores[key] = s * (1.0 + 0.5 * (1 - abs(position - expected)))
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if scores[best] < 1.5:
        return "other", 0.0
    return best, round(scores[best] / total, 3)


def detect_industry(text: str) -> str:
    t = text.lower()
    scored = {k: sum(t.count(w) for w in ws) for k, ws in INDUSTRIES.items()}
    best = max(scored, key=scored.get)
    return best if scored[best] >= 3 else "general"


def parse_pdf(data: bytes, filename: str) -> dict:
    """Runs in a worker process. Never raises: returns an error field instead so one bad PDF can't sink a batch."""
    sha = hashlib.sha256(data).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        raw = [(p.extract_text() or "") for p in reader.pages]
    except Exception as e:  # corrupt / password-protected
        return {"sha": sha, "filename": filename, "error": f"{type(e).__name__}: {e}"}
    n = max(len(raw), 1)
    pages = []
    for i, txt in enumerate(raw):
        txt = redact(re.sub(r"\s+", " ", txt).strip())
        stype, conf = classify_page(txt, i / max(n - 1, 1))
        pages.append({"page_no": i + 1, "slide_type": stype, "confidence": conf,
                      "words": len(txt.split()), "text": txt[:4000]})
    scanned = sum(p["words"] for p in pages) < 5 * n
    return {"sha": sha, "filename": filename, "pages": pages,
            "industry": detect_industry(" ".join(p["text"] for p in pages)),
            "warning": "little extractable text (scanned PDF? add OCR)" if scanned else None}


def ingest_many(store, files: list[tuple[str, bytes]], workers: int = 4) -> list[dict]:
    """files: [(filename, bytes)]. Dedupes by hash before parsing, parses in parallel, writes atomically per doc."""
    report, todo, seen = [], [], set()
    for name, data in files:
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen or store.has_doc(sha):
            report.append({"filename": name, "status": "duplicate", "sha": sha[:12]})
            continue
        seen.add(sha)
        todo.append((name, data))
    if not todo:
        return report
    if len(todo) == 1 or workers <= 1:
        results = [parse_pdf(d, n) for n, d in todo]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            futs = [ex.submit(parse_pdf, d, n) for n, d in todo]
            results = [f.result() for f in as_completed(futs)]
    for r in results:
        if r.get("error"):
            report.append({"filename": r["filename"], "status": "failed", "error": r["error"]})
            continue
        store.add_doc(r["sha"], r["filename"], r["industry"], r["pages"])
        found = sorted({p["slide_type"] for p in r["pages"]} - {"other"})
        report.append({"filename": r["filename"], "status": "indexed", "sha": r["sha"][:12],
                       "pages": len(r["pages"]), "industry": r["industry"],
                       "slides_detected": found, "warning": r["warning"]})
    return report
