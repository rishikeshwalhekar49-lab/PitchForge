"""SQLite storage + in-memory vector index over reference-deck pages.

- Documents are keyed by SHA-256 of their bytes, so re-uploading the same PDF is a no-op (idempotent ingest).
- The TF-IDF index is rebuilt lazily only when the corpus version changes.
"""
import json
import os
import sqlite3
import threading

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DB_PATH = os.environ.get("PITCHFORGE_DB", "pitchforge.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  sha TEXT PRIMARY KEY, filename TEXT, pages INTEGER, industry TEXT,
  slide_order TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_sha TEXT, page_no INTEGER,
  slide_type TEXT, confidence REAL, words INTEGER, text TEXT,
  FOREIGN KEY(doc_sha) REFERENCES documents(sha));
CREATE INDEX IF NOT EXISTS idx_pages_type ON pages(slide_type);
CREATE TABLE IF NOT EXISTS decks (
  id TEXT PRIMARY KEY, input TEXT, slides TEXT, analysis TEXT,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._index_version = -1
        self._vec = None
        self._matrix = None
        self._rows = []

    # ---------- corpus ----------
    def has_doc(self, sha: str) -> bool:
        return self._conn.execute("SELECT 1 FROM documents WHERE sha=?", (sha,)).fetchone() is not None

    def add_doc(self, sha, filename, industry, pages):
        """pages: list of dicts(page_no, slide_type, confidence, words, text). Atomic per document."""
        order = [p["slide_type"] for p in pages if p["slide_type"] != "other"]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO documents(sha, filename, pages, industry, slide_order) VALUES (?,?,?,?,?)",
                (sha, filename, len(pages), industry, json.dumps(order)))
            self._conn.executemany(
                "INSERT INTO pages(doc_sha, page_no, slide_type, confidence, words, text) VALUES (?,?,?,?,?,?)",
                [(sha, p["page_no"], p["slide_type"], p["confidence"], p["words"], p["text"]) for p in pages])
            self._bump_version()

    def _bump_version(self):
        v = int(self._meta("version") or 0) + 1
        self._conn.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (str(v),))

    def _meta(self, k):
        r = self._conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r["v"] if r else None

    def docs(self):
        return [dict(r) for r in self._conn.execute(
            "SELECT sha, filename, pages, industry, slide_order, created_at FROM documents ORDER BY created_at DESC")]

    def pages_of_type(self, slide_type):
        return [dict(r) for r in self._conn.execute(
            "SELECT p.*, d.filename, d.industry FROM pages p JOIN documents d ON d.sha=p.doc_sha WHERE slide_type=?",
            (slide_type,))]

    def all_orders(self):
        return [json.loads(r["slide_order"]) for r in self._conn.execute("SELECT slide_order FROM documents")]

    # ---------- vector search ----------
    def _ensure_index(self):
        v = int(self._meta("version") or 0)
        if v == self._index_version:
            return
        rows = [dict(r) for r in self._conn.execute(
            "SELECT p.id, p.slide_type, p.text, p.page_no, d.filename, d.industry "
            "FROM pages p JOIN documents d ON d.sha=p.doc_sha WHERE length(p.text) > 40")]
        self._rows = rows
        if rows:
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
            self._matrix = self._vec.fit_transform([r["text"] for r in rows])
        else:
            self._vec, self._matrix = None, None
        self._index_version = v

    def search(self, query: str, slide_type: str | None = None, k: int = 3, industry: str | None = None):
        with self._lock:
            self._ensure_index()
            if self._vec is None:
                return []
            scores = (self._matrix @ self._vec.transform([query]).T).toarray().ravel()
        results = []
        for i in np.argsort(-scores):
            r = self._rows[i]
            if slide_type and r["slide_type"] != slide_type:
                continue
            boost = 0.1 if industry and r["industry"] and industry.lower() in r["industry"].lower() else 0
            results.append({**r, "score": round(float(scores[i]) + boost, 4)})
            if len(results) >= k * 3:
                break
        results.sort(key=lambda x: -x["score"])
        return results[:k]

    # ---------- decks ----------
    def save_deck(self, deck_id, inp, slides, analysis):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO decks(id, input, slides, analysis, updated_at) VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                (deck_id, json.dumps(inp), json.dumps(slides), json.dumps(analysis)))

    def get_deck(self, deck_id):
        r = self._conn.execute("SELECT * FROM decks WHERE id=?", (deck_id,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "input": json.loads(r["input"]), "slides": json.loads(r["slides"]),
                "analysis": json.loads(r["analysis"]), "updated_at": r["updated_at"]}
