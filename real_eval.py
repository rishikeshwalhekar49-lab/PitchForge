"""Evaluate slide classification on REAL, hand-labeled pitch decks — the number reviewers will trust.

1. Put PDFs in a folder, e.g. eval/real_decks/
2. python -m eval.real_eval template eval/real_decks        -> writes eval/labels.csv with a guess per page
3. Open labels.csv, correct the `label` column by hand (use: problem, solution, market, business_model,
   competition, gtm, team, financials, traction, ask, other). Ideally 2 annotators -> report Cohen's kappa.
4. python -m eval.real_eval score eval/real_decks            -> eval/results/real_results.md

Tip: public decks (Airbnb, Uber, LinkedIn, Buffer, Front, YouTube ...) are widely mirrored; cite your source list.
"""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from app.ingest import classify_page, redact

LABELS = Path(__file__).parent / "labels.csv"
OUT = Path(__file__).parent / "results"


def pages(folder):
    for pdf in sorted(Path(folder).glob("*.pdf")):
        raw = [(p.extract_text() or "") for p in PdfReader(io.BytesIO(pdf.read_bytes())).pages]
        for i, t in enumerate(raw):
            yield pdf.name, i + 1, i / max(len(raw) - 1, 1), redact(" ".join(t.split()))


def template(folder):
    with open(LABELS, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "page", "label", "label_annotator2", "model_guess", "text_preview"])
        for name, pg, pos, txt in pages(folder):
            g = classify_page(txt, pos)[0]
            w.writerow([name, pg, g, "", g, txt[:120]])
    print(f"wrote {LABELS} — now correct the `label` column by hand")


def score(folder):
    gold = {(r["file"], int(r["page"])): r for r in csv.DictReader(open(LABELS))}
    X, pos, y, grp, y2 = [], [], [], [], []
    for name, pg, p, txt in pages(folder):
        r = gold.get((name, pg))
        if not r:
            continue
        X.append(txt); pos.append(p); y.append(r["label"]); grp.append(name)
        y2.append(r.get("label_annotator2") or None)
    rows = {}
    for m, use in (("keyword only", False), ("keyword + position (ours)", True)):
        pred = [classify_page(x, p if use else None)[0] for x, p in zip(X, pos)]
        rows[m] = (accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred)
    if len(set(grp)) >= 3:  # supervised, leave-one-deck-out
        pred = np.empty(len(y), dtype=object)
        for tr, te in LeaveOneGroupOut().split(X, y, grp):
            v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            clf = LogisticRegression(max_iter=2000).fit(v.fit_transform([X[i] for i in tr]), [y[i] for i in tr])
            pred[te] = clf.predict(v.transform([X[i] for i in te]))
        rows["TF-IDF + LogReg (supervised, leave-one-deck-out)"] = (accuracy_score(y, pred),
                                                                    f1_score(y, list(pred), average="macro"), list(pred))
    L = [f"# Real-deck evaluation\n\n{len(set(grp))} decks, {len(y)} labeled pages\n",
         "| Method | Accuracy | Macro-F1 |", "|---|---|---|"]
    L += [f"| {m} | {a:.3f} | {f:.3f} |" for m, (a, f, _) in rows.items()]
    pairs = [(a, b) for a, b in zip(y, y2) if b]
    if pairs:
        L.append(f"\nInter-annotator agreement (Cohen's κ, n={len(pairs)}): "
                 f"{cohen_kappa_score(*zip(*pairs)):.3f}")
    L += ["\n## Per-class report (ours)\n```", classification_report(y, rows["keyword + position (ours)"][2],
                                                                   zero_division=0), "```"]
    OUT.mkdir(exist_ok=True)
    (OUT / "real_results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"template": template, "score": score}[sys.argv[1]](sys.argv[2])
