"""Command-line entry points.

  python cli.py ingest ./sample_decks          # bulk-index a folder of PDFs (parallel, idempotent)
  python cli.py generate --idea "..." --audience "..." --industry fintech --out deck.pptx
  python cli.py make-samples                   # create 12 synthetic reference decks for testing
"""
import argparse
import json
import os
import time
from pathlib import Path

from app import analysis, export, generator
from app.ingest import ingest_many
from app.store import Store


def cmd_ingest(a):
    store = Store()
    files = [(p.name, p.read_bytes()) for p in sorted(Path(a.folder).rglob("*.pdf"))]
    t = time.time()
    report = ingest_many(store, files, workers=a.workers)
    for r in report:
        print(json.dumps(r))
    ok = sum(r["status"] == "indexed" for r in report)
    print(f"\n{ok} indexed, {len(report) - ok} skipped/failed, {len(files)} files in {time.time() - t:.2f}s")


def cmd_generate(a):
    store = Store()
    deck = generator.generate(store, {"idea": a.idea, "audience": a.audience,
                                      "industry": a.industry, "company": a.company})
    deck["analysis"] = analysis.analyze(store, deck)
    store.save_deck(deck["id"], deck["input"], deck["slides"], deck["analysis"])
    Path(a.out).write_bytes(export.to_pptx(deck)) if a.out.endswith(".pptx") else \
        Path(a.out).write_text(export.to_markdown(deck))
    r = deck["analysis"]["readiness"]
    print(f"Deck {deck['id']} ({deck['engine']}, grounded on {deck['corpus_decks']} decks) -> {a.out}")
    print(f"Readiness {r['overall']}/100 · weakest: {', '.join(r['weakest'])}")


def cmd_samples(a):
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
    from app.schema import SLIDES
    os.makedirs(a.folder, exist_ok=True)
    cos = [("PayLoop", "fintech", "payments"), ("MediQ", "healthtech", "patient triage"),
           ("LearnLab", "edtech", "student tutoring"), ("FlowOps", "saas", "B2B workflow"),
           ("Stallr", "marketplace", "buyers and sellers"), ("GridZero", "climate", "solar energy"),
           ("Vibe", "consumer", "creator app"), ("LendBee", "fintech", "SMB lending"),
           ("CareLink", "healthtech", "clinic scheduling"), ("DataDock", "saas", "API dashboard"),
           ("Roomy", "marketplace", "hosts listing"), ("CarbonIQ", "climate", "carbon emission")]
    for name, ind, topic in cos:
        c = canvas.Canvas(f"{a.folder}/{name}.pdf", pagesize=landscape(letter))
        for key, title, kws in SLIDES:
            c.setFont("Helvetica-Bold", 28); c.drawString(50, 540, title)
            c.setFont("Helvetica", 16)
            lines = [f"{name} {kws[0]} for {topic} in {ind}.", f"Key point about {kws[1]} and {kws[-1]}.",
                     f"{name} reached $1.2M ARR with 40% MoM growth." if key == "traction" else
                     f"TAM $40B, SAM $6B, SOM $300M" if key == "market" else f"Detail on {kws[2]}."]
            for i, l in enumerate(lines):
                c.drawString(60, 470 - i * 40, l)
            c.showPage()
        c.save()
    print(f"wrote {len(cos)} sample decks to {a.folder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    i = sub.add_parser("ingest"); i.add_argument("folder"); i.add_argument("--workers", type=int, default=os.cpu_count())
    i.set_defaults(fn=cmd_ingest)
    g = sub.add_parser("generate")
    for f in ("idea", "audience", "industry"):
        g.add_argument(f"--{f}", required=True)
    g.add_argument("--company"); g.add_argument("--out", default="deck.pptx"); g.set_defaults(fn=cmd_generate)
    s = sub.add_parser("make-samples"); s.add_argument("folder", nargs="?", default="sample_decks")
    s.set_defaults(fn=cmd_samples)
    a = p.parse_args(); a.fn(a)
