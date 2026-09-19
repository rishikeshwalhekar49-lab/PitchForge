"""PitchForge evaluation suite — produces the tables for the paper.

  python -m eval.run_eval            # all experiments, ~1-2 min
  python -m eval.run_eval --quick    # smaller sizes, for CI

Outputs: eval/results/results.json, results.md, tables.tex
E1  Cross-slide numerical consistency checking (planted contradictions)  -> P / R / F1 per type
E2  Slide-type classification ablation (keyword vs +position vs supervised) -> accuracy, macro-F1, 95% CI
E3  Deck DNA order recovery vs. fixed template                         -> Kendall tau vs #decks
E4  Ingestion throughput & idempotency                                   -> pages/s, speedup, re-ingest time
"""
import argparse
import json
import os
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from app.analysis import consistency, deck_dna, market_math
from app.ingest import classify_page, ingest_many
from app.schema import SLIDE_KEYS
from eval import synth

OUT = Path(__file__).parent / "results"


def bootstrap_ci(y, p, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.array(y), np.array(p)
    vals = [metric(y[idx], p[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# ------------------------------------------------------------------ E1
def _detect(slides):
    """Map checker output strings to contradiction types."""
    found = set()
    for msg in consistency(slides) + market_math(slides)["issues"]:
        m = msg.lower()
        if "months" in m and "burn" in m: found.add("runway_mismatch")
        elif "use-of-funds" in m: found.add("funds_not_100")
        elif "sam is larger than tam" in m: found.add("sam_gt_tam")
        elif "som is" in m or "som exceeds" in m: found.add("som_share")
        elif "pre-revenue" in m: found.add("prerev_vs_financials")
    return found


def e1_consistency(n_per_type=200, n_clean=500, seed=1):
    rng = random.Random(seed)
    cases = [(synth.contradiction_deck(rng, t), t) for t in synth.CONTRA_TYPES for _ in range(n_per_type)]
    cases += [(synth.contradiction_deck(rng, None), None) for _ in range(n_clean)]
    t0 = time.perf_counter()
    preds = [_detect(s) for s, _ in cases]
    ms = (time.perf_counter() - t0) * 1000 / len(cases)
    per = {}
    for t in synth.CONTRA_TYPES:
        tp = sum(t in p and g == t for p, (_, g) in zip(preds, cases))
        fp = sum(t in p and g != t for p, (_, g) in zip(preds, cases))
        fn = sum(t not in p and g == t for p, (_, g) in zip(preds, cases))
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        per[t] = {"precision": round(P, 4), "recall": round(R, 4),
                  "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0, "support": n_per_type}
    clean_fp = sum(bool(p) for p, (_, g) in zip(preds, cases) if g is None) / n_clean
    # deck-level: did we flag the deck at all?
    y = [g is not None for _, g in cases]; yhat = [bool(p) for p in preds]
    return {"per_type": per, "macro_f1": round(statistics.mean(v["f1"] for v in per.values()), 4),
            "clean_deck_false_alarm_rate": round(clean_fp, 4),
            "deck_level_f1": round(f1_score(y, yhat), 4), "ms_per_deck": round(ms, 3),
            "baseline_flag_everything_f1": round(f1_score(y, [True] * len(y)), 4),
            "n_decks": len(cases)}


def e1_llm_baseline(n=100, seed=11):
    """Optional: ask the LLM to find contradictions (needs GEMINI_API_KEY). Same planted decks, same scoring."""
    from app import llm
    if llm.client() is None:
        return {"skipped": "set GEMINI_API_KEY to run the LLM baseline"}
    rng = random.Random(seed)
    types = synth.CONTRA_TYPES + [None]
    cases = [(synth.contradiction_deck(rng, t), t) for t in types for _ in range(n // len(types))]
    hit = fa = pos = neg = 0
    t0 = time.perf_counter()
    for slides, g in cases:
        deck = {s["key"]: s["bullets"] for s in slides if s["bullets"]}
        out = llm.generate_json("Does this pitch deck contain any numerical contradiction or implausible number "
                                '(runway math, percentages, TAM/SAM/SOM funnel, revenue vs traction)? Return JSON '
                                '{"contradiction": true|false, "reason": str}. DECK: ' + json.dumps(deck)) or {}
        flag = bool(out.get("contradiction"))
        if g: pos += 1; hit += flag
        else: neg += 1; fa += flag
    P = hit / (hit + fa) if hit + fa else 0; R = hit / pos
    return {"deck_level_precision": round(P, 4), "deck_level_recall": round(R, 4),
            "deck_level_f1": round(2 * P * R / (P + R), 4) if P + R else 0,
            "clean_false_alarm_rate": round(fa / neg, 4), "s_per_deck": round((time.perf_counter() - t0) / len(cases), 2),
            "model": llm.FAST_MODEL, "n": len(cases)}


# ------------------------------------------------------------------ E2
def e2_classifier(n_decks=200, seed=7):
    rows = synth.classifier_corpus(n_decks=n_decks, seed=seed)
    X = [r[0] for r in rows]; pos = [r[1] for r in rows]; y = [r[2] for r in rows]; g = [r[3] for r in rows]
    res = {}
    for name, use_pos in (("keyword_only", False), ("keyword_plus_position (ours)", True)):
        p = [classify_page(x, ps if use_pos else None)[0] for x, ps in zip(X, pos)]
        res[name] = {"accuracy": round(accuracy_score(y, p), 4),
                     "macro_f1": round(f1_score(y, p, average="macro", labels=SLIDE_KEYS), 4),
                     "acc_95ci": bootstrap_ci(y, p, accuracy_score), "needs_labels": False}
    # supervised upper bound: TF-IDF + logistic regression, deck-grouped 5-fold CV (no deck leaks across folds)
    preds = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, g):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform([X[i] for i in tr]), [y[i] for i in tr])
        preds[te] = clf.predict(vec.transform([X[i] for i in te]))
    res["tfidf_logreg_supervised"] = {"accuracy": round(accuracy_score(y, preds), 4),
                                      "macro_f1": round(f1_score(y, list(preds), average="macro"), 4),
                                      "acc_95ci": bootstrap_ci(y, list(preds), accuracy_score), "needs_labels": True}
    # robustness: vary title presence (titles are the strongest cue)
    sweep = {}
    for tr in (0.0, 0.3, 0.6, 1.0):
        r2 = synth.classifier_corpus(n_decks=n_decks // 2, seed=seed + 1, title_rate=tr)
        sweep[str(tr)] = {m: round(accuracy_score([r[2] for r in r2],
                                                  [classify_page(r[0], r[1] if up else None)[0] for r in r2]), 4)
                          for m, up in (("keyword_only", False), ("ours", True))}
    return {"methods": res, "title_rate_sweep": sweep, "n_pages": len(y)}


# ------------------------------------------------------------------ E3
def kendall_tau(a, b):
    ra = {k: i for i, k in enumerate(a)}; rb = {k: i for i, k in enumerate(b)}
    keys = list(ra); c = d = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s = (ra[keys[i]] - ra[keys[j]]) * (rb[keys[i]] - rb[keys[j]])
            c += s > 0; d += s < 0
    return (c - d) / (c + d)


class _Stub:
    def __init__(self, decks): self.decks = decks
    def all_orders(self): return [o for _, o in self.decks]
    def docs(self): return [{"industry": i, "slide_order": json.dumps(o)} for i, o in self.decks]


def e3_deck_dna(trials=30, seed=3):
    rng = random.Random(seed)
    out = {}
    for n in (3, 5, 10, 20, 40):
        learned, fixed = [], []
        for _ in range(trials):
            decks = [(ind, synth.noisy_order(rng, base)) for ind, base in synth.INDUSTRY_ORDERS.items()
                     for _ in range(n)]
            for ind, truth in synth.INDUSTRY_ORDERS.items():
                learned.append(kendall_tau(deck_dna(_Stub(decks), ind)["recommended_order"], truth))
                fixed.append(kendall_tau(SLIDE_KEYS, truth))
        out[n] = {"deck_dna_tau": round(statistics.mean(learned), 4),
                  "deck_dna_tau_sd": round(statistics.pstdev(learned), 4),
                  "fixed_template_tau": round(statistics.mean(fixed), 4)}
    return {"decks_per_industry": out, "trials": trials}


# ------------------------------------------------------------------ E4
def _make_pdfs(folder, n, pages=15, seed=5):
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
    rng = random.Random(seed)
    for d in range(n):
        c = canvas.Canvas(f"{folder}/deck_{d:03d}.pdf", pagesize=landscape(letter))
        for p in range(pages):
            key = SLIDE_KEYS[p % 10]
            c.setFont("Helvetica", 13)
            for i, line in enumerate(synth.page(rng, key, True, 0.3).split(". ")):
                c.drawString(40, 560 - i * 22, line[:110])
            c.drawString(40, 60, f"deck {d} page {p} contact founder{d}@example.com")
            c.showPage()
        c.save()


def e4_throughput(sizes=(10, 25, 50, 100)):
    from app import ingest as ing
    from app.store import Store
    cpus = os.cpu_count() or 4
    t = time.perf_counter()
    list(ing._pool(cpus).map(abs, range(cpus)))  # warm the persistent pool, report its one-off cost
    cold = round(time.perf_counter() - t, 3)
    out = {}
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            _make_pdfs(tmp, n)
            files = [(p.name, p.read_bytes()) for p in sorted(Path(tmp).glob("*.pdf"))]
            row = {}
            for workers in (1, cpus):
                store = Store(f"{tmp}/w{workers}.db")
                t = time.perf_counter(); rep = ingest_many(store, files, workers=workers)
                dt = time.perf_counter() - t
                row[f"workers_{workers}_s"] = round(dt, 3)
                row[f"workers_{workers}_pages_per_s"] = round(n * 15 / dt, 1)
                ok = sum(r["status"] == "indexed" for r in rep)
            t = time.perf_counter(); rep2 = ingest_many(store, files, workers=cpus)
            row["reingest_s"] = round(time.perf_counter() - t, 4)
            row["reingest_all_duplicates"] = all(r["status"] == "duplicate" for r in rep2)
            row["indexed"] = ok
            row["pii_leaked"] = sum("@example.com" in p["text"] for k in SLIDE_KEYS for p in store.pages_of_type(k))
            row["speedup"] = round(row["workers_1_s"] / row[f"workers_{cpus}_s"], 2)
            out[n] = row
    return {"by_n_decks": out, "cpus": cpus, "pages_per_deck": 15, "pool_cold_start_s": cold}


# ------------------------------------------------------------------ report
def to_markdown(R):
    L = ["# PitchForge evaluation results\n", "_All data synthetic & seeded unless stated. See eval/README.md._\n"]
    e1 = R["E1"]
    L += ["## E1 · Cross-slide numerical consistency", "| Contradiction | Precision | Recall | F1 |", "|---|---|---|---|"]
    L += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |" for k, v in e1["per_type"].items()]
    L += [f"\nMacro-F1 **{e1['macro_f1']:.3f}** · deck-level F1 {e1['deck_level_f1']:.3f} "
          f"(flag-everything baseline {e1['baseline_flag_everything_f1']:.3f}) · false alarms on clean decks "
          f"{e1['clean_deck_false_alarm_rate']:.1%} · {e1['ms_per_deck']} ms/deck · n={e1['n_decks']}\n"]
    b = R.get("E1_llm", {})
    L.append(f"LLM baseline ({b['model']}, n={b['n']}): deck-level F1 {b['deck_level_f1']:.3f}, "
             f"false alarms {b['clean_false_alarm_rate']:.1%}, {b['s_per_deck']} s/deck\n" if "model" in b
             else f"LLM baseline: {b.get('skipped', 'n/a')}\n")
    L += ["## E2 · Slide-type classification", "| Method | Labels needed | Accuracy (95% CI) | Macro-F1 |", "|---|---|---|---|"]
    for k, v in R["E2"]["methods"].items():
        L.append(f"| {k} | {'yes' if v['needs_labels'] else 'no'} | {v['accuracy']:.3f} "
                 f"[{v['acc_95ci'][0]:.3f}, {v['acc_95ci'][1]:.3f}] | {v['macro_f1']:.3f} |")
    L += ["\nTitle-presence sweep (accuracy):", "| Title rate | keyword only | ours |", "|---|---|---|"]
    L += [f"| {k} | {v['keyword_only']:.3f} | {v['ours']:.3f} |" for k, v in R["E2"]["title_rate_sweep"].items()]
    L += ["\n## E3 · Deck DNA slide-order recovery (Kendall τ vs. true order)",
          "| Decks / industry | Deck DNA τ (±sd) | Fixed template τ |", "|---|---|---|"]
    L += [f"| {n} | {v['deck_dna_tau']:.3f} ± {v['deck_dna_tau_sd']:.3f} | {v['fixed_template_tau']:.3f} |"
          for n, v in R["E3"]["decks_per_industry"].items()]
    c = R["E4"]["cpus"]
    L += [f"\n## E4 · Ingestion throughput ({c} CPUs, 15 pages/deck)",
          f"Persistent pool cold start (one-off per server): {R['E4']['pool_cold_start_s']} s\n",
          f"| Decks | 1 worker (s) | {c} workers (s) | Speedup | Pages/s | Re-ingest (s) | PII leaked |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {n} | {v['workers_1_s']} | {v[f'workers_{c}_s']} | {v['speedup']}× | "
          f"{v[f'workers_{c}_pages_per_s']} | {v['reingest_s']} | {v['pii_leaked']} |"
          for n, v in R["E4"]["by_n_decks"].items()]
    return "\n".join(L) + "\n"


def to_latex(R):
    e1 = R["E1"]["per_type"]
    t1 = "\n".join(f"{k.replace('_', ' ')} & {v['precision']:.3f} & {v['recall']:.3f} & {v['f1']:.3f} \\\\"
                   for k, v in e1.items())
    t2 = "\n".join(f"{k.replace('_', ' ').replace('(ours)', '(ours)')} & {v['accuracy']:.3f} & {v['macro_f1']:.3f} \\\\"
                   for k, v in R["E2"]["methods"].items())
    t3 = "\n".join(f"{n} & {v['deck_dna_tau']:.3f} & {v['fixed_template_tau']:.3f} \\\\"
                   for n, v in R["E3"]["decks_per_industry"].items())
    tab = lambda cap, lab, cols, head, body: (  # noqa: E731
        f"\\begin{{table}}[t]\\centering\\caption{{{cap}}}\\label{{{lab}}}\n\\begin{{tabular}}{{{cols}}}\n"
        f"\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return (tab("Cross-slide contradiction detection.", "tab:e1", "lccc", "Type & P & R & F1", t1)
            + tab("Slide-type classification.", "tab:e2", "lcc", "Method & Acc. & Macro-F1", t2)
            + tab("Slide-order recovery (Kendall $\\tau$).", "tab:e3", "lcc", "Decks/industry & Deck DNA & Fixed", t3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    q = a.quick
    R = {}
    print("E1 consistency…"); R["E1"] = e1_consistency(40 if q else 200, 100 if q else 500)
    print("E1b LLM baseline…"); R["E1_llm"] = e1_llm_baseline(30 if q else 120)
    print("E2 classifier…");  R["E2"] = e2_classifier(40 if q else 200)
    print("E3 deck DNA…");    R["E3"] = e3_deck_dna(5 if q else 30)
    print("E4 throughput…");  R["E4"] = e4_throughput((10,) if q else (10, 25, 50, 100))
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(R, indent=2))
    (OUT / "results.md").write_text(to_markdown(R))
    (OUT / "tables.tex").write_text(to_latex(R))
    print(to_markdown(R))
