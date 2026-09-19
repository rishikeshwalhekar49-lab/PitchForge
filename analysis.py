"""Innovation layer — checks most pitch generators don't do.

1. Deck DNA          – learns the slide ORDER winning decks use (per industry) via mean normalized rank.
2. Market Math Guard – parses TAM/SAM/SOM values and flags impossible or implausible funnels.
3. Cross-Slide Lie Detector – ask vs burn vs runway, use-of-funds summing to 100%, traction vs financials.
4. Evidence Gap Map  – counts unverified [ASSUMPTION]/[X] placeholders per slide = diligence to-do list.
5. Readiness Score   – benchmarks specificity & length against the reference corpus, not a fixed rubric.
6. VC Red-Team       – the hardest questions an investor will ask about your weakest slides (LLM Pro tier).
"""
import json
import re
import statistics

from . import llm
from .schema import SLIDE_KEYS, SLIDE_TITLES

_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s*(k|m|mm|b|bn|t|thousand|million|billion|trillion)?", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6,
         "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12, "trillion": 1e12}


def money(text: str) -> list[float]:
    return [float(n) * _MULT.get((u or "").lower(), 1) for n, u in _MONEY.findall(text)]


def _slide_text(s):
    return " ".join([s.get("headline", "")] + list(s.get("bullets", [])))


# 1 ------------------------------------------------------------------
def deck_dna(store, industry: str | None = None):
    orders = store.all_orders()
    if industry:
        filtered = [json.loads(d["slide_order"]) for d in store.docs() if d["industry"] == industry]
        orders = filtered if len(filtered) >= 3 else orders
    ranks = {k: [] for k in SLIDE_KEYS}
    for order in orders:
        seen = list(dict.fromkeys(order))
        for i, k in enumerate(seen):
            ranks[k].append(i / max(len(seen) - 1, 1))
    scored = {k: statistics.mean(v) if v else SLIDE_KEYS.index(k) / 9 for k, v in ranks.items()}
    return {"recommended_order": sorted(SLIDE_KEYS, key=scored.get),
            "coverage": {k: round(len(v) / max(len(orders), 1), 2) for k, v in ranks.items()},
            "based_on_decks": len(orders)}


# 2 ------------------------------------------------------------------
def market_math(slides):
    s = next(x for x in slides if x["key"] == "market")
    found = {}
    for line in [s["headline"], *s["bullets"]]:
        for tag in ("TAM", "SAM", "SOM"):
            if tag in line.upper() and money(line):
                found[tag] = money(line)[0]
    issues = []
    if {"TAM", "SAM"} <= found.keys() and found["SAM"] > found["TAM"]:
        issues.append("SAM is larger than TAM — impossible funnel.")
    if {"SAM", "SOM"} <= found.keys():
        share = found["SOM"] / found["SAM"] if found["SAM"] else 0
        if share > 0.2:
            issues.append(f"SOM is {share:.0%} of SAM — investors expect 1-10% in 3-5 years.")
        if found["SOM"] > found["SAM"]:
            issues.append("SOM exceeds SAM.")
    if len(found) < 3:
        issues.append("Give explicit $ values for all of TAM, SAM and SOM (bottom-up: customers x price).")
    return {"values": found, "issues": issues}


# 3 ------------------------------------------------------------------
def consistency(slides):
    by = {s["key"]: _slide_text(s) for s in slides}
    issues = []
    ask_amt = money(by["ask"])
    burn = re.search(r"burn[^$]*\$\s?(\d+(?:\.\d+)?)\s*(k|m)?", by["financials"] + by["ask"], re.I)
    runway = re.search(r"(\d+)\s*(?:-\s*\d+\s*)?months?", by["ask"], re.I)
    if ask_amt and burn and runway:
        b = float(burn.group(1)) * _MULT.get((burn.group(2) or "").lower(), 1)
        implied = ask_amt[0] / b if b else 0
        stated = int(runway.group(1))
        if implied and abs(implied - stated) / stated > 0.35:
            issues.append(f"Ask ${ask_amt[0]:,.0f} / burn ${b:,.0f}/mo = {implied:.0f} months, but slide claims {stated}.")
    pcts = [int(p) for p in re.findall(r"(\d{1,3})%\s*(?:to\s+)?\w+", by["ask"])]
    if len(pcts) >= 2 and sum(pcts) != 100:
        issues.append(f"Use-of-funds percentages sum to {sum(pcts)}%, not 100%.")
    if re.search(r"\b(?:0|no|zero)\s+(?:revenue|customers|paying)|pre-revenue", by["traction"], re.I):
        if any(v > 1e6 for v in money(by["financials"])):
            issues.append("Pre-revenue traction but year-1 financials exceed $1M: show the bridge.")
    return issues


# 4 + 5 --------------------------------------------------------------
_GAP = re.compile(r"\[(?:ASSUMPTION|X|Y|N|\s*)\]|\.\.\.|\$\[", re.I)


def readiness(slides):
    per, total = {}, 0
    for s in slides:
        txt = _slide_text(s)
        gaps = len(_GAP.findall(txt))
        nums = len(re.findall(r"\d", txt))
        words = len(txt.split())
        med = (s.get("benchmark") or {}).get("median_words")
        length_ok = 1.0 if not med else max(0.0, 1 - abs(words - med) / max(med, 1))
        score = round(100 * (0.45 * min(nums / 6, 1) + 0.35 * max(0, 1 - gaps / 4) + 0.20 * length_ok))
        per[s["key"]] = {"score": score, "evidence_gaps": gaps, "words": words, "ref_median_words": med}
        total += score
    weakest = sorted(per, key=lambda k: per[k]["score"])[:3]
    return {"overall": round(total / len(slides)), "per_slide": per, "weakest": weakest}


# 6 ------------------------------------------------------------------
_STOCK_QS = {
    "problem": "How do you know this is a top-3 priority for the buyer and not a nice-to-have?",
    "solution": "What stops an incumbent from shipping this as a feature in one quarter?",
    "market": "Walk me through the bottom-up math — how many customers, at what price?",
    "business_model": "What is your CAC today, and what is the payback period?",
    "competition": "Who would you be most scared of if they raised $50M tomorrow?",
    "gtm": "Which single channel has produced your last 10 customers, and does it scale?",
    "team": "Why are you the team that wins this, and what key role is missing?",
    "financials": "Which one assumption, if wrong by 50%, breaks this model?",
    "traction": "What is your month-3 cohort retention?",
    "ask": "What exact milestone does this round get you to, and is that Series-A-able?",
}


def red_team(deck, weakest):
    prompt = ("Act as a skeptical seed investor. For the deck below return JSON "
              '{"questions":[{"slide":key,"question":str,"why_it_matters":str}], "kill_risk": str} '
              f"with 2 brutal questions for each of these weakest slides {weakest}.\nDECK: "
              + str([{k: s[k] for k in ('key', 'headline', 'bullets')} for s in deck["slides"]]))
    out = llm.generate_json(prompt, deep=True)
    if out:
        return out
    return {"questions": [{"slide": k, "question": _STOCK_QS[k],
                           "why_it_matters": f"Weakest-scoring slide: {SLIDE_TITLES[k]}"} for k in weakest],
            "kill_risk": "Unverified numbers — every [ASSUMPTION] is a diligence question you will be asked."}


def analyze(store, deck):
    r = readiness(deck["slides"])
    return {"readiness": r, "market_math": market_math(deck["slides"]),
            "consistency": consistency(deck["slides"]),
            "deck_dna": deck_dna(store, deck["input"].get("industry")),
            "red_team": red_team(deck, r["weakest"])}

