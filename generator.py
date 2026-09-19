"""Turns a raw idea into a grounded, editable 10-slide framework."""
import json
import statistics
import uuid

from . import llm
from .schema import SLIDE_KEYS, SLIDE_TITLES

GUIDANCE = {
    "problem": "One painful, specific, frequent problem. Quantify the cost. Name who feels it.",
    "solution": "Show the product, the 'aha' moment, and why it is 10x better — not 10% better.",
    "market": "Bottom-up TAM -> SAM -> SOM. Show the math (units x price), not just a top-down report number.",
    "business_model": "Who pays, how much, how often. Include target unit economics (CAC, LTV, gross margin).",
    "competition": "2x2 matrix or feature table. Be honest about alternatives, including 'do nothing'.",
    "gtm": "First 100 customers: channel, cost per acquisition, sales cycle. Then how it scales.",
    "team": "Why THIS team wins: founder-market fit, prior exits, key hires still needed.",
    "financials": "3-5 year revenue, burn and headcount. State the 3 assumptions that drive the model.",
    "traction": "The steepest honest curve you have: revenue, users, retention, LOIs, pilots.",
    "ask": "Amount, instrument, runway in months, use of funds split, and the milestone it unlocks.",
}


def _benchmarks(store):
    """Per-slide-type stats learned from the reference corpus."""
    out = {}
    for key in SLIDE_KEYS:
        pages = store.pages_of_type(key)
        words = [p["words"] for p in pages if p["words"]]
        out[key] = {"examples": len(pages),
                    "median_words": int(statistics.median(words)) if words else None,
                    "decks_with_slide": len({p["doc_sha"] for p in pages})}
    out["_total_decks"] = len(store.docs())
    return out


def _template_slide(key, inp, refs):
    idea, aud, ind = inp["idea"], inp["audience"], inp["industry"]
    name = inp.get("company") or "Our startup"
    t = {
        "problem": (f"{aud} are losing time and money every week",
                    [f"Today, {aud} rely on manual, fragmented workarounds",
                     "[ASSUMPTION] Quantify: hours lost / $ wasted per customer per month",
                     f"Existing {ind} tools were not built for this segment",
                     "Customer quote from discovery interviews goes here"]),
        "solution": (f"{name}: {idea}",
                     ["Core workflow in 3 steps (show screenshot / demo GIF)",
                      f"Built specifically for {aud}",
                      "Key insight competitors missed: ...",
                      "Result: [X]x faster / [Y]% cheaper than status quo"]),
        "market": (f"A multi-billion dollar {ind} opportunity",
                   ["TAM: total # of potential customers x annual price",
                    f"SAM: {aud} reachable in launch geographies",
                    "SOM: realistic 3-year capture (typically 1-5% of SAM)",
                    "Market tailwind: why now?"]),
        "business_model": ("Simple, recurring revenue",
                           ["Pricing: $[X]/user/month (or % take rate)",
                            "Target gross margin: 70%+",
                            "LTV:CAC target > 3:1, CAC payback < 12 months",
                            "Expansion revenue via upsell / seats"]),
        "competition": ("Why we win",
                        ["Direct competitors: ...", "Indirect / status quo: spreadsheets, agencies, do nothing",
                         "Our moat: data network effect / workflow lock-in / distribution",
                         "2x2: [axis A] vs [axis B] - we own the top-right"]),
        "gtm": ("Path to the first 1,000 customers",
                [f"Beachhead: a narrow slice of {aud}", "Channel 1: founder-led sales / community",
                 "Channel 2: partnerships & integrations", "Target CAC $[X], payback [Y] months"]),
        "team": ("Founder-market fit",
                 ["CEO - domain expertise in " + ind, "CTO - built and scaled similar systems",
                  "Advisors: industry operator + investor", "Next key hires: ..."]),
        "financials": ("Path to $10M ARR",
                       ["Year 1 / 2 / 3 revenue: $[ ] / $[ ] / $[ ]", "Monthly burn: $[ ]",
                        "Key drivers: customers added/mo, ARPU, churn", "Break-even in month [ ]"]),
        "traction": ("Early proof it works",
                     ["[N] paying customers / pilots / LOIs", "MoM growth: [X]%", "Retention / NPS: ...",
                      "Notable logos or partnerships"]),
        "ask": ("Raising $[X] to reach [milestone]",
                ["Instrument: SAFE / priced seed", "Use of funds: 50% product, 30% GTM, 20% ops",
                 "Runway: 18-24 months", "Milestone unlocked: Series A metrics ($1M+ ARR)"]),
    }[key]
    return t


def _llm_prompt(inp, grounding, bench):
    return f"""You are a partner at a top seed VC who has seen 10,000 pitch decks.
Write a fundable 10-slide pitch framework as JSON: {{"slides": [{{"key", "headline", "bullets": [4 strings], "speaker_notes"}}]}}
with keys in this order: {SLIDE_KEYS}.
Rules: headlines are claims not labels; use concrete numbers; mark unverifiable numbers with "[ASSUMPTION]";
keep each slide near the reference median word count; do NOT copy reference text, learn its structure.

STARTUP: {json.dumps(inp)}
GUIDANCE PER SLIDE: {json.dumps(GUIDANCE)}
CORPUS BENCHMARKS: {json.dumps(bench)}
REFERENCE EXCERPTS (from successful decks, for structure only):
{grounding}"""


def generate(store, inp: dict) -> dict:
    bench = _benchmarks(store)
    query = f"{inp['idea']} {inp['audience']} {inp['industry']}"
    refs = {k: store.search(f"{query} {SLIDE_TITLES[k]}", slide_type=k, k=2, industry=inp["industry"])
            for k in SLIDE_KEYS}
    grounding = "\n".join(f"[{k}] {r['filename']} p{r['page_no']}: {r['text'][:350]}"
                          for k, rs in refs.items() for r in rs)

    drafted = llm.generate_json(_llm_prompt(inp, grounding, bench))
    by_key = {s.get("key"): s for s in (drafted or {}).get("slides", [])}

    slides = []
    for i, key in enumerate(SLIDE_KEYS, 1):
        s = by_key.get(key)
        if s:
            headline, bullets, notes = s.get("headline", ""), s.get("bullets", []), s.get("speaker_notes", "")
        else:
            headline, bullets = _template_slide(key, inp, refs[key])
            notes = GUIDANCE[key]
        slides.append({
            "n": i, "key": key, "title": SLIDE_TITLES[key], "headline": headline,
            "bullets": bullets, "speaker_notes": notes, "guidance": GUIDANCE[key],
            "benchmark": bench.get(key),
            "references": [{"file": r["filename"], "page": r["page_no"], "score": r["score"],
                            "snippet": r["text"][:220]} for r in refs[key]],
        })
    return {"id": uuid.uuid4().hex[:10], "input": inp, "slides": slides,
            "engine": "gemini" if drafted else "template", "corpus_decks": bench["_total_decks"]}
