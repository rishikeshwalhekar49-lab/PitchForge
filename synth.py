"""Seeded synthetic data generators for the evaluation suite.

Everything here is synthetic on purpose: ground truth is known exactly. Real-deck evaluation
(eval/real_eval.py) uses hand labels instead. Report both in a paper and state which is which.
"""
import random

from app.schema import SLIDE_KEYS

# Paraphrase pools deliberately include words that are NOT in the classifier's keyword list,
# plus cross-slide distractors, so the task is not trivially solvable.
PHRASES = {
    "problem": ["{aud} waste hours every week on manual work", "the status quo is slow and error-prone",
                "{aud} are frustrated with spreadsheets", "a costly pain point nobody has fixed",
                "today this process is broken", "teams lose revenue to churn and delays"],
    "solution": ["our platform automates the whole workflow", "we built a product that does it in minutes",
                 "introducing an AI assistant for {aud}", "how it works: connect, analyze, act",
                 "one dashboard replaces five tools", "customers get results on day one"],
    "market": ["TAM of $40B across {ind}", "a $6B serviceable addressable market",
               "market size is growing 20% a year", "we target a SOM of $300M",
               "billion-dollar opportunity in {ind}", "millions of potential buyers"],
    "business_model": ["subscription pricing at $49 per seat", "SaaS with 80% gross margin",
                       "a 3% take rate on each transaction", "unit economics: LTV to CAC of 4x",
                       "ARPU grows with usage tiers", "annual contracts paid upfront"],
    "competition": ["competitors are legacy vendors", "our moat is proprietary data",
                    "the competitive landscape is fragmented", "we differentiate on speed and price",
                    "alternatives include agencies and doing nothing", "incumbents cannot move this fast"],
    "gtm": ["go-to-market starts with founder-led sales", "distribution through partner channels",
            "customer acquisition via content and community", "a land-and-expand sales motion",
            "marketing to {aud} on LinkedIn", "channel partnerships with resellers"],
    "team": ["founder and CEO previously built a unicorn", "CTO with 10 years experience at Google",
             "advisors from top {ind} firms", "the team shipped products used by millions",
             "ex-McKinsey operator runs growth", "we are hiring a head of sales"],
    "financials": ["revenue projection reaches $12M by 2028", "forecast shows EBITDA positive in year 3",
                   "monthly burn of $120k", "the P&L model assumes 5% churn",
                   "financial plan for the next five years", "gross revenue triples each year"],
    "traction": ["ARR of $1.2M growing 15% MoM", "300 paying customers and rising",
                 "net retention above 120%", "pilots with 3 Fortune 500 companies",
                 "10,000 users joined the waitlist", "hit every milestone since launch"],
    "ask": ["we are raising a $3M seed round", "funding to reach Series A",
            "use of funds: product, hiring, marketing", "18 months of runway",
            "seeking investment from strategic angels", "the ask: $3M on a SAFE"],
}
AUD = ["freelancers", "clinics", "teachers", "SMBs", "creators", "retailers"]
IND = ["fintech", "healthtech", "edtech", "saas", "marketplace", "climate", "consumer"]


def page(rng, key, with_title: bool, noise: float):
    aud, ind = rng.choice(AUD), rng.choice(IND)
    lines = [p.format(aud=aud, ind=ind) for p in rng.sample(PHRASES[key], 3)]
    if rng.random() < noise:  # distractor sentence from another slide type
        other = rng.choice([k for k in SLIDE_KEYS if k != key])
        lines.append(rng.choice(PHRASES[other]).format(aud=aud, ind=ind))
    rng.shuffle(lines)
    title = key.replace("_", " ").replace("gtm", "go-to-market").title() + ". " if with_title else ""
    return title + " ".join(lines)


def classifier_corpus(n_decks=200, seed=7, title_rate=0.6, noise=0.5, shuffle_rate=0.3):
    """Returns [(text, position, label, deck_id)]."""
    rng = random.Random(seed)
    rows = []
    for d in range(n_decks):
        order = list(SLIDE_KEYS)
        if rng.random() < shuffle_rate:  # some decks reorder slides — tests over-reliance on position
            i, j = rng.sample(range(10), 2)
            order[i], order[j] = order[j], order[i]
        for pos, key in enumerate(order):
            rows.append((page(rng, key, rng.random() < title_rate, noise), pos / 9, key, d))
    return rows


# ---------------- contradiction benchmark ----------------
CONTRA_TYPES = ["runway_mismatch", "funds_not_100", "sam_gt_tam", "som_share", "prerev_vs_financials"]


def _fmt(v, rng):
    """Vary money formatting like real decks do."""
    style = rng.choice(["M", "comma", "k"])
    if style == "M" and v >= 1e6:
        return f"${v / 1e6:.1f}M".replace(".0M", "M")
    if style == "k" and v < 1e6:
        return f"${v / 1e3:.0f}k"
    return f"${v:,.0f}"


def contradiction_deck(rng, kind: str | None):
    ask = rng.choice([5e5, 1e6, 1.5e6, 2e6, 3e6, 5e6])
    burn = rng.choice([4e4, 6e4, 8e4, 1e5, 1.5e5, 2e5])
    runway = round(ask / burn)
    split = rng.choice([[50, 30, 20], [40, 40, 20], [60, 25, 15], [45, 35, 20]])
    tam = rng.choice([2e10, 4e10, 8e10]); sam = tam * rng.uniform(0.05, 0.3); som = sam * rng.uniform(0.01, 0.1)
    prerev = rng.random() < 0.5
    y1 = rng.choice([2e5, 4e5, 8e5]) if prerev else rng.choice([1.5e6, 3e6])

    if kind == "runway_mismatch":
        runway = max(1, round(runway * rng.choice([rng.uniform(0.3, 0.55), rng.uniform(1.7, 3.0)])))
    elif kind == "funds_not_100":
        split = split.copy(); split[0] += rng.choice([-25, -15, -10, 10, 15, 25])
    elif kind == "sam_gt_tam":
        tam, sam = sam, tam; som = tam * 0.02
    elif kind == "som_share":
        som = sam * rng.uniform(0.3, 0.8)
    elif kind == "prerev_vs_financials":
        prerev, y1 = True, rng.choice([2e6, 4e6, 6e6])

    f = lambda v: _fmt(v, rng)  # noqa: E731
    traction = (["Pre-revenue", "3 design-partner pilots"] if prerev
                else [f"{f(y1 / 12)} MRR", "120 paying customers"])
    # year-3 figure is always large: a realistic trap for naive "any value > $1M" checks
    fin = [f"Year 1 revenue: {f(y1)}", f"Year 3 revenue: {f(y1 * 12)}", f"Monthly burn: {f(burn)}"]
    slides = {k: {"key": k, "headline": "", "bullets": []} for k in SLIDE_KEYS}
    slides["market"]["bullets"] = [f"TAM: {f(tam)}", f"SAM: {f(sam)}", f"SOM: {f(som)}"]
    slides["financials"]["bullets"] = fin
    slides["traction"]["bullets"] = traction
    slides["ask"]["bullets"] = [f"Raising {f(ask)} on a SAFE", f"Runway: {runway} months",
                                f"Use of funds: {split[0]}% product, {split[1]}% GTM, {split[2]}% operations"]
    return list(slides.values())


# ---------------- deck DNA ----------------
INDUSTRY_ORDERS = {  # ground-truth "house styles" used to generate decks
    "marketplace": ["problem", "solution", "traction", "market", "business_model", "gtm",
                    "competition", "team", "financials", "ask"],
    "saas": ["problem", "solution", "market", "business_model", "traction", "gtm",
             "competition", "financials", "team", "ask"],
    "climate": ["problem", "market", "solution", "team", "business_model", "competition",
                "gtm", "traction", "financials", "ask"],
}


def noisy_order(rng, base, swaps=2):
    o = list(base)
    for _ in range(swaps):
        i = rng.randrange(9); o[i], o[i + 1] = o[i + 1], o[i]
    return o
