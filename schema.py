"""The canonical 10-slide framework and keyword taxonomy used to classify reference pages."""

SLIDES = [
    ("problem", "Problem", ["problem", "pain", "challenge", "struggle", "broken", "today", "frustrat"]),
    ("solution", "Solution", ["solution", "product", "platform", "how it works", "introducing", "we built"]),
    ("market", "Market Size (TAM/SAM/SOM)", ["tam", "sam", "som", "market size", "addressable", "billion", "market opportunity"]),
    ("business_model", "Business Model", ["business model", "revenue model", "pricing", "subscription", "saas", "take rate", "unit economics", "arpu"]),
    ("competition", "Competitive Landscape", ["competition", "competitor", "competitive", "landscape", "alternatives", "vs", "moat", "differentiat"]),
    ("gtm", "Go-To-Market Strategy", ["go-to-market", "go to market", "gtm", "distribution", "acquisition", "channel", "sales strategy", "marketing"]),
    ("team", "Team Composition", ["team", "founder", "ceo", "cto", "advisor", "experience", "ex-", "previously"]),
    ("financials", "Financial Projections", ["financial", "projection", "forecast", "revenue", "ebitda", "burn", "p&l", "2027", "2028"]),
    ("traction", "Traction Metrics", ["traction", "growth", "mrr", "arr", "users", "customers", "retention", "pilot", "milestone"]),
    ("ask", "Funding Ask", ["ask", "raising", "funding", "use of funds", "investment", "seed", "series", "runway"]),
]
SLIDE_KEYS = [s[0] for s in SLIDES]
SLIDE_TITLES = {k: t for k, t, _ in SLIDES}
KEYWORDS = {k: kw for k, _, kw in SLIDES}
