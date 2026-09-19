"""LLM layer. Tiered like a real review pipeline: a fast model drafts, a stronger model red-teams.
Falls back to a deterministic template engine when no API key is set, so the app always works offline."""
import json
import os
import re

FAST_MODEL = os.environ.get("PITCHFORGE_FAST_MODEL", "gemini-2.5-flash")
DEEP_MODEL = os.environ.get("PITCHFORGE_DEEP_MODEL", "gemini-2.5-pro")

_client = None


def client():
    global _client
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=key)
        except Exception:
            return None
    return _client


def generate_json(prompt: str, deep: bool = False) -> dict | None:
    c = client()
    if c is None:
        return None
    try:
        resp = c.models.generate_content(
            model=DEEP_MODEL if deep else FAST_MODEL, contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.6})
        text = resp.text.strip()
        text = re.sub(r"^```(json)?|```$", "", text).strip()
        return json.loads(text)
    except Exception as e:
        print(f"[llm] falling back to templates: {e}")
        return None
