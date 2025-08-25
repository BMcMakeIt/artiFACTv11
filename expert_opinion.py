from __future__ import annotations

import re

# Persona vibe briefs capture stylistic guidance for each category.
PERSONA_VIBE_BRIEFS = {
    "minerals": (
        "Voice: cheerful science explainer for all ages; clear, punchy, playful curiosity; simple analogies; "
        "gentle humor; one exclamation max; avoid jargon and numeric stats."
    ),
    "shells": (
        "Voice: ocean explorer-poet; adventurous and reverent; imagery of tides, currents, and light; "
        "romantic but restrained; no exclamation marks; no conservation sermonizing."
    ),
    "fossils": (
        "Voice: gentle natural-history narrator; awe at deep time; calm, observational cadence; "
        "evokes past worlds without melodrama; avoid numbers, measurements, and dates."
    ),
    "vinyl": (
        "Voice: warm late-20th-century radio DJ; genial countdown vibe; hooky phrases and smooth patter; "
        "nostalgic nods to chart moments; keep it friendly, not snarky; no brand-dropping."
    ),
    "zoological": (
        "Voice: 19th-century naturalist; precise, empirical observations with modest wonder; "
        "notes variation and selection without lecturing; measured tone; no measurements or stats."
    ),
    "other": (
        "Voice: boundlessly enthusiastic, childlike wonder; candy-and-holiday metaphors; clean humor; "
        "earnest and upbeat; keep it to one tidy sentence to avoid rambling."
    ),
}

# Optional: for UI labels only. Never pass these into model prompts.
PERSONA_UI_LABEL = {
    "minerals": "Bill Nye vibe",
    "shells": "Jacques Cousteau vibe",
    "fossils": "David Attenborough vibe",
    "vinyl": "Casey Kasem vibe",
    "zoological": "Charles Darwin vibe",
    "other": "Buddy the Elf vibe",
}

# Bump when persona briefs change to avoid cache collisions.
PERSONA_VIBES_VERSION = "v1.0"


# --- Prompt builder -------------------------------------------------------

def build_expert_opinion_prompt(category: str, item_name: str) -> str:
    cat = (category or "").strip().lower()
    vibe = PERSONA_VIBE_BRIEFS.get(cat, PERSONA_VIBE_BRIEFS["other"])
    rules = (
        "Produce exactly ONE sentence (14–28 words). "
        "Use only the item name and category as context. "
        "Do NOT mention size, weight, measurements, hardness, composition, locality, formation, dates, or toxicity. "
        "Avoid listy phrasing and generic openings like 'This specimen is'. "
        "No emojis, no hashtags, no quotes."
    )
    return (
        f"You are writing a short 'expert opinion' blurb.\n"
        f"Category: {cat}\n"
        f"Item name: {item_name}\n\n"
        f"{rules}\n\n"
        f"Adopt this stylistic vibe:\n{vibe}\n\n"
        f"Now write the single-sentence blurb."
    )


# --- Post-processing ------------------------------------------------------

def postprocess_opinion(text: str) -> str:
    if not text:
        return ""
    s = " ".join(text.strip().split())
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    s = m.group(1) if m else s
    if len(s.split()) > 30:
        s = " ".join(s.split()[:30]).rstrip(",;:") + "."
    s = s.lstrip("“”\"'—–- ").strip()
    return s


# --- Opinion generation ---------------------------------------------------

# cache_get/cache_set and openai_complete are expected to exist in the
# surrounding application. We import them lazily to keep this module light.
try:  # pragma: no cover - simple import guard
    from cache import cache_get, cache_set  # type: ignore
except Exception:  # pragma: no cover
    def cache_get(key: str):
        return None
    def cache_set(key: str, value: str, ttl_days: int = 0):
        return None

try:  # pragma: no cover - simple import guard
    from openai_helpers import openai_complete  # type: ignore
except Exception:  # pragma: no cover
    def openai_complete(*args, **kwargs):
        return ""


def generate_expert_opinion(item: dict, enrichment: dict | None = None) -> str:
    category = (item.get("category") or "").strip().lower()
    item_name = (item.get("name") or item.get("display_name") or "").strip()
    if not item_name:
        return ""
    cache_key = f"opinion:{category}:{item_name}:{PERSONA_VIBES_VERSION}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    prompt = build_expert_opinion_prompt(category, item_name)
    try:
        text = openai_complete(prompt=prompt, temperature=0.7, max_tokens=60)
        opinion = postprocess_opinion(text)
    except Exception:
        opinion = ""

    if opinion:
        cache_set(cache_key, opinion, ttl_days=90)
    return opinion


# --- UI helper ------------------------------------------------------------

def get_persona_ui_label(category: str) -> str:
    cat = (category or "").strip().lower()
    return PERSONA_UI_LABEL.get(cat, PERSONA_UI_LABEL["other"])

