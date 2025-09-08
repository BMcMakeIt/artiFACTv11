import tkinter as tk
import numpy as np  # if not already imported
import random
import requests
import webbrowser
import io
import json
import base64
import threading
import tempfile
import glob
import shutil
import os
import cv2
import sqlite3
import time
from datetime import datetime
from functools import partial
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from tkinter import filedialog, messagebox, ttk
import re
import sys

# Image classifier (uses your existing module when present)
try:
    # provides vc.classify_image(path, category)
    import openai_classifier_v1 as vc
except Exception:
    vc = None  # allow app to run without the module


def run_classifier(image_path: str, category: str) -> dict:
    """
    Return {'openai': {'guesses': [...]}}; guaranteed structure.
    Uses vc.classify_image if available, else a safe empty fallback.
    """
    # real classifier
    if vc and hasattr(vc, "classify_image"):
        try:
            return vc.classify_image(image_path, category)
        except Exception:
            pass
    # fallback (no crash, but also no guesses)
    return {"openai": {"guesses": []}}


def place_window(win, w_ratio=0.5, h_ratio=0.5, anchor="center"):
    """
    Size a Tk/Toplevel to a fraction of the screen and position it.
    anchor: "center" | "left" | "right" | "top" | "bottom"
    """
    win.update_idletasks()  # ensure screen dims are current
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w, h = int(sw * w_ratio), int(sh * h_ratio)

    # default center
    x, y = (sw - w) // 2, (sh - h) // 2
    if anchor == "left":
        x, y = 0, (sh - h) // 2
    if anchor == "right":
        x, y = sw - w, (sh - h) // 2
    if anchor == "top":
        x, y = (sw - w) // 2, 0
    if anchor == "bottom":
        x, y = (sw - w) // 2, sh - h

    win.geometry(f"{w}x{h}+{x}+{y}")


CASE_TERMS_RE = re.compile(
    r"\b("
    r"display\s*case|shadow\s*box|shadowbox|"
    r"case|box|frame|framed|"
    r"mount(?:ed)?|mounted|"
    r"display|presentation|plaque|cabinet"
    r")\b",
    re.I
)


def _decontainerize_label(label: str) -> str:
    """Remove container terms like 'display case' so guesses focus on the specimen."""
    s = (label or "").strip()
    if not s:
        return s
    cleaned = CASE_TERMS_RE.sub("", s)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -–—:_").strip()
    return cleaned or s


ZOO_FALLBACK_MAP = {
    # container-y → organism-centric
    "framed insect specimen": "insect specimen",
    "entomology display": "insect specimen",
    "decorative butterfly display": "butterfly specimen",
    "framed butterfly specimen": "butterfly specimen",
    "framed beetle specimen": "beetle specimen",
    "butterfly display": "butterfly specimen",
    "beetle display": "beetle specimen",
    "mounted insect": "insect specimen",
    "mounted butterfly": "butterfly specimen",
    "mounted beetle": "beetle specimen",
    "shadow box": "insect specimen",
    "display case": "zoological specimen",
    "decorative art piece": "zoological specimen",
}


def normalize_zoo_label(raw_label: str) -> str:
    s = (raw_label or "").lower().strip()
    if not s:
        return raw_label
    # strip container words first, then map
    s = _decontainerize_label(s)
    for k, v in ZOO_FALLBACK_MAP.items():
        if k in s:
            return v
    return s or raw_label


def _decontainerize_guesses(guesses):
    out = []
    for g in (guesses or []):
        g = dict(g) if isinstance(g, dict) else {}
        g["label"] = _decontainerize_label(g.get("label", ""))
        out.append(g)
    return out


# --- Species utilities & zoological hints ---
# Genus species (optional subspecies)
SPECIES_RE = re.compile(r"^[A-Z][a-z]+(?:\s[a-z\-]+){1,2}$")


def _is_species_name(s: str) -> bool:
    return bool(SPECIES_RE.match((s or "").strip()))


ZOO_GROUP_HINTS = {
    "butterfly": "insect", "moth": "insect", "beetle": "insect", "insect": "insect",
    "dragonfly": "insect", "mantis": "insect", "wasp": "insect", "bee": "insect",
    "spider": "arachnid", "scorpion": "arachnid",
    "tooth": "tooth", "fang": "tooth", "tusk": "tooth",
    "antler": "antler", "horn": "antler",
    "bone": "bone", "skull": "bone", "vertebra": "bone", "rib": "bone", "femur": "bone", "metacarpal": "bone", "phalanx": "bone",
    "feather": "feather", "skin": "skin", "hide": "skin", "taxidermy": "skin"
}


def _infer_zoo_group(label: str) -> str:
    s = (label or "").lower()
    for k, v in ZOO_GROUP_HINTS.items():
        if k in s:
            return v
    return "insect"  # safe default


def _autocrop_inside_case(src_path: str) -> str:
    """
    Try to crop the image to the interior of a display case/shadow box
    so the model focuses on the specimen. Returns a temp file path, or
    the original path if no reliable crop is found.
    """
    try:
        img = cv2.imread(src_path)
        if img is None:
            return src_path
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 30, 30)
        edges = cv2.Canny(gray, 70, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, ww, hh = cv2.boundingRect(approx)
                area = ww * hh
                if area > best_area and area > 0.25 * w * h:
                    best = (x, y, ww, hh)
                    best_area = area
        if best:
            x, y, ww, hh = best
            inset = int(min(ww, hh) * 0.06)
            x2, y2 = max(0, x + inset), max(0, y + inset)
            x3, y3 = min(w, x + ww - inset), min(h, y + hh - inset)
            if (x3 - x2) > 80 and (y3 - y2) > 80:
                crop = img[y2:y3, x2:x3].copy()
                crop = cv2.convertScaleAbs(crop, alpha=1.08, beta=6)
                out = os.path.join(tempfile.gettempdir(),
                                   "artifacts_autocrop.jpg")
                cv2.imwrite(out, crop)
                return out
        return src_path
    except Exception:
        return src_path


# --- Foreground isolation helper ---


def isolate_specimen_grabcut(src_path: str) -> str:
    """
    Use GrabCut to isolate foreground specimen and return a temp PNG with alpha.
    Falls back to original on error.
    """
    try:
        img = cv2.imread(src_path)
        if img is None:
            return src_path
        h, w = img.shape[:2]
        rect = (int(0.10 * w), int(0.10 * h), int(0.80 * w), int(0.80 * h))
        mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        fg = img * mask2[:, :, None]
        ys, xs = np.where(mask2 > 0)
        if len(xs) < 100 or len(ys) < 100:
            return src_path
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = fg[y1: y2 + 1, x1: x2 + 1]
        alpha = (mask2[y1: y2 + 1, x1: x2 + 1] * 255).astype(np.uint8)
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        out = os.path.join(tempfile.gettempdir(), "artifacts_fg.png")
        cv2.imwrite(out, rgba)
        return out
    except Exception:
        return src_path


# --- Discogs API helpers & pricing ---
def _discogs_user_agent():
    return "artiFACTS/1.0 (+local)"


def _discogs_token():
    return os.getenv("DISCOGS_TOKEN", "").strip()


def discogs_search_release(artist: str = "", title: str = "", barcode: str = "", catno: str = "", per_page: int = 10):
    tok = _discogs_token()
    headers = {"User-Agent": _discogs_user_agent()}
    params = {"type": "release", "per_page": per_page}
    if tok:
        params["token"] = tok
    q_parts = []
    if artist:
        q_parts.append(artist)
    if title:
        q_parts.append(title)
    if q_parts:
        params["q"] = " ".join(q_parts)
    if barcode:
        params["barcode"] = barcode
    if catno:
        params["catno"] = catno
    try:
        r = requests.get("https://api.discogs.com/database/search",
                         headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def discogs_get_release(release_id: int):
    tok = _discogs_token()
    headers = {"User-Agent": _discogs_user_agent()}
    params = {"token": tok} if tok else {}
    try:
        r = requests.get(
            f"https://api.discogs.com/releases/{release_id}", headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def discogs_price_suggestions(release_id: int):
    tok = _discogs_token()
    headers = {"User-Agent": _discogs_user_agent()}
    params = {"token": tok} if tok else {}
    try:
        r = requests.get(
            f"https://api.discogs.com/marketplace/price_suggestions/{release_id}", headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}


def discogs_release_stats(release_id: int, currency: str = "USD"):
    tok = _discogs_token()
    headers = {"User-Agent": _discogs_user_agent()}
    params = {"token": tok, "curr_abbr": currency} if tok else {
        "curr_abbr": currency}
    try:
        r = requests.get(
            f"https://api.discogs.com/marketplace/stats/{release_id}", headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}


def _pick_best_discogs_match(results, artist="", title=""):
    if not results:
        return None
    artist_l = (artist or "").strip().lower()
    title_l = (title or "").strip().lower()

    def score(it):
        s = 0
        t = (it.get("title") or "").lower()
        if artist_l and artist_l in t:
            s += 2
        if title_l and title_l in t:
            s += 2
        if it.get("type") == "release":
            s += 1
        return s
    return max(results, key=score)


def _discogs_format_fields(release: dict):
    fmt_list = release.get("formats") or []
    format_tokens, size, rpm, weight_grams, color = [], "", "", "", ""
    for f in fmt_list:
        name = f.get("name")
        descs = f.get("descriptions") or []
        if name:
            format_tokens.append(name)
        for d in descs:
            format_tokens.append(d)
            dl = str(d).lower()
            if "rpm" in dl and not rpm:
                rpm = d
            if "gram" in dl and not weight_grams:
                weight_grams = d
            if any(c in dl for c in ["black", "color", "coloured", "colored", "marbled", "splatter", "pink", "blue", "red", "green", "white", "clear", "silver", "gold", "orange", "purple"]):
                if not color:
                    color = d
            if '"' in d and not size:
                size = d
    format_joined = ", ".join(dict.fromkeys([t for t in format_tokens if t]))
    return format_joined, size, rpm, weight_grams, color


def _parse_duration_to_seconds(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    parts = [p for p in s.split(":") if p.isdigit()]
    if not parts:
        return 0
    parts = list(map(int, parts))
    if len(parts) == 3:
        h, m, sec = parts
        return h*3600 + m*60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m*60 + sec
    if len(parts) == 1:
        return parts[0]
    return 0


def _fmt_seconds(secs: int) -> str:
    if secs <= 0:
        return ""
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def map_discogs_release_to_schema(release: dict) -> dict:
    if not release:
        return {}
    labels = release.get("labels") or []
    label_names = [l.get("name") for l in labels if l.get("name")]
    label = label_names[0] if label_names else ""
    tracks = release.get("tracklist") or []
    tot_secs, parts = 0, []
    for i, tr in enumerate(tracks):
        title = (tr.get("title") or "").strip()
        pos = (tr.get("position") or "").strip() or str(i+1)
        dur = (tr.get("duration") or "").strip()
        sec = _parse_duration_to_seconds(dur)
        tot_secs += sec
        if title:
            seg = f"{pos} {title}"
            if dur:
                seg += f" — {dur}"
            parts.append(seg)
    tracklist = "; ".join(parts)
    genres = "; ".join(release.get("genres") or [])
    styles = "; ".join(release.get("styles") or [])
    fmt, size, rpm, weight_grams, color = _discogs_format_fields(release)
    contributors = []
    for a in (release.get("extraartists") or []):
        nm = a.get("name")
        role = a.get("role")
        if nm and role:
            contributors.append(f"{nm} — {role}")
    out = {
        "artist":       "; ".join(a.get("name") for a in (release.get("artists") or []) if a.get("name")) or "",
        "release_title": release.get("title", "") or "",
        "label":         label,
        "labels_all":    "; ".join(dict.fromkeys(label_names)) if label_names else "",
        "country":       release.get("country", "") or "",
        "release_year":  str(release.get("year") or ""),
        "genres":        genres,
        "styles":        styles,
        "tracklist":     tracklist,
        "total_runtime": _fmt_seconds(tot_secs),
        "format":        fmt,
        "size":          size,
        "rpm":           rpm,
        "weight_grams":  weight_grams,
        "color":         color,
        "pressing_plant": "",
        "edition_notes":  (release.get("notes") or "").strip(),
        "contributors_primary": "; ".join(contributors[:10]),
        "discogs_release_id": str(release.get("id") or ""),
        "discogs_url": release.get("uri") or "",
        "discogs_median_price_usd": "",
        "discogs_low_high_usd": "",
        "description": "",
        "fact": "",
    }
    for c in (release.get("companies") or []):
        role = (c.get("entity_type_name") or c.get("role") or "").lower()
        if "pressed" in role and c.get("name"):
            out["pressing_plant"] = c["name"]
            break
    return out


# --- Fossil enrichment helpers ---

# small ontology to infer broader fossil types from classifier labels
FOSSIL_GENERIC_CONCEPTS = {
    "sand dollar": "echinoid_fossil",
    "sea biscuit": "echinoid_fossil",
    "echinoid": "echinoid_fossil",
    "ammonite": "ammonite_fossil",
    "trilobite": "trilobite_fossil",
    "shark tooth": "shark_tooth_fossil",
    "tooth": "vertebrate_tooth_fossil",
    "petrified wood": "petrified_wood",
    "wood": "petrified_wood",
    "belemnite": "belemnite_fossil",
    "crinoid": "crinoid_fossil",
    "brachiopod": "brachiopod_fossil",
    "bivalve": "bivalve_fossil",
    "gastropod": "gastropod_fossil",
    "coral": "coral_fossil",
    "coprolite": "coprolite_fossil",
    "bone": "vertebrate_bone_fossil",
    "vertebra": "vertebrate_bone_fossil",
}

# fields expected from fossil enrichment
FOSSIL_ENRICH_KEYS = [
    "scientific_name", "geological_period", "estimated_age", "fossil_type",
    "preservation_mode", "size_range", "typical_locations",
    "paleoecology", "toxicity_safety", "description", "fact"
]


# Tiny zoological ontology and enrichment keys
ZOO_GENERIC_CONCEPTS = {
    # insects
    "butterfly": "insect_specimen",
    "moth": "insect_specimen",
    "beetle": "insect_specimen",
    "dragonfly": "insect_specimen",
    "mantis": "insect_specimen",
    "bee": "insect_specimen",
    "wasp": "insect_specimen",
    "insect": "insect_specimen",
    # vertebrate parts
    "tooth": "tooth_specimen",
    "fang": "tooth_specimen",
    "tusk": "tooth_specimen",
    "antler": "antler_horn_specimen",
    "horn": "antler_horn_specimen",
    "bone": "bone_specimen",
    "skull": "bone_specimen",
    "vertebra": "bone_specimen",
    # other
    "feather": "feather_specimen",
    "skin": "skin_taxidermy_specimen",
    "hide": "skin_taxidermy_specimen",
    "taxidermy": "skin_taxidermy_specimen",
}

ZOO_ENRICH_KEYS = [
    "scientific_name", "specimen_type", "common_name", "taxonomic_rank",
    "size_range", "identification_tips", "material", "care_preservation",
    "toxicity_safety", "description", "fact"
]

# Default JSON-capable OpenAI model
_OPENAI_MODEL_JSON = "gpt-4o-mini"


def _safe_first_text(resp) -> str:
    """Safely extract the first text response from an OpenAI response object."""
    try:
        txt = getattr(resp, "output_text", "")
        if txt:
            return txt.strip()
    except Exception:
        pass
    try:
        out = getattr(resp, "output", [])
        if out and getattr(out[0], "content", None):
            seg = out[0].content
            if seg and getattr(seg[0], "text", ""):
                return seg[0].text.strip()
    except Exception:
        pass
    return ""


def _parse_json_object(text: str) -> dict:
    """Robustly parse a JSON object from a string, returning an empty dict on failure."""
    try:
        return json.loads(text)
    except Exception:
        try:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(text[s:e+1])
        except Exception:
            pass
    return {}


def _json_chat(prompt: str) -> dict:
    """Send a simple text prompt and parse JSON response."""
    client = _openai_client()
    resp = client.responses.create(
        model=_OPENAI_MODEL_JSON,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt}]}],
        temperature=0.2,
        max_output_tokens=700,
    )
    text = getattr(resp, "output_text", "").strip()
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e+1]) if (s != -1 and e != -1 and e > s) else {}


# --- Insect family pre-pass ---


INSECT_FAMILIES = [
    "Cerambycidae (longhorn beetles)",
    "Scarabaeidae (scarab beetles)",
    "Buprestidae (jewel beetles)",
    "Nymphalidae (brush-footed butterflies)",
    "Papilionidae (swallowtails)",
    "Pieridae (whites and sulphurs)",
    "Sphingidae (hawk moths)",
    "Noctuidae (owlet moths)",
    "Saturniidae (silk moths)"
]


def guess_insect_family(photo_path: str) -> dict:
    """
    Return {'family':'Cerambycidae','confidence':0.0-1.0,'rationale':...}
    """
    da = _img_to_data_url(photo_path) if photo_path and os.path.exists(
        photo_path) else ""
    if not da:
        return {}
    system = "You are a museum entomology curator. Choose the single best insect FAMILY from a fixed list."
    user = (
        "Pick ONE family from this list that best matches the specimen:\n"
        + "; ".join(INSECT_FAMILIES)
        + "\nReturn ONLY JSON: {\"family\":\"\",\"confidence\":0.0,\"rationale\":\"\"}"
    )
    resp = _openai_client().responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": [
                {"type": "input_text", "text": system}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": user},
                {"type": "input_image", "image_url": da},
            ]},
        ],
        temperature=0.0,
        max_output_tokens=300,
    )
    txt = getattr(resp, "output_text", "").strip()
    try:
        data = json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        data = json.loads(txt[s:e+1]) if (s != -
                                          1 and e != -1 and e > s) else {}
    fam = (data.get("family") or "").strip()
    conf = float(data.get("confidence") or 0.0)
    rat = (data.get("rationale") or "").strip()
    if "(" in fam:
        fam = fam.split("(")[0].strip()
    return {"family": fam, "confidence": max(0.0, min(1.0, conf)), "rationale": rat}


def guess_species_from_image(photo_path: str, family_hint: str = "", max_results: int = 4):
    """
    Single-shot species attempt (the model may infer family internally).
    Uses isolated specimen if available, then downsized JPEG for upload.
    """
    # try cheap isolation
    try:
        iso = isolate_specimen_grabcut(photo_path)
    except Exception:
        iso = photo_path

    # ensure small upload and cached data URL
    prep = _prep_for_vision(iso or photo_path, max_dim=1024, quality=85)
    da = _img_to_data_url(prep)
    if not da:
        return []

    fam_clause = f"Family hint: {family_hint}." if family_hint else "Infer the most likely family as part of your reasoning."
    system = (
        "You are a museum taxonomist. Identify the organism to SPECIES (or subspecies) "
        "from a single photo. Output JSON array only, no prose."
    )
    user = (
        f"{fam_clause}\n"
        "Each item must be:\n"
        '{"scientific_name":"","common_name":"","family":"","genus":"","rank":"species","confidence":0.0,"rationale":""}\n'
        f"Limit {max_results}. Confidence 0..1. Only valid binomials (Genus species)."
    )

    resp = _openai_client().responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": [
                {"type": "input_text", "text": system}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": user},
                {"type": "input_image", "image_url": da}
            ]}
        ],
        temperature=0.0,
        max_output_tokens=480  # was 700
    )

    txt = getattr(resp, "output_text", "").strip()
    try:
        arr = json.loads(txt)
    except Exception:
        s, e = txt.find("["), txt.rfind("]")
        arr = json.loads(txt[s:e+1]) if (s != -1 and e != -1 and e > s) else []

    SPECIES_RE = re.compile(r"^[A-Z][a-z]+(?:\s[a-z\-]+){1,2}$")
    out = []
    for it in (arr if isinstance(arr, list) else []):
        sci = (it.get("scientific_name") or "").strip()
        if not SPECIES_RE.match(sci):
            continue
        conf = float(it.get("confidence") or 0)
        out.append({
            "scientific_name": sci,
            "common_name": (it.get("common_name") or "").strip(),
            "family": (it.get("family") or "").strip(),
            "genus": (it.get("genus") or "").strip(),
            "rank": "species",
            "confidence": max(0.0, min(1.0, conf)),
            "rationale": (it.get("rationale") or "").strip(),
        })
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out[:max_results]


PART_ELEMENTS = {"antler", "horn", "tooth", "bone",
                 "skull", "feather", "skin", "taxidermy"}
WHOLE_ELEMENTS = {"insect", "butterfly", "moth",
                  "beetle", "arachnid", "spider", "scorpion"}


REF_KEYWORDS_BY_ELEMENT = {
    # many antler photos show skull caps
    "antler":  ["antler", "skull"],
    "horn":    ["horn", "skull"],
    # 'fossil' boosts museum-style images if applicable
    "tooth":   ["tooth", "fossil"],
    # generic bone -> include skull for context
    "bone":    ["bone", "skull"],
    "skull":   ["skull"],
    "feather": ["feather", "specimen"],
    "skin":    ["skin", "taxidermy"],
    "taxidermy": ["taxidermy", "mount"]
}


def _ref_query_for_zoo(species: str, element: str) -> str:
    """
    Build the best reference-image query for zoological items.
    - Whole organism -> species only
    - Part -> species + element keywords
    """
    sp = (species or "").strip()
    el = (element or "").strip().lower()
    if not sp:
        return el or "zoological specimen"
    if not el or el in WHOLE_ELEMENTS:
        return sp
    kws = REF_KEYWORDS_BY_ELEMENT.get(el, [el])
    return f"{sp} " + " ".join(kws)


def _parse_element_species_label(label: str) -> tuple[str, str]:
    """
    Parse labels like 'Antler — Capreolus capreolus' into (element, species).
    Returns ('', species) if no element prefix is found.
    """
    s = (label or "").strip()
    # Try em dash / hyphen separators
    for sep in ["—", " - ", " — ", "–", "-"]:
        if sep in s:
            left, right = s.split(sep, 1)
            el = left.strip().lower()
            # strip any trailing brackets or quotes from species part
            sp = right.split("[", 1)[0].replace(
                "“", "").replace("”", "").strip()
            return (el, sp)
    # No separator: assume the whole string is a species name
    sp = s.split("[", 1)[0].replace("“", "").replace("”", "").strip()
    return ("", sp)


def guess_specimen_element(photo_path: str) -> dict:
    da = _img_to_data_url(photo_path) if photo_path and os.path.exists(
        photo_path) else ""
    if not da:
        return {"element": "", "confidence": 0.0}
    options = (
        "antler, horn, tooth, bone, skull, feather, skin, taxidermy, "
        "insect, butterfly, moth, beetle, arachnid, spider, scorpion"
    )
    system = "Classify the specimen ELEMENT only (not the case). Reply JSON only."
    user = f"Choose ONE token from this list:\n{options}\nReturn JSON: {{\"element\":\"\",\"confidence\":0.0}}"
    resp = _openai_client().responses.create(
        model=_OPENAI_MODEL_JSON,
        input=[
            {"role": "system", "content": [
                {"type": "input_text", "text": system}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": user},
                {"type": "input_image", "image_url": da}
            ]}
        ],
        temperature=0.0,
        max_output_tokens=140
    )
    txt = getattr(resp, "output_text", "").strip()
    try:
        data = json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        data = json.loads(txt[s:e+1]) if (s != -
                                          1 and e != -1 and e > s) else {}
    el = (data.get("element") or "").strip().lower()
    conf = float(data.get("confidence") or 0.0)
    aliases = {"deer antler": "antler", "horns": "horn",
               "bones": "bone", "hide": "skin"}
    el = aliases.get(el, el)
    return {"element": el, "confidence": max(0.0, min(1.0, conf))}


def _json_chat_vision(prompt: str, photo_path: str) -> dict:
    """
    Send a JSON-only instruction + the actual image to the model.
    Returns a dict; on any error, returns {}.
    """
    try:
        da = _img_to_data_url(photo_path) if photo_path and os.path.exists(
            photo_path) else None
        if not da:
            # fall back to text-only if image is missing
            return _json_chat(prompt) or {}
        msgs = [
            {"role": "system", "content": "You are a careful cataloger. Reply ONLY with a single valid JSON object, no prose."},
            {"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": da}
            ]}
        ]
        resp = _openai_client().responses.create(
            model=_OPENAI_MODEL_JSON,
            input=msgs,
            max_output_tokens=600
        )
        text = _safe_first_text(resp)
        return _parse_json_object(text)
    except Exception:
        return {}


def guess_parent_concept(raw_label: str) -> str:
    s = raw_label.lower()
    for k, v in FOSSIL_GENERIC_CONCEPTS.items():
        if k in s:
            return v
    return "generic_fossil"


def make_prompt(name, label, concept=None, fill_only=None, forbid_specifics=False):
    scope = concept or label
    rules = [
        f'Item name: "{name}"',
        f'Classifier label: "{label}"',
        f'Concept: "{scope}"',
        "Write concise, true, widely applicable statements. 1–2 sentences per field.",
        "Avoid formation names, mines, counties, or species unless present in the label with high confidence.",
        "If you’re unsure, use phrasing like typically, often, commonly.",
    ]
    if forbid_specifics:
        rules.append(
            "Do NOT include exact formations, mines, counties, member names, GPS, or species.")
    fields = ", ".join(fill_only or FOSSIL_ENRICH_KEYS)
    return (
        "Enrich this fossil catalog card.\n\n" +
        "Rules:\n- " + "\n- ".join(rules) +
        "\n\nReturn strict JSON with exactly these keys: " +
        (fields if fill_only else ", ".join(FOSSIL_ENRICH_KEYS)) +
        ".\nIf a field cannot be filled generically, return an empty string for that key."
    )


def count_filled(d: dict) -> int:
    return sum(1 for k in FOSSIL_ENRICH_KEYS if (d.get(k) or "").strip())


def merge_missing(dst: dict, src: dict):
    for k in FOSSIL_ENRICH_KEYS:
        if not (dst.get(k) or "").strip():
            v = (src.get(k) or "").strip()
            if v:
                dst[k] = v
    return dst


def soften_certainty(s: str) -> str:
    return (s.replace(" always ", " often ")
            .replace(" Always ", " Often ")
            .replace(" exactly ", " typically ")
            .replace(" Exactly ", " Typically "))


def postprocess_all(d: dict) -> dict:
    for k, v in list(d.items()):
        if isinstance(v, str) and v:
            d[k] = soften_certainty(v)
    return d


LOCALITY_RE = re.compile(
    r"\b(Formation|Fm\.|Member|Shale|Limestone|Sandstone|Mine|Quarry|Pit|County|Parish|Fm)\b",
    flags=re.I,
)

BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]+)\s([a-z]{3,})\b")

SPEC_ABBR_RE = re.compile(r"\b(sp\.|cf\.|aff\.)\b", flags=re.I)


def remove_specifics(text: str) -> str:
    t = LOCALITY_RE.sub("marine sediments", text)
    t = BINOMIAL_RE.sub("a related taxon", t)
    t = SPEC_ABBR_RE.sub("", t)
    return t


def validate_text(card: dict, label_conf: float) -> dict:
    safe = dict(card)
    for k in FOSSIL_ENRICH_KEYS:
        v = safe.get(k, "")
        if not isinstance(v, str) or not v.strip():
            continue
        if label_conf < 0.7:
            v = remove_specifics(v)
            v = re.sub(r"\b(\d{1,3})\s*(million|mya|Ma)\b",
                       "tens of millions of years", v, flags=re.I)
            v = v.replace(" always ", " often ").replace(" Always ", " Often ")
        safe[k] = v.strip()[:600]
    return safe


def enrich_fossil_two_pass(name: str, raw_label: str, label_conf: float = 1.0, min_fields: int = 7) -> dict:
    p1 = make_prompt(name, raw_label, concept=None,
                     forbid_specifics=(label_conf < 0.7))
    r1 = _json_chat(p1)
    out = {k: r1.get(k, "") for k in FOSSIL_ENRICH_KEYS}

    if count_filled(out) >= min_fields:
        return postprocess_all(validate_text(out, label_conf))

    parent = guess_parent_concept(raw_label)
    missing = [k for k in FOSSIL_ENRICH_KEYS if not (out.get(k) or "").strip()]
    if missing:
        p2 = make_prompt(name, raw_label, concept=parent,
                         fill_only=missing, forbid_specifics=True)
        r2 = _json_chat(p2)
        out = merge_missing(out, r2)

    return postprocess_all(validate_text(out, label_conf))


# Zoological two-pass enrichment helpers

def _zoo_guess_parent(raw_label: str) -> str:
    s = (raw_label or "").lower()
    for k, v in ZOO_GENERIC_CONCEPTS.items():
        if k in s:
            return v
    return "zoological_specimen"


def _zoo_make_prompt(name, label, concept=None, fill_only=None):
    scope = concept or "zoological_specimen"
    fields = ", ".join(fill_only or ZOO_ENRICH_KEYS)
    rules = [
        f'Item name: "{name}"',
        f'Classifier label: "{label}"',
        f'Concept: "{scope}"',
        "Describe the preserved organism (insect, beetle, butterfly, moth, bone, tooth, antler/horn, feather, skin/taxidermy). Ignore frames/cases.",
        "Use safe taxonomy: give Genus species only if confident; otherwise provide order/family.",
        "size_range refers to the specimen, not the container.",
        "identification_tips must reference visible features of the organism (wing pattern, elytra, antennae, tooth shape, etc.).",
        "Return concise, widely applicable statements (1–2 sentences per field).",
        "If uncertain, leave the field empty rather than guessing."
    ]
    return (
        "Enrich this zoological catalog card.\n\nRules:\n- " +
        "\n- ".join(rules) +
        "\n\nReturn strict JSON with exactly these keys: " +
        (fields if fill_only else ", ".join(ZOO_ENRICH_KEYS))
    )


def enrich_zoological_two_pass(name: str, raw_label: str, photo_path: str, label_conf: float = 1.0, min_fields: int = 7) -> dict:
    label_norm = normalize_zoo_label(raw_label)

    # Pass 1: try with the normalized label
    p1 = _zoo_make_prompt(name, label_norm)
    r1 = _json_chat_vision(p1, photo_path) or {}
    out = {k: r1.get(k, "") for k in ZOO_ENRICH_KEYS}

    def _filled(d): return sum(
        1 for k in ZOO_ENRICH_KEYS if (d.get(k) or "").strip())
    if _filled(out) >= min_fields:
        return out

    # Pass 2: parent concept fallback derived from normalized label
    parent = _zoo_guess_parent(label_norm)
    missing = [k for k in ZOO_ENRICH_KEYS if not (out.get(k) or "").strip()]
    if missing:
        p2 = _zoo_make_prompt(
            name, label_norm, concept=parent, fill_only=missing)
        r2 = _json_chat_vision(p2, photo_path) or {}
        for k in missing:
            v = (r2.get(k) or "").strip()
            if v:
                out[k] = v

    for k, v in list(out.items()):
        if isinstance(v, str) and v:
            out[k] = v.replace(" always ", " often ").replace(
                " Always ", " Often ").strip()[:600]
    return out


def merge_gpt_and_discogs(gpt: dict, discogs: dict) -> dict:
    gpt = gpt or {}
    if not discogs:
        return gpt
    merged = dict(gpt)
    prefer = ["artist", "release_title", "label", "labels_all", "country", "release_year",
              "genres", "styles", "tracklist", "total_runtime", "format", "size", "rpm", "weight_grams", "color",
              "pressing_plant", "edition_notes", "contributors_primary",
              "discogs_release_id", "discogs_url"]
    for k in prefer:
        if discogs.get(k):
            merged[k] = discogs[k]
    return merged


def choose_discogs_release(results: list) -> int:
    try:
        import tkinter as _tk
    except Exception:
        best = _pick_best_discogs_match(results) if results else None
        return int(best.get("id")) if best and best.get("id") else 0
    rows = []
    for it in results[:3]:
        rid = it.get("id")
        title = it.get("title", "")
        year = it.get("year", "")
        label = it.get("label", "") or ", ".join(
            it.get("label", []) if isinstance(it.get("label"), list) else [])
        catno = it.get("catno", "")
        country = it.get("country", "")
        fmt = ", ".join(it.get("format", [])) if isinstance(
            it.get("format"), list) else (it.get("format", "") or "")
        rows.append(
            (rid, f"{title} • {year} • {label} • {catno} • {country} • {fmt}"))
    root = _tk._default_root or _tk.Tk()
    win = _tk.Toplevel(root)
    win.title("Choose release")
    win.configure(bg=COLORS.get('bg_panel', '#111827'))
    _tk.Label(win, text="Select the correct release:", bg=COLORS.get('bg_panel', '#111827'),
              fg=COLORS.get('fg_primary', '#e5e7eb')).pack(padx=10, pady=8, anchor='w')
    lb = _tk.Listbox(win)
    lb.pack(fill='both', expand=True, padx=10, pady=6)
    for _, desc in rows:
        lb.insert(_tk.END, desc)
    choice = {"rid": 0}

    def _ok():
        idx = lb.curselection()
        if idx:
            choice["rid"] = int(rows[idx[0]][0] or 0)
        win.destroy()

    def _cancel():
        choice["rid"] = 0
        win.destroy()
    btns = _tk.Frame(win, bg=COLORS.get('bg_panel', '#111827'))
    btns.pack(fill='x', padx=10, pady=8)
    _tk.Button(btns, text="OK", command=_ok).pack(side='left')
    _tk.Button(btns, text="Cancel", command=_cancel).pack(side='left', padx=6)
    win.transient(root)
    win.grab_set()
    root.wait_window(win)
    return choice["rid"]


DEBUG_ENRICH = True


def _norm_key(k: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (k or '').lower())


def _value_to_str(v):
    if isinstance(v, list):
        return ", ".join(map(str, v))
    if isinstance(v, dict):
        return json.dumps(v, separators=(",", ":"))
    return "" if v is None else str(v).strip()


# Aliases across categories so we accept minor variants the model emits

# Aliases across categories so we accept minor variants the model emits
ALIASES = {
    "artist": ("primary artist", "main artist"),
    "release_title": ("title", "album title", "release"),
    "labels_all": ("labels", "label(s)", "record label"),
    "release_year": ("year", "released", "release date"),
    "genres": ("genre",),
    "styles": ("style",),
    "tracklist": ("tracks", "track list"),
    "total_runtime": ("runtime", "duration", "total run time"),
    "format": ("media format", "format type"),
    "size": ("record size", "disc size"),
    "rpm": ("speed", "play speed"),
    "weight_grams": ("weight", "vinyl weight"),
    "color": ("colour", "vinyl color"),
    "pressing_plant": ("plant", "pressing plant"),
    "edition_notes": ("edition", "edition notes"),
    "contributors_primary": ("credits", "contributors"),
    "discogs_median_price_usd": ("median price usd", "discogs median price"),
    "discogs_low_high_usd": ("low high usd", "price range usd"),
    "description": ("desc",),
    "fact": ("fun fact", "trivia"),

    # natural items
    "scientific_name": ("scientific name", "species", "latin name", "binomial name"),
    "subcategory":     ("sub category", "sub-type", "type", "group", "family"),
    "material":        ("materials", "composition", "made of", "material type"),
    "common_uses":     ("common uses", "uses", "typical uses", "applications", "function"),
    "toxicity_safety": ("toxicity safety", "toxicity", "safety", "hazards", "handling"),
    "specimen_type":   ("type", "specimen type", "category", "group"),
    "common_name":     ("common name", "vernacular name"),
    "taxonomic_rank":  ("taxon rank", "rank", "order", "family", "clade"),
    "size_range":      ("size", "dimensions", "size range"),
    "identification_tips": ("identification", "id tips", "how to identify", "diagnostic features"),
    "care_preservation":   ("care", "preservation", "storage care", "handling"),

    # vinyl
    "release_title": ("title", "album title", "release"),
    "labels_all":    ("labels", "label(s)"),
    "release_year":  ("year", "released"),
    "total_runtime": ("runtime", "duration"),
    "pressing_plant": ("plant", "pressing plant"),
    "contributors_primary": ("credits", "contributors"),
    "discogs_median_price_usd": ("median price usd", "discogs median price"),
    "discogs_low_high_usd":    ("low high usd", "price range usd"),

}


def map_llm_json_to_keys(parsed: dict, key_list: list) -> dict:
    normed = {}
    if isinstance(parsed, dict):
        for kk, vv in parsed.items():
            normed[_norm_key(kk)] = vv

    out = {}
    for k in key_list:
        cands = (k,) + ALIASES.get(k, ())
        val = ""
        for c in cands:
            v = normed.get(_norm_key(c))
            if v not in (None, "", [], {}):
                val = _value_to_str(v)
                break
        out[k] = val
    return out

# --- OpenAI client (uses your existing _openai_client) ---


DATA_URL_CACHE = {}  # key: (abs_path, mtime, tag) -> data_url


def _prep_for_vision(path: str, max_dim: int = 1024, quality: int = 85) -> str:
    """
    Load, downscale to max_dim, and write to a temp JPEG for fast upload.
    Returns the path of the prepared file (or original on error).
    """
    try:
        import cv2
        import os
        import tempfile
        if not os.path.exists(path):
            return path
        img = cv2.imread(path)
        if img is None:
            return path
        h, w = img.shape[:2]
        scale = max_dim / float(max(h, w))
        if scale < 1.0:
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out = os.path.join(tempfile.gettempdir(), "artifacts_vision.jpg")
        cv2.imwrite(out, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return out
    except Exception:
        return path


def _img_to_data_url(path: str) -> str:
    try:
        ap = os.path.abspath(path)
        mt = os.path.getmtime(ap)
        key = (ap, mt, "jpeg1024")
        if key in DATA_URL_CACHE:
            return DATA_URL_CACHE[key]
        # ensure small upload
        small = _prep_for_vision(ap, max_dim=1024, quality=85)
        with open(small, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = (os.path.splitext(small)[1] or ".jpg").lower().replace(".", "")
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        url = f"data:image/{ext};base64,{b64}"
        DATA_URL_CACHE[key] = url
        return url
    except Exception:
        return ""


def enrich_generic_openai(name: str, category: str, photo_path: str, helpers: dict, key_list: list) -> dict:
    client = _openai_client()

    data_url = _img_to_data_url(photo_path) if photo_path else ""
    provenance = helpers.get("provenance", "").strip()
    date_hint = helpers.get("collected_date", "").strip()

    # strict key template for the model to fill
    keys_json = "{" + ",".join([f'"{k}":""' for k in key_list]) + "}"

    system = (
        "You are a master cataloger and professional archivist. "
        "Assume the role of a renowned domain expert for the given category: "
        "minerals/shells/fossils → experienced field geologist and museum registrar; "
        "vinyl → Discogs-grade metadata editor and record store buyer; "
        "coins → PSA/NGC-style grader with auction catalog experience. "
        "Your job: produce strictly structured, accurate catalog metadata. "
        "Never invent facts—leave the field blank if uncertain. "
        "Return ONLY minified JSON with exactly the provided keys."
    )

    guidance = ""
    if category.lower() == "shell":
        guidance = (
            "\nField guidance for shells:"
            "\n- subcategory = common group (e.g., whelk, conch, cowrie, scallop, cone, murex, tulip)."
            "\n- material = 'calcium carbonate (aragonite)' unless a known exception applies."
            "\n- common_uses = short list like 'decorative, jewelry, teaching specimens'."
            "\n- toxicity_safety = 'non-toxic' unless a specific hazard is known."
            "\n- description = 1–2 plain sentences summarizing appearance and notable traits."
            "\n- fact = one short, interesting, verifiable tidbit."
        )
    elif category.lower() == "fossil":
        guidance = (
            "\nField guidance for fossils:"
            "\n- toxicity_safety = 'non-toxic' unless the specimen itself is hazardous."
            "\n- description = 1–2 plain sentences summarizing the fossil’s appearance and significance."
            "\n- fact = one short, interesting, verifiable tidbit (age, locality, or paleo note)."
        )
    elif category.lower() == "zoological":
        guidance = (
            "\nField guidance for zoological specimens:"
            "\n- Focus on the preserved organism itself (insect, bone, tooth, feather, antler, skin, etc.), not its container or frame."
            "\n- scientific_name = Genus species if visible/known; otherwise give the safest higher rank (order/family)."
            "\n- specimen_type = broad type (e.g., insect, butterfly, beetle, bone, tooth, feather, skin/taxidermy)."
            "\n- common_name = simple descriptive name (e.g., butterfly, scarab beetle, deer antler)."
            "\n- taxonomic_rank = the safest confident rank; prefer order/family if species is unclear."
            "\n- size_range = dimensions of the specimen itself (not the case)."
            "\n- identification_tips = 1–2 cues about the specimen’s visible features (wing patterns, tooth shape, bone morphology)."
            "\n- material = keratin, chitin, bone (hydroxyapatite), etc., depending on the specimen."
            "\n- care_preservation = tips on caring for the specimen (avoid UV, humidity, pests)."
            "\n- toxicity_safety = 'non-toxic' unless known hazards exist (e.g., dust precautions)."
            "\n- description = 1–2 sentences about the organism itself, ignoring its container."
            "\n- fact = one short, verifiable fact about that type of specimen."
            "\n- Ignore frames, boxes, cases, and presentation styles—they are irrelevant to the scientific specimen."
        )

    if category.lower() in {"mineral", "shell", "fossil", "zoological"}:
        guidance += (
            "\n- If the photo shows a display case/shadow box, ignore the container and describe only the specimen."
            "\n- Do not call the item a 'display case' or 'shadow box'."
        )

    user_text = (
        f"Item category: {category}\n"
        f"Label/name: {name}\n"
        f"Provenance: {provenance}\n"
        f"Collected/Purchase date: {date_hint}\n\n"
        f"Return JSON with these keys exactly:\n{keys_json}\n"
        "- Use plain strings; units OK; no markdown.\n"
        "- If uncertain, use empty string."
        f"{guidance}"
    )

    content = [{"type": "input_text", "text": user_text}]
    if data_url:
        content.append({"type": "input_image", "image_url": data_url})

    resp = _openai_client().responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": [
                {"type": "input_text", "text": system}]},
            {"role": "user", "content": content}
        ],
        temperature=0.2, max_output_tokens=700
    )

    text = getattr(resp, "output_text", "").strip()
    # robust parse
    try:
        parsed = json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        parsed = json.loads(text[s:e+1]) if (s != -
                                             1 and e != -1 and e > s) else {}

    out = map_llm_json_to_keys(parsed, key_list)

    # Shell heuristics/fallbacks
    if category.lower() == "shell":
        def _guess_shell_subcategory(nm: str) -> str:
            nm = (nm or "").lower()
            for kw in ("whelk", "conch", "cowrie", "scallop", "cone", "murex", "tulip", "abalone", "cockle", "olive", "triton", "turban"):
                if kw in nm:
                    return kw
            return ""
        if not out.get("subcategory"):
            out["subcategory"] = _guess_shell_subcategory(name)
        if not out.get("material"):
            out["material"] = "calcium carbonate (aragonite)"
        if not out.get("toxicity_safety"):
            out["toxicity_safety"] = "non-toxic"

    return out


def enrich_vinyl(name: str, photo_path: str, helpers: dict) -> dict:
    client = _openai_client()
    key_list = SCHEMAS["vinyl"]["api"]

    runout_a = helpers.get("runout_deadwax_code_side_a", "").strip()
    runout_b = helpers.get("runout_deadwax_code_side_b", "").strip()
    user_barcode = (helpers.get("barcode_ean_upc") or "").strip()
    keys_json = "{" + ",".join([f'"{k}":""' for k in key_list]) + "}"

    system = (
        "You are a Discogs-grade metadata editor and record buyer with deep catalog knowledge. "
        "Act like a mastering engineer meets archivist: read labels, jackets, and runouts rigorously. "
        "Avoid hallucinations; leave fields blank if unknown. "
        "Return ONLY minified JSON using the exact keys provided."
    )
    guidance = (
        "- tracklist as 'A1 …; A2 …; B1 …' is fine.\n"
        "- If color/weight/rpm unknown, leave blank.\n"
        "- If no price data, leave price fields blank."
    )

    user_text = (
        f"Item category: vinyl\n"
        f"Label/name on sleeve or user label: {name}\n"
        f"Runout/deadwax A: {runout_a}\n"
        f"Runout/deadwax B: {runout_b}\n\n"
        f"Return JSON with these keys exactly:\n{keys_json}\n"
        f"{guidance}"
    )

    content = [{"type": "input_text", "text": user_text}]
    data_url = _img_to_data_url(photo_path) if photo_path else ""
    if data_url:
        content.append({"type": "input_image", "image_url": data_url})

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": [
                {"type": "input_text", "text": system}]},
            {"role": "user", "content": content}
        ],
        temperature=0.2, max_output_tokens=900
    )

    text = getattr(resp, "output_text", "").strip()
    try:
        gpt_parsed = json.loads(text)
    except Exception:
        s_idx, e_idx = text.find("{"), text.rfind("}")
        gpt_parsed = json.loads(
            text[s_idx:e_idx+1]) if (s_idx != -1 and e_idx != -1 and e_idx > s_idx) else {}

    discogs_data = {}
    release_id = 0
    try:
        artist_q = (helpers.get("artist") or "").strip() or (
            gpt_parsed.get("artist") or "").strip()
        title_q = (helpers.get("release_title") or "").strip() or (
            gpt_parsed.get("release_title") or "").strip()
        barcode_q = user_barcode or ((gpt_parsed.get("barcode_ean_upc") or "").split(
            ";")[0].strip() if isinstance(gpt_parsed.get("barcode_ean_upc"), str) else "")
        catno_q = (gpt_parsed.get("catalog_number") or "").strip() if isinstance(
            gpt_parsed.get("catalog_number"), str) else ""

        results = discogs_search_release(
            artist=artist_q, title=title_q, barcode=barcode_q, catno=catno_q, per_page=10)
        if results:
            import threading
            if len(results) > 1 and threading.current_thread() is threading.main_thread():
                rid = choose_discogs_release(results[:3])
            else:
                rid = int(results[0].get("id") or 0)
            if rid:
                release_id = rid
            else:
                best = _pick_best_discogs_match(
                    results, artist=artist_q, title=title_q)
                release_id = int(best.get("id") or 0) if best else 0

        if release_id:
            release = discogs_get_release(release_id)
            discogs_data = map_discogs_release_to_schema(
                release) if release else {}

            try:
                sugg = discogs_price_suggestions(release_id) or {}
                vgplus = sugg.get("Very Good Plus (VG+)") or sugg.get("VG+")
                nm = sugg.get("Near Mint (NM or M-)") or sugg.get("NM or M-")
                target = vgplus or nm
                if target and isinstance(target, dict) and target.get("value"):
                    discogs_data["discogs_median_price_usd"] = f"${float(target['value']):.2f}"
                stats = discogs_release_stats(release_id, "USD") or {}
                low = (stats.get("lowest_price") or {}).get("value")
                high = (stats.get("highest_price") or {}).get("value") or None
                if low:
                    if high and float(high) >= float(low):
                        discogs_data["discogs_low_high_usd"] = f"${float(low):.2f} - ${float(high):.2f}"
                    else:
                        discogs_data["discogs_low_high_usd"] = f"${float(low):.2f}+"
            except Exception:
                pass
    except Exception:
        discogs_data = {}

    merged = merge_gpt_and_discogs(gpt_parsed, discogs_data)
    if DEBUG_ENRICH:
        print("== VINYL GPT parsed ==", gpt_parsed)
        print("== VINYL Discogs data ==", discogs_data)
        print("== VINYL merged ==", merged)
    return map_llm_json_to_keys(merged, key_list)


def enrich_coin(name: str, helpers: dict) -> dict:
    key_list = SCHEMAS["coin"]["api"]
    # We don’t have the photo path here; pass empty string (fine for coins).
    return enrich_generic_openai(name, "coin", "", helpers, key_list)


# Your image classifier module (unchanged)
# --- OpenAI client
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


def _openai_client():
    if not _OPENAI_OK:
        raise RuntimeError("Missing dependency: pip install openai>=1.40")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable")
    return OpenAI(api_key=api_key)


COLORS = {
    'bg_app':      '#0f172a',
    'bg_panel':    '#111827',
    'fg_primary':  '#e5e7eb',
    'fg_muted':    '#9ca3af',
    'accent_a':    '#22d3ee',
    'accent_b':    '#a78bfa',
    'accent_warn': '#f59e0b',
    'blurb_fg':    '#0d3b66',
    'border':      '#1f2937',
}

# --- Blurb UI config (smooth, semi-transparent bubble) ---
BLURB_MAX_W = 280                # wrap width for text
BLURB_FONT = ('Comic Sans MS', 11)   # hand-written style font
BLURB_ALPHA = 220                # 0..255; 220 ≈ 86% opaque
BLURB_RADIUS = 12                # corner radius
BLURB_SHADOW_BLUR = 6            # shadow softness; 0 to disable

# Position in the photos canvas: 'top-left'|'top-right'|'bottom-left'|'bottom-right'|'custom'
BLURB_POSITION = 'bottom-left'
# Pixels from the chosen corner (or absolute when using 'custom')
BLURB_OFFSET_X, BLURB_OFFSET_Y = 24, 24


def _blurb_anchor_and_xy(self, bubble_w: int, bubble_h: int):
    """
    Return (anchor, x, y) to place the blurb relative to the CURRENTLY VISIBLE
    region of the photos canvas, so it tracks as you scroll.
    """
    # Visible viewport in canvas coords
    vx = self.image_canvas.canvasx(0)
    vy = self.image_canvas.canvasy(0)
    vw = self.image_canvas.winfo_width()
    vh = self.image_canvas.winfo_height()

    pos = (BLURB_POSITION or 'bottom-left').lower().strip()
    ax, ay = BLURB_OFFSET_X, BLURB_OFFSET_Y

    if pos == 'top-left':
        return 'nw', vx + ax, vy + ay
    if pos == 'top-right':
        return 'ne', vx + vw - ax, vy + ay
    if pos == 'bottom-right':
        return 'se', vx + vw - ax, vy + vh - ay
    if pos == 'custom':
        # Treat offsets as absolute canvas coords if you really want to pin it
        return 'nw', BLURB_OFFSET_X, BLURB_OFFSET_Y
    # default: bottom-left
    return 'sw', vx + ax, vy + vh - ay


DB_PATH = "collection_catalog.db"
PHOTO_DIR = "photos"
PHOTO_SLOTS = {
    "vinyl": 4,
    "coin": 2,
    "fossil": 4,
    "shell": 4,
    "mineral": 4,
    "zoological": 4,
}

LIBRARY_CATEGORIES = [
    "mineral", "shell", "fossil",
    "vinyl", "coin", "zoological", "other"
]
# Carousel sizing
CAROUSEL_CARD_W = 680
CAROUSEL_CARD_H = 420
CAROUSEL_PAD_X = 10
CAROUSEL_PAD_Y = 8

# Smooth auto-scroll knobs (tweak to taste)
CAROUSEL_SCROLL_FPS = 60   # animation tick rate
CAROUSEL_SCROLL_SPEED_PX = 80   # target speed in px/sec
CAROUSEL_SCROLL_EASE = 0.12  # 0..1; higher = snappier accel/decel

# Expert personas and example blurbs
EXPERT_PERSONAS = {
    "minerals": {
        "persona": "Measured Mineralogist",
        "examples": [
            "Quartz is one of the most widespread minerals on Earth. Its structure is basic, but it remains a cornerstone for identifying other specimens.",
            "Fluorite is valued for its cubic form and fluorescence. Its scientific importance outweighs its modest economic use."
        ]
    },
    "shells": {
        "persona": "Reserved Marine Biologist",
        "examples": [
            "This cowrie was once used as currency. Its polished surface made it practical to carry, though it was valued more for appearance than durability.",
            "Scallop shells are common along shorelines. Their symmetrical form is aesthetically pleasing, though it’s purely a byproduct of growth."
        ]
    },
    "fossils": {
        "persona": "Pragmatic Paleontologist",
        "examples": [
            "This ammonite is common in Cretaceous deposits. Its coiled shell structure is a textbook case of buoyancy control in marine invertebrates.",
            "Knightia fossils are frequent in Green River Formation deposits. They represent entire schools preserved in a single event."
        ]
    },
    "vinyl": {
        "persona": "Music Critic",
        "examples": [
            "On this LP, the fuzzed-out guitar riffs defined proto-metal’s raw DNA."
        ]
    },
    "zoological": {
        "persona": "Museum Anatomist",
        "examples": [
            "Shark teeth are abundant because sharks continually shed them. The abundance makes them accessible, though it reduces their rarity.",
            "Snake skins are preserved when the animal sheds in one piece. They are common finds, though rarely intact enough for study."
        ]
    },
    "other": {
        "persona": "Archivist of the Strange",
        "examples": [
            "This taxidermy mount reflects a period when curiosity cabinets blurred the line between science and spectacle.",
            "This oddity reflects a period when private collections blurred the line between science and spectacle."
        ]
    }
}

try:
    import openai  # optional; used if available
except Exception:
    openai = None

# ---------------- Expert blurb: deterministic, item-aware fallback ----------------


def _contains_any(s: str, words: tuple[str, ...]) -> bool:
    sl = (s or "").lower()
    return any(w in sl for w in words)


def _fallback_blurb(category: str, item_name: str) -> str:
    """Return a concise, safe, persona-styled blurb that references the *selected* item."""
    c = (category or "other").strip().lower()
    name = (item_name or "This specimen").strip()

    if c in ("mineral", "minerals"):
        return (
            f"{name} is evaluated by hardness, luster, and cleavage. "
            "Simple traits, but they’re the backbone of reliable identification."
        )

    if c in ("shell", "shells"):
        if _contains_any(name, ("cowrie",)):
            return (
                f"{name} has a polished surface from growth against the mantle. "
                "Its appeal is visual; the form is just the outcome of growth."
            )
        if _contains_any(name, ("scallop", "pecten")):
            return (
                f"{name} shows radial ribs and bilateral symmetry. "
                "Symmetry reads as design; it’s merely how the shell lays itself down."
            )
        if _contains_any(name, ("conch", "whelk", "murex", "tulip")):
            return (
                f"{name} is a spiral gastropod shell. "
                "Spiral thickening adds strength; ornament is incidental."
            )
        return (
            f"{name} is calcium carbonate (aragonite) laid in layers. "
            "Patterning is a byproduct of growth, not intention."
        )

    if c in ("fossil", "fossils"):
        if _contains_any(name, ("tooth", "teeth")):
            return (
                f"{name} is a fossil tooth. Enamel preserves well, which is why tooth fossils are common; "
                "shape records feeding mechanics more than species identity."
            )
        if _contains_any(name, ("ammonite",)):
            return (
                f"{name} is a coiled cephalopod shell fossil. Chambered construction is a standard case of buoyancy control."
            )
        if _contains_any(name, ("trilobite",)):
            return (
                f"{name} preserves dorsal exoskeleton segments. Molts and complete bodies are both found; complete ones are rarer."
            )
        if _contains_any(name, ("sand dollar", "echinoid", "sea biscuit")):
            return (
                f"{name} represents an echinoid test. Star patterns mark ambulacral areas; fine details survive best in quiet sediments."
            )
        return (
            f"{name} records morphology and context from deep time. "
            "Scientific value exceeds decorative value; specifics depend on locality."
        )

    if c in ("zoological",):
        if _contains_any(name, ("tooth", "teeth", "fang", "tusk")):
            return (
                f"{name} is dentition. Enamel’s hardness aids preservation; wear facets and shape are more diagnostic than color."
            )
        if _contains_any(name, ("antler", "horn")):
            return (
                f"{name} shows growth rings and vascular traces. Breakage patterns reveal stress more reliably than surface polish."
            )
        if _contains_any(name, ("bone", "skull", "vertebra")):
            return (
                f"{name} is osseous tissue (hydroxyapatite). Articulations matter; isolated pieces are common, complete series are not."
            )
        if _contains_any(name, ("feather",)):
            return (
                f"{name} preserves vane and rachis structure. UV and pests are the main risks; climate control is more important than display."
            )
        if _contains_any(name, ("skin", "hide", "taxidermy")):
            return (
                f"{name} is preserved integument. Oils and light degrade keratin; storage practice determines longevity."
            )
        return (
            f"{name} is a preserved organismal specimen. Diagnostic features, not mounting style, determine identification."
        )

    if c in ("vinyl",):
        return (
            f"On {name}, arrangement and tone shape the record more than volume. "
            "The cut emphasizes the midrange where the songs actually live."
        )

    return (
        f"{name} reflects a moment when collecting blurred curiosity and study. "
        "Context explains why it was kept; the object itself stays unapologetically odd."
    )


# ===== UNIQUE, ITEM-SPECIFIC EXPERT BLURBS (vision+details, with caching) =====

def _first_photo_for_item(detail_map: dict, fallback_path: str) -> str:
    """Prefer img1_path, then any img*_path, then fallback_path."""
    if not isinstance(detail_map, dict):
        return fallback_path
    p = detail_map.get("img1_path", "")
    if p and os.path.exists(p):
        return p
    # try other slots
    for k, v in sorted(detail_map.items()):
        if k.startswith("img") and k.endswith("_path") and v and os.path.exists(v):
            return v
    return fallback_path or ""


def _item_context_for_blurb(category: str, name: str, details: dict) -> str:
    """Compress per-category fields into a few high-signal bullets."""
    cat = (category or "").lower()
    d = details or {}
    lines = [f"category: {cat}", f"name: {name}"]

    def add(k, label=None):
        v = (d.get(k) or "").strip()
        if v:
            lines.append(f"{(label or k).replace('_',' ')}: {v}")

    if cat == "mineral":
        for k in ("scientific_name", "mohs_hardness", "cleavage_fracture", "habit_or_form", "typical_colors", "description", "fact"):
            add(k)
    elif cat == "shell":
        for k in ("scientific_name", "subcategory", "material", "description", "fact"):
            add(k)
    elif cat == "fossil":
        for k in ("scientific_name", "fossil_type", "geological_period", "estimated_age", "preservation_mode", "description", "fact"):
            add(k)
    elif cat == "zoological":
        for k in ("scientific_name", "specimen_type", "taxonomic_rank", "identification_tips", "size_range", "material", "description", "fact"):
            add(k)
    elif cat == "vinyl":
        for k in ("artist", "release_title", "release_year", "genres", "styles", "description", "fact"):
            add(k)
    else:
        for k in ("description", "fact"):
            add(k)
    # keep it short; the model just needs hooks to be specific
    return "\n".join(lines[:14])


def _persona_for_category(cat_key: str) -> str:
    info = EXPERT_PERSONAS.get(cat_key, EXPERT_PERSONAS["other"])
    return info.get("persona", "Archivist of the Strange")


def _catkey(c: str) -> str:
    m = {"mineral": "minerals", "shell": "shells", "fossil": "fossils"}
    return m.get((c or "").lower(), (c or "other").lower())


# --- Style profiles (choose one in BLURB_STYLE_PROFILE) ---
# 'house' | 'attenborough_hint' | 'sagan_hint' | 'rollingstone_70s'
BLURB_STYLE_PROFILE = 'house'
STYLE_HINTS = {
    'house': "",
    'attenborough_hint': "Write with the calm, observational cadence of a natural history narrator; respectful, precise, no theatrics.",
    'sagan_hint': "Carry a brief note of wonder—no more than six words—woven into plainspoken clarity.",
    'rollingstone_70s': "Lean into album-era criticism: texture, scene, and moment—no star-ratings language."
}


def _category_rules(category: str) -> str:
    c = (category or "").lower()
    if c == "mineral":
        return ("Never mention Mohs hardness, cleavage, fracture, chemical formulas, or the phrase 'variety of chalcedony'. "
                "Offer one visible, checkable trait or a locality/provenance angle that helps a visitor see the piece differently.")
    if c == "zoological":
        return ("Do not mention physical size, dimensions, inches, centimeters, weight, or 'specimen size'. "
                "Prefer preservation notes, diagnostic morphology, or behavior/ecology context in plain language.")
    if c == "fossil":
        return ("Avoid geological period name-drops unless in the item name; prefer one note on preservation/taphonomy or collection context.")
    if c == "shell":
        return ("Avoid chemical composition; prefer aperture, spire, ribs, pattern, or wear—things a visitor can actually notice.")
    return "Avoid repeating catalog fields; add one concrete observation."


# Build a dynamic "do-not-repeat" list from item details so the blurb doesn't parrot enrichment fields.
def _collect_banned_phrases(details: dict, category: str) -> list[str]:
    if not isinstance(details, dict):
        return []
    d = {k.lower(): (v or "").strip()
         for k, v in details.items() if isinstance(v, str)}
    ban = set()

    # High-redundancy fields likely to be parroted
    for k in (
        "description", "fact", "scientific_name", "mohs_hardness", "material", "composition",
        "luster", "typical_colors", "color", "cleavage_fracture", "habit_or_form",
        "geological_period", "estimated_age", "preservation_mode", "size", "dimensions", "weight", "length", "width", "height"
    ):
        v = d.get(k, "")
        if v:
            ban.add(v)

    # Technical clichés / fragments
    tech_bits = [
        "mohs", "hardness", "luster", "streak", "cleavage", "fracture",
        "chemical formula", "composition", "matrix", "variety of chalcedony",
        "calcite", "aragonite", "calcium carbonate", "silicon dioxide", "sio2",
        "specimen size", "inches", "inch", "cm", "centimeter", "mm", "millimeter"
    ]
    ban.update(tech_bits)

    # Numbers/units from details blob
    for token in re.findall(r"\b\d+(?:\.\d+)?\s?(?:cm|mm|in|inch|inches|g|kg|yrs?|years?|ma|million)?", " ".join(d.values()), re.I):
        ban.add(token.lower())

    # Category-specific add-ons
    cat = (category or "").lower()
    if cat == "mineral":
        ban.update({"conchoidal fracture", "variety of chalcedony"})
    elif cat == "zoological":
        ban.update({"length", "width", "height",
                   "tall", "long", "weighs", "weight"})
    return [s for s in list(ban) if s][:28]

# Pick a single narrative angle so the line feels like a human observation, not a datasheet.


def _pick_commentary_angle(category: str, details: dict) -> str:
    c = (category or "").lower()
    d = details or {}
    prov = (d.get("provenance") or d.get("origin") or "").strip()
    if prov:
        return f"Give a subtle curatorial note that situates the piece in its place of origin or find context ({prov})."
    if c in ("fossil", "zoological") and (d.get("preservation_mode") or d.get("condition")):
        return "Comment briefly on preservation/condition and what that implies for study or display (no jargon)."
    if c == "vinyl" and (d.get("release_year") or d.get("genres") or d.get("styles")):
        return "Situate the record in its era/scene with one concrete sonic or scene detail (no track-by-track)."
    if c == "mineral" and (d.get("habit_or_form") or d.get("cleavage_fracture")):
        return "Point to one visually checkable trait a casual observer can notice (e.g., growth zoning, twinning)."
    if c == "shell" and (d.get("subcategory") or d.get("scientific_name")):
        return "Note a small, visible trait (ribbing, aperture shape, spiral) that separates it from lookalikes."
    return "Offer one specific, viewer-facing observation that adds meaning beyond basic facts."


# Light post-edit to remove robotic scaffolding and banned fragments that slipped through.
_SCAFFOLD_RE = re.compile(
    r"(?i)\b(?:is|are|it is|this (?:item|specimen|fossil|shell|record)) (?:known for|characterized by|features|has)\b")


def _humanize_blurb(text: str, banned: list[str]) -> str:
    s = (text or "").strip()
    if not s:
        return s
    # prune boilerplate
    s = _SCAFFOLD_RE.sub("shows", s)
    # “a variety of chalcedony…"
    s = re.sub(r"(?i)\b,?\s*a (?:variety|type) of [^,.;]+,?\s*", " ", s)
    s = re.sub(r"(?i)\b(?:it is|it’s|it is made of|is composed of)\b", "is", s)
    # remove banned fragments
    for b in banned:
        if b and b.lower() in s.lower():
            s = re.sub(re.escape(b), "", s, flags=re.I)
    # clean spacing/punct
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s).strip()
    # keep to 1–2 sentences
    parts = re.split(r"(?<=[.!?])\s+", s)
    return " ".join(parts[:2]).strip()


def generate_expert_blurb(category: str, item_name: str, details: dict | None = None, photo_path: str | None = None) -> str:
    # Cache hit?
    if isinstance(details, dict):
        cached = (details.get("expert_blurb") or "").strip()
        if cached:
            return cached

    persona = _persona_for_category(_catkey(category))
    context = _item_context_for_blurb(category, item_name, details or {})
    banned = _collect_banned_phrases(details or {}, category)
    angle = _pick_commentary_angle(category, details or {})
    style_hint = STYLE_HINTS.get(BLURB_STYLE_PROFILE, "")

    sys = (
        f"You are the {persona}. Write ONE or TWO sentences as a tight, human-sounding expert opinion for the specific item.\n"
        f"{style_hint}\n"
        f"Rules: {_category_rules(category)} "
        f"Write like a curator speaking to a visitor; add meaning beyond catalog facts. "
        f"No lists, no numbers unless essential; avoid formulas, Mohs, and checklist jargon. "
        f"Length: 28–55 words. No markdown."
    )
    user = (
        f"ITEM FACTS (for context, not to copy verbatim):\n{context}\n\n"
        f"ANGLE: {angle}\n"
        f"AVOID repeating these words/phrases verbatim: {', '.join(banned)}\n\n"
        f"Return only the blurb."
    )

    # Try vision + details via OpenAI (with slightly higher creativity)
    try:
        client = _openai_client()
        content = [{"type": "input_text", "text": user}]
        img = _first_photo_for_item(details or {}, photo_path or "")
        if img and os.path.exists(img):
            content.append(
                {"type": "input_image", "image_url": _img_to_data_url(img)})

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "system", "content": [{"type": "input_text", "text": sys}]},
                   {"role": "user",   "content": content}],
            temperature=0.6,
            max_output_tokens=140
        )
        raw = (_safe_first_text(resp) or "").strip()
        if raw:
            out = _humanize_blurb(raw, banned)
            if 20 <= len(out) <= 320:
                return out
    except Exception:
        pass

    # Deterministic, item-aware fallback if API fails
    return _humanize_blurb(_fallback_blurb(category, item_name), banned)


def save_blurb_to_db(item_id: int, text: str):
    """Store the generated blurb so it persists and renders instantly next time."""
    if not (item_id and text):
        return
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("""INSERT INTO item_details(item_id, key, value)
                       VALUES(?,?,?)
                       ON CONFLICT(item_id, key) DO UPDATE SET value=excluded.value""",
                    (item_id, "expert_blurb", text))
        con.commit()
        con.close()
    except Exception:
        pass


# OPTIONAL: keep a thin wrapper for legacy calls
def get_expert_blurb(category: str, item_name: str, details: dict | None = None, photo_path: str | None = None) -> str:
    return generate_expert_blurb(category, item_name, details, photo_path)


# ---------- DB ----------


def _ensure_tables():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Make sure the old items table exists (some installs may not have it yet)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            subcategory TEXT,
            description TEXT,
            fact TEXT,
            found_on TEXT,
            found_location TEXT,
            estimated_age TEXT,
            material TEXT,
            display_case TEXT,
            notes TEXT,
            photo_path TEXT,
            added_on DATETIME DEFAULT CURRENT_TIMESTAMP,
            scientific_name TEXT,
            region TEXT,
            era_or_epoch TEXT,
            common_uses TEXT,
            toxicity_safety TEXT
        )
    """)

    # Ensure auxiliary details table exists with composite unique key for upserts
    cur.execute("""
        CREATE TABLE IF NOT EXISTS item_details (
            item_id INTEGER,
            key TEXT,
            value TEXT
        )
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_item_details_item_key
        ON item_details (item_id, key)
    """)

    # Discover current columns on 'items'
    cur.execute("PRAGMA table_info(items)")
    old_cols = [r[1] for r in cur.fetchall()]

    # Desired final schema (with UNIQUE(name, category))
    new_cols = [
        "id", "name", "category", "subcategory", "description", "fact",
        "found_on", "found_location", "estimated_age", "material",
        "display_case", "notes", "photo_path", "added_on",
        "scientific_name", "region", "era_or_epoch", "common_uses",
        "toxicity_safety"
    ]

    # If either 'category' is missing OR the unique constraint is missing,
    # rebuild the table safely.
    needs_rebuild = ("category" not in old_cols)

    if not needs_rebuild:
        # Check if UNIQUE(name, category) exists
        cur.execute("PRAGMA index_list(items)")
        idx_rows = cur.fetchall()
        unique_ok = False
        for _, idx_name, _, _, _ in idx_rows:
            cur.execute(f"PRAGMA index_info({idx_name})")
            cols = [r[2] for r in cur.fetchall()]
            # A unique index on (name, category) could be named anything
            if set(cols) == {"name", "category"}:
                # Confirm it's actually UNIQUE
                cur.execute(
                    f"SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (idx_name,))
                sql = (cur.fetchone() or [""])[0] or ""
                if "UNIQUE" in sql.upper():
                    unique_ok = True
                    break
        if not unique_ok:
            needs_rebuild = True

    if needs_rebuild:
        # If a previous migration left behind a partial items_new, drop it so we
        # can recreate the table with the correct schema (including new cols)
        cur.execute("DROP TABLE IF EXISTS items_new")

        # Build the new table with the desired schema + unique constraint
        cur.execute("""
            CREATE TABLE items_new (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                subcategory TEXT,
                description TEXT,
                fact TEXT,
                found_on TEXT,
                found_location TEXT,
                estimated_age TEXT,
                material TEXT,
                display_case TEXT,
                notes TEXT,
                photo_path TEXT,
                added_on DATETIME DEFAULT CURRENT_TIMESTAMP,
                scientific_name TEXT,
                region TEXT,
                era_or_epoch TEXT,
                common_uses TEXT,
                toxicity_safety TEXT,
                UNIQUE(name, category)
            )
        """)

        # Build a SELECT that pulls what exists, fills what doesn't.
        select_cols = []
        for col in new_cols:
            if col in old_cols:
                select_cols.append(col)
            elif col == "category":
                # Old installs may not have category yet
                select_cols.append("'' AS category")
            else:
                select_cols.append(f"NULL AS {col}")

        cur.execute(f"""
            INSERT INTO items_new ({', '.join(new_cols)})
            SELECT {', '.join(select_cols)} FROM items
        """)

        cur.execute("DROP TABLE items")
        cur.execute("ALTER TABLE items_new RENAME TO items")

    con.commit()
    con.close()


_ensure_tables()


def upsert_item(name: str, category: str, photo_path: str) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT id, category FROM items WHERE lower(name)=lower(?) AND lower(category)=lower(?)",
        (name, category))
    row = cur.fetchone()
    if row:
        item_id = row[0]
        if photo_path:
            cur.execute("UPDATE items SET photo_path=? WHERE id=?",
                        (photo_path, item_id))
    else:
        # check if item exists under a different category (category change)
        cur.execute(
            "SELECT id, category FROM items WHERE lower(name)=lower(?)", (name,))
        row = cur.fetchone()
        if row:
            item_id, old_cat = row
            if old_cat and old_cat.lower() != (category or '').lower():
                old_dir = os.path.join(
                    PHOTO_DIR, old_cat.lower(), str(item_id))
                shutil.rmtree(old_dir, ignore_errors=True)
                cur.execute(
                    "DELETE FROM item_details WHERE item_id=? AND key LIKE 'img%_path'", (item_id,))
                cur.execute(
                    "DELETE FROM item_details WHERE item_id=? AND key LIKE 'img%_src'", (item_id,))
                cur.execute(
                    "DELETE FROM item_details WHERE item_id=? AND key='upload_path'", (item_id,))
            if photo_path:
                cur.execute("UPDATE items SET category=?, photo_path=? WHERE id=?",
                            (category, photo_path, item_id))
            else:
                cur.execute("UPDATE items SET category=? WHERE id=?",
                            (category, item_id))
        else:
            cur.execute("INSERT INTO items(name, category, photo_path) VALUES (?,?,?)",
                        (name, category, photo_path))
            item_id = cur.lastrowid
    con.commit()
    con.close()
    return item_id


def save_item_full(name: str, category: str, photo_path: str, details: dict) -> int:
    item_id = upsert_item(name, category, photo_path)
    details = details or {}
    item_dir_abs = os.path.abspath(os.path.join(
        PHOTO_DIR, category.lower(), str(item_id)))
    uploaded_arg = photo_path if (
        photo_path and os.path.commonpath([
            os.path.abspath(photo_path), item_dir_abs]) == item_dir_abs
    ) else photo_path  # allow first-time external upload; populator will copy & promote
    photo_meta = populate_photo_slots(
        item_id, category, name, details, uploaded_arg)
    details.update(photo_meta)
    if details:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("PRAGMA table_info(items)")
        cols = {r[1] for r in cur.fetchall()}
        for k, v in details.items():
            if not k:
                continue
            if k in cols:
                cur.execute(f"UPDATE items SET {k}=? WHERE id=?", (str(
                    v) if v is not None else "", item_id))
            cur.execute("""INSERT INTO item_details(item_id, key, value)
                           VALUES(?,?,?)
                           ON CONFLICT(item_id, key) DO UPDATE SET value=excluded.value""",
                        (item_id, k, str(v) if v is not None else ""))
        con.commit()
        con.close()
    return item_id


def update_item_full(item_id: int, name: str, category: str, photo_path: str, details: dict) -> int:
    """Update an existing item and its details by ``item_id``.

    All fields in both the ``items`` table and ``item_details`` may be
    modified.  Category changes will move the associated photo directory.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT category, photo_path FROM items WHERE id=?", (item_id,))
    row = cur.fetchone()
    old_cat = row[0] if row else ""
    cur.execute("UPDATE items SET name=?, category=?, photo_path=? WHERE id=?",
                (name, category, photo_path, item_id))
    con.commit()
    con.close()

    # Move photo directory if category changed
    if old_cat and old_cat.lower() != (category or '').lower():
        old_dir = os.path.join(PHOTO_DIR, old_cat.lower(), str(item_id))
        new_dir = os.path.join(PHOTO_DIR, category.lower(), str(item_id))
        if os.path.exists(old_dir):
            os.makedirs(os.path.dirname(new_dir), exist_ok=True)
            shutil.move(old_dir, new_dir)

    details = details or {}
    item_dir_abs = os.path.abspath(os.path.join(
        PHOTO_DIR, category.lower(), str(item_id)))
    uploaded_arg = photo_path if (
        photo_path and os.path.commonpath([
            os.path.abspath(photo_path), item_dir_abs]) == item_dir_abs
    ) else photo_path
    photo_meta = populate_photo_slots(
        item_id, category, name, details, uploaded_arg)
    details.update(photo_meta)
    if details:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("PRAGMA table_info(items)")
        cols = {r[1] for r in cur.fetchall()}
        for k, v in details.items():
            if not k:
                continue
            if k in cols:
                cur.execute(f"UPDATE items SET {k}=? WHERE id=?",
                            (str(v) if v is not None else "", item_id))
            cur.execute("""INSERT INTO item_details(item_id, key, value)
                           VALUES(?,?,?)
                           ON CONFLICT(item_id, key) DO UPDATE SET value=excluded.value""",
                        (item_id, k, str(v) if v is not None else ""))
        con.commit()
        con.close()
    return item_id


# ---------- Tailored per-category schemas ONLY ----------
HIDDEN_META_KEYS = {"discogs_release_id", "discogs_url"}

SCHEMAS = {
    "mineral": {  # excludes region/era
        "user": ["name", "collected_date", "provenance", "purchase_price", "storage_or_display_location", "notes"],
        "api": [
            "scientific_name", "mohs_hardness", "luster", "streak",
            "cleavage_fracture", "habit_or_form", "specific_gravity", "formation_process",
            "typical_colors", "common_uses", "toxicity_safety", "description", "fact"
        ]
    },
    "shell": {
        "user": ["name", "collected_date", "provenance", "purchase_price", "storage_or_display_location", "notes"],
        "api": [
            "scientific_name", "subcategory", "material",
            "common_uses", "toxicity_safety", "description", "fact"
        ]
    },
    "fossil": {
        "user": ["name", "collected_date", "provenance", "purchase_price", "storage_or_display_location", "notes"],
        "api": [
            "scientific_name", "geological_period", "estimated_age", "fossil_type", "preservation_mode",
            "size_range", "typical_locations", "paleoecology", "toxicity_safety", "description", "fact"
        ]
    },
    "zoological": {
        "user": ["name", "collected_date", "provenance", "storage_or_display_location", "notes"],
        "api": [
            "scientific_name", "specimen_type", "common_name", "taxonomic_rank", "size_range",
            "identification_tips", "material", "care_preservation", "toxicity_safety", "description", "fact"
        ]
    },
    "vinyl": {
        "user": ["name", "purchase_price", "storage_or_display_location", "notes",
                 "runout_deadwax_code_side_a", "runout_deadwax_code_side_b", "barcode_ean_upc"],
        "api": [
            "artist", "release_title", "label", "labels_all", "country", "release_year", "genres", "styles", "tracklist", "total_runtime", "format", "size", "rpm", "weight_grams", "color", "pressing_plant", "edition_notes", "contributors_primary", "discogs_median_price_usd", "discogs_low_high_usd", "description", "fact", "discogs_release_id", "discogs_url"]
    },
    "coin": {
        "user": ["name", "collected_date", "provenance", "purchase_price", "storage_or_display_location", "notes"],
        "api": [
            "country", "denomination", "year_minted", "mint_mark", "issuing_authority", "series_name", "coin_type", "catalog_numbers", "composition", "weight_grams", "diameter_mm", "thickness_mm", "shape", "edge_type", "obverse_design", "obverse_designer", "reverse_design", "reverse_designer", "orientation", "mintage", "mint_location", "minting_technique", "original_face_value", "current_melt_value_usd", "catalog_value_ranges", "auction_record_price_usd", "description", "fact"
        ]
    },
    "other": {
        "user": ["name", "notes"],
        "api": ["description", "fact"]
    }
}

# ---------- UI helpers ----------


class SlateFrame(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg=COLORS['bg_panel'], highlightthickness=1,
                       highlightbackground=COLORS['border'], bd=0)


class SlateLabel(tk.Label):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg=COLORS['bg_panel'],
                       fg=COLORS['fg_primary'], font=('Segoe UI', 11))


class SlateTitle(tk.Label):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg=COLORS['bg_panel'], fg=COLORS['fg_primary'],
                       font=('Segoe UI Semibold', 14))


class SlateButton(tk.Button):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg=COLORS['accent_b'], fg='black', activebackground=COLORS['accent_a'], activeforeground='black',
                       font=('Segoe UI Semibold', 10), relief='flat', bd=0, padx=14, pady=8, cursor='hand2',
                       highlightthickness=1, highlightbackground=COLORS['border'])
        self.bind('<Enter>', lambda e: self.configure(bg=COLORS['accent_a']))
        self.bind('<Leave>', lambda e: self.configure(bg=COLORS['accent_b']))


class SlateText(tk.Text):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg=COLORS['bg_panel'], fg=COLORS['fg_primary'], insertbackground=COLORS['accent_a'],
                       selectbackground=COLORS['accent_b'], selectforeground='black',
                       font=('Consolas', 10), relief='flat', wrap='word', padx=10, pady=10,
                       highlightthickness=1, highlightbackground=COLORS['border'])


class SlateListbox(tk.Listbox):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg=COLORS['bg_panel'], fg=COLORS['fg_primary'], selectbackground=COLORS['accent_a'],
                       selectforeground='black', relief='flat', font=('Segoe UI', 10), highlightthickness=1,
                       highlightbackground=COLORS['border'])


class SlateCanvas(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg=COLORS['bg_panel'], relief='flat',
                       highlightthickness=1, highlightbackground=COLORS['border'])


def draw_banner(canvas: tk.Canvas, w: int, h: int):
    canvas.delete('all')
    step = 6
    for i in range(0, h, step):
        color = COLORS['accent_a'] if (
            i // step) % 2 == 0 else COLORS['accent_b']
        canvas.create_rectangle(0, i, w, i + step, fill=color, width=0)
    canvas.create_rectangle(0, 0, w, h, fill='black',
                            stipple='gray25', width=0)
    title = "artiFACTS"
    font_settings = ('Segoe UI Black', 44)
    for radius, col in [(3, 'gray40'), (2, 'gray30'), (1, 'gray20')]:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    canvas.create_text(
                        w // 2 + dx, h // 2 + dy, text=title, font=font_settings, fill=col)
    canvas.create_text(w // 2, h // 2, text=title,
                       font=font_settings, fill='white')

# ScrollFrame for long forms


class ScrollFrame(SlateFrame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(
            self, bg=COLORS['bg_panel'], highlightthickness=0)
        self.vbar = tk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.inner = SlateFrame(self.canvas)
        self.inner_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas)

    def _on_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel)
        widget.bind_all("<Button-4>", self._on_mousewheel_linux)
        widget.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

# ---------- Library Window ----------


class LibraryWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("artiFACTS Library")
        self.configure(bg=COLORS['bg_app'])
        try:
            self.state('zoomed')
        except tk.TclError:
            self.attributes('-zoomed', True)

        banner_frame = SlateFrame(self)
        banner_frame.pack(fill=tk.X, padx=12, pady=(12, 6))
        self.banner = tk.Canvas(banner_frame, height=64,
                                bg=COLORS['bg_panel'], highlightthickness=0)
        self.banner.pack(fill=tk.X)
        self.banner.bind('<Configure>', lambda e: draw_banner(
            self.banner, e.width, e.height))

        main = SlateFrame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        left = SlateFrame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        SlateTitle(left, text='Item Catalog').pack(anchor='w', pady=(6, 4))
        # --- Treeview + styling (merged) ---
        # Wrapper with scroll support
        tree_wrap = SlateFrame(left)
        tree_wrap.pack(fill=tk.BOTH, expand=True)

        # Style setup (codex branch had fuller styling + theme fallback)
        tree_style = ttk.Style()
        try:
            tree_style.theme_use('clam')
        except tk.TclError:
            pass

        tree_style.configure(
            'Library.Treeview',
            background=COLORS['bg_panel'],
            fieldbackground=COLORS['bg_panel'],
            foreground=COLORS['fg_primary'],
            bordercolor=COLORS.get('border', COLORS['bg_panel']),
            rowheight=20
        )
        tree_style.map(
            'Library.Treeview',
            background=[('selected', COLORS.get('accent_a', '#ccccff'))],
            foreground=[('selected', 'black')]
        )
        tree_style.layout('Library.Treeview', [
                          ('Treeview.treearea', {'sticky': 'nswe'})])

        # Treeview (in wrapper so the scrollbar can live beside it)
        self.tree = ttk.Treeview(
            tree_wrap, show='tree', style='Library.Treeview', selectmode='browse', height=20
        )
        self.tree.tag_configure('category', font=('Segoe UI', 10, 'bold'))
        self.tree.pack(side='left', fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        tree_scroll = tk.Scrollbar(
            tree_wrap, orient='vertical', command=self.tree.yview)
        tree_scroll.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=tree_scroll.set)

        # Selection handler
        self.tree.bind("<<TreeviewSelect>>", self.display_item_info)

        right = SlateFrame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        SlateTitle(right, text='Item Photos').pack(anchor='w', pady=(6, 6))

        img_frame = SlateFrame(right)
        img_frame.pack(fill=tk.X, pady=(0, 10))
        self.image_canvas = SlateCanvas(img_frame, height=320)
        self.image_canvas.pack(side='top', fill='x', expand=True)
        sx = tk.Scrollbar(img_frame, orient='horizontal',
                          command=self.image_canvas.xview)
        sx.pack(side='bottom', fill='x')
        self.image_canvas.configure(xscrollcommand=sx.set)
        self.image_canvas.bind(
            '<ButtonPress-1>', lambda e: self.image_canvas.scan_mark(e.x, e.y))
        self.image_canvas.bind(
            '<B1-Motion>', lambda e: self.image_canvas.scan_dragto(e.x, e.y, gain=1))

        # Expert opinion is now shown in a pop-up window, so no repositioning needed

        SlateTitle(right, text='Item Details').pack(anchor='w', pady=(6, 6))
        details_wrap = SlateFrame(right)
        details_wrap.pack(fill=tk.BOTH, expand=True)
        self.info_text = SlateText(details_wrap, height=14)
        self.info_text.pack(side='left', fill=tk.BOTH, expand=True)
        info_scroll = tk.Scrollbar(
            details_wrap, orient='vertical', command=self.info_text.yview)
        info_scroll.pack(side='right', fill='y')
        self.info_text.configure(yscrollcommand=info_scroll.set)

        btn_row = SlateFrame(right)
        btn_row.pack(anchor='e', pady=(6, 0))
        self.edit_btn = SlateButton(
            btn_row, text='Edit Item', command=self.edit_item, state='disabled')
        self.edit_btn.pack(side='right')
        self.blurb_btn = SlateButton(
            btn_row, text='Hide Expert Opinion', command=self.toggle_blurb, state='disabled'
        )
        self.blurb_btn.pack(side='right', padx=(0, 6))
        self.blurb_canvas = None
        self._blurb_popup = None
        self.current_blurb_text = ''
        self.blurb_visible = False
        self.blurb_enabled = True
        self._blurb_item_id = None  # track which item the current blurb was generated for
        self._blurb_img_ref = None   # holds the Pillow-rendered background

        self.photo_refs = []
        self._img_populating = set()
        self.item_rows = {}
        self._current_item = None
        self.load_items()

    def load_items(self):
        for child in self.tree.get_children():
            self.tree.delete(child)
        self.item_rows = {}
        for cat in LIBRARY_CATEGORIES:
            self.tree.insert('', 'end', iid=f'cat_{cat}', text=cat.title(),
                             open=False, tags=('category',))
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name, category FROM items ORDER BY name")
        for item_id, name, category in cur.fetchall():
            cat = (category or 'other').lower()
            if cat not in LIBRARY_CATEGORIES:
                cat = 'other'
            self.tree.insert(f'cat_{cat}', 'end', iid=str(item_id), text=name)
            self.item_rows[item_id] = (item_id, name, cat)
        conn.close()

    def display_item_info(self, event):
        sel = self.tree.selection()
        if not sel:
            self.edit_btn.config(state='disabled')
            self.hide_blurb()
            self.blurb_btn.config(state='disabled')
            self._current_item = None
            return
        iid = sel[0]
        if iid.startswith('cat_'):
            self.edit_btn.config(state='disabled')
            self.hide_blurb()
            self.blurb_btn.config(state='disabled')
            self._current_item = None
            return
        item_id = int(iid)
        item_id, name, category = self.item_rows.get(
            item_id, (None, None, None))
        if item_id is None:
            return
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM items WHERE id = ?", (item_id,))
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
        photo_path = ""
        if row:
            category = row[cols.index("category")] or ""
            photo_path = row[cols.index("photo_path")] or ""

        detail_map = {}
        if item_id is not None:
            cur.execute(
                "SELECT key, value FROM item_details WHERE item_id=?", (item_id,))
            detail_map = {k: v for k, v in cur.fetchall()}
        conn.close()

        self._current_item = {
            "id": item_id,
            "row": row,
            "cols": cols,
            "details": detail_map,
            "name": name,
            "category": category,
        }
        self.edit_btn.config(state='normal')
        self.hide_blurb()
        self.blurb_btn.config(state='normal')
        self.blurb_visible = self.blurb_enabled
        self.blurb_btn.config(
            text='Hide Expert Opinion' if self.blurb_enabled else 'Show Expert Opinion'
        )
        first_photo = (detail_map.get("img1_path") or detail_map.get(
            "upload_path") or photo_path or "")
        self.current_blurb_text = get_expert_blurb(
            category, name, detail_map, first_photo)
        self._blurb_item_id = item_id
        if self.current_blurb_text and (detail_map.get("expert_blurb", "") != self.current_blurb_text):
            save_blurb_to_db(item_id, self.current_blurb_text)

        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        if row:
            schema = SCHEMAS.get(category.lower(), SCHEMAS["other"])
            order = list(dict.fromkeys(schema["user"] + schema["api"]))
            seen = set()
            for i, val in enumerate(row):
                col = cols[i]
                if col in ("id", "added_on", "photo_path", "category"):
                    continue
                if val:
                    label = col.replace('_', ' ').capitalize()
                    self.info_text.insert(tk.END, f"• {label}: {val}\n\n")
                    seen.add(col)
            for key in order:
                if key in seen or key in HIDDEN_META_KEYS or key == "category":
                    continue
                val = detail_map.get(key)
                if val:
                    label = key.replace('_', ' ').capitalize()
                    self.info_text.insert(tk.END, f"• {label}: {val}\n\n")
                    seen.add(key)
        else:
            self.info_text.insert(tk.END, "Item not found.")
        self.info_text.config(state='disabled')

        n = PHOTO_SLOTS.get(category.lower(), 4)
        paths = []
        for i in range(1, n+1):
            p = detail_map.get(f"img{i}_path")
            paths.append(p if p and os.path.exists(p) else None)

        if any(p is None for p in paths) and item_id is not None:
            # prevent parallel/looped populates
            if item_id in self._img_populating:
                return
            self._img_populating.add(item_id)

            def _bg():
                try:
                    item_dir = os.path.join(
                        PHOTO_DIR, (category or "").lower(), str(item_id))
                    uploaded = photo_path if (
                        photo_path and os.path.commonpath([
                            os.path.abspath(photo_path), os.path.abspath(item_dir)])
                        == os.path.abspath(item_dir)) else ""
                    meta = populate_photo_slots(
                        item_id, category, name, detail_map, uploaded)
                    if meta:
                        con = sqlite3.connect(DB_PATH)
                        cur = con.cursor()
                        cur.execute("PRAGMA table_info(items)")
                        cols_set = {r[1] for r in cur.fetchall()}
                        for k, v in meta.items():
                            if k in cols_set:
                                cur.execute(f"UPDATE items SET {k}=? WHERE id=?",
                                            (str(v) if v is not None else "", item_id))
                            cur.execute("""INSERT INTO item_details(item_id, key, value)
                                           VALUES(?,?,?)
                                           ON CONFLICT(item_id, key) DO UPDATE SET value=excluded.value""",
                                        (item_id, k, str(v) if v is not None else ""))
                        con.commit()
                        con.close()
                finally:
                    # clear the in-flight flag and re-render once
                    self._img_populating.discard(item_id)
                    self.after(0, lambda: self.display_item_info(None))
            threading.Thread(target=_bg, daemon=True).start()

        self.image_canvas.delete('all')
        self.photo_refs.clear()
        for i in range(n):
            x = 16 + i*312
            p = paths[i]
            if p:
                try:
                    im = Image.open(p)
                    im.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    tk_im = ImageTk.PhotoImage(im)
                    self.image_canvas.create_image(
                        x, 10, anchor='nw', image=tk_im)
                    self.photo_refs.append(tk_im)
                except Exception:
                    self.image_canvas.create_rectangle(
                        x, 10, x+300, 310, outline='#555', fill='#222')
            else:
                self.image_canvas.create_rectangle(
                    x, 10, x+300, 310, outline='#555', fill='#222')
        self.image_canvas.config(scrollregion=self.image_canvas.bbox('all'))
        if self.blurb_visible:
            self._draw_blurb(self.current_blurb_text)
            self.blurb_btn.config(text='Hide Expert Opinion')
        else:
            self.hide_blurb()

    def _round_rect(self, canvas, x1, y1, x2, y2, r=10, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _draw_blurb(self, text):
        """Render the expert blurb inside a dedicated pop-up window."""
        self.hide_blurb()
        if not text:
            return
        try:
            popup = tk.Toplevel(self)
            popup.title("Expert Opinion")
            popup.configure(bg=COLORS['bg_app'])
            popup.resizable(False, False)
            popup.transient(self)
            popup.protocol("WM_DELETE_WINDOW", self.toggle_blurb)

            frame = SlateFrame(popup)
            frame.pack(padx=10, pady=10)

            # --- simple expert silhouette ---
            expert = tk.Canvas(frame, width=60, height=60,
                               bg=frame.cget('bg'), highlightthickness=0)
            expert.pack(side='left', padx=(0, 10))
            fg = COLORS['blurb_fg']
            expert.create_oval(15, 0, 45, 30, fill=fg, outline=fg)
            expert.create_rectangle(18, 30, 42, 58, fill=fg, outline=fg)

            # --- speech bubble with text ---
            bubble = tk.Canvas(frame, bg=frame.cget('bg'),
                               highlightthickness=0, borderwidth=0)
            bubble.pack(side='left')

            pad = 10
            max_w = BLURB_MAX_W
            text_id = bubble.create_text(
                0, 0, text=text, width=max_w,
                anchor='nw', font=BLURB_FONT, fill=COLORS['blurb_fg']
            )
            bubble.update_idletasks()
            tb = bubble.bbox(text_id) or (0, 0, pad*2 + 40, pad*2 + 20)
            txt_w = max(tb[2] - tb[0], 40)
            txt_h = max(tb[3] - tb[1], 20)
            bubble_w = txt_w + pad * 2
            bubble_h = txt_h + pad * 2
            tail_w = 12
            W = bubble_w + tail_w
            H = bubble_h

            img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            if BLURB_SHADOW_BLUR > 0:
                sh = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                sd = ImageDraw.Draw(sh)
                sd.rounded_rectangle(
                    [tail_w + 2, 2, W - 2, H - 2], radius=BLURB_RADIUS,
                    fill=(0, 0, 0, 110)
                )
                sd.polygon(
                    [(tail_w, H // 2 - 10), (2, H // 2), (tail_w, H // 2 + 10)],
                    fill=(0, 0, 0, 110)
                )
                sh = sh.filter(ImageFilter.GaussianBlur(BLURB_SHADOW_BLUR))
                img.alpha_composite(sh)

            draw.rounded_rectangle(
                [tail_w, 0, W - 1, H - 1], radius=BLURB_RADIUS,
                fill=(255, 255, 255, BLURB_ALPHA)
            )
            draw.polygon(
                [(tail_w, H // 2 - 10), (0, H // 2), (tail_w, H // 2 + 10)],
                fill=(255, 255, 255, BLURB_ALPHA)
            )

            tk_bg = ImageTk.PhotoImage(img)
            self._blurb_img_ref = tk_bg
            bubble.config(width=W, height=H)
            bubble.create_image(0, 0, image=tk_bg, anchor='nw')
            bubble.coords(text_id, tail_w + pad, pad)
            bubble.tag_raise(text_id)

            close_id = bubble.create_text(
                W - 6, 6, text='×', anchor='ne',
                font=('Segoe UI', 10, 'bold'), fill='#333333'
            )
            bubble.tag_bind(close_id, '<Button-1>',
                            lambda e: self.toggle_blurb())

            popup.update_idletasks()
            px = self.winfo_rootx() + (self.winfo_width() - popup.winfo_width()) // 2
            py = self.winfo_rooty() + (self.winfo_height() - popup.winfo_height()) // 2
            popup.geometry(f"+{px}+{py}")

            self.blurb_canvas = bubble
            self._blurb_popup = popup
            self.blurb_btn.config(text='Hide Expert Opinion')
        except Exception:
            return

    def hide_blurb(self):
        if self._blurb_popup is not None:
            try:
                self._blurb_popup.destroy()
            except Exception:
                pass
            self._blurb_popup = None
        if self.blurb_canvas is not None:
            try:
                self.blurb_canvas.destroy()
            except Exception:
                pass
            self.blurb_canvas = None
        self._blurb_img_ref = None
        self.blurb_visible = False
        self.blurb_btn.config(text='Show Expert Opinion')

    def toggle_blurb(self):
        print(
            f"DEBUG: toggle_blurb called, blurb_canvas: {self.blurb_canvas}, _current_item: {self._current_item}")

        # If visible, hide it.
        if self.blurb_canvas:
            print("DEBUG: Hiding existing blurb")
            self.blurb_enabled = False
            self.hide_blurb()
            return

        # If hidden, (re)generate text if needed and draw.
        if self._current_item:
            self.blurb_enabled = True
            current_id = self._current_item.get('id')
            if current_id != self._blurb_item_id or not self.current_blurb_text:
                data = self._current_item
                print(
                    f"DEBUG: Generating blurb for item: {data.get('name')} ({data.get('category')})")
                self.current_blurb_text = get_expert_blurb(
                    data.get('category'), data.get(
                        'name'), data.get('details') or {},
                    (data.get('details') or {}).get('img1_path') or (data.get('row') and data.get('row')[
                        data.get('cols').index("photo_path")] if data.get('cols') and data.get('row') else "")
                )
                save_blurb_to_db(self._blurb_item_id, self.current_blurb_text)
                print(
                    f"DEBUG: Generated blurb text: {self.current_blurb_text}")
            else:
                print(
                    f"DEBUG: Using existing blurb text: {self.current_blurb_text}")

            # Only draw if the selection hasn't changed mid-operation
            if self._current_item and self._current_item.get('id') == self._blurb_item_id:
                print("DEBUG: Drawing blurb...")
                self._draw_blurb(self.current_blurb_text)
                self.blurb_visible = True
                self.blurb_btn.config(text='Hide Expert Opinion')
            else:
                print("DEBUG: Item changed before drawing; aborting blurb display")
        else:
            print("DEBUG: No current item selected")

    def edit_item(self):
        data = self._current_item
        if not data:
            return
        item_id = data.get("id")
        row = data.get("row") or []
        cols = data.get("cols") or []
        detail_map = data.get("details") or {}
        name = row[cols.index("name")] if row and "name" in cols else ""
        category = row[cols.index(
            "category")] if row and "category" in cols else ""
        photo_path = row[cols.index("photo_path")
                         ] if row and "photo_path" in cols else ""
        initial_details = {}
        if row:
            for i, col in enumerate(cols):
                if col in ("id", "added_on", "photo_path", "category", "name"):
                    continue
                val = row[i]
                if val:
                    initial_details[col] = val
        initial_details.update(detail_map)

        def _reload():
            sel = str(item_id)
            self.load_items()
            if self.tree.exists(sel):
                self.tree.selection_set(sel)
                self.display_item_info(None)

        ItemDetailWindow(self, initial_name=name, category=category,
                         photo_path=photo_path, item_id=item_id,
                         initial_details=initial_details, on_save=_reload)


# ---- DDG image search ----
try:
    from ddgs import DDGS   # modern library
    _DDG_OK = True
except Exception:
    _DDG_OK = False


class SearchError(Exception):
    """Base class for image search failures."""


class SearchRateLimited(SearchError):
    """Raised when the provider rate limits requests."""


def _build_reference_query(category: str, term: str) -> str:
    c = (category or "").lower().strip()
    t = _decontainerize_label(term or "").strip()
    # tokens we may add
    need = []

    def want(word: str) -> None:
        if word and word.lower() not in t.lower():
            need.append(word)

    if c == "vinyl":
        want("album cover")
        want("front")
        want("high resolution")
    elif c == "coin":
        want("coin")
        want("obverse")
        want("reverse")
    elif c == "fossil":
        want("fossil")
        want("specimen")
    elif c == "shell":
        # accept "shell" OR "seashell" already present
        if not re.search(r"\b(sea)?shell(s)?\b", t, flags=re.I):
            need.append("seashell")
        want("specimen")
    elif c == "mineral":
        want("mineral")
        want("specimen")
    elif c == "zoological":
        want("specimen")
        want("dorsal")

    return " ".join([t] + need) if need else t


# --- DDG session + throttled image search ---

_DDG_LOCK = threading.Lock()
_DDG_SESSION = None
_DDG_CACHE = {}
_DDG_CACHE_TTL = 600.0
_last_ddg_t = 0.0
_MIN_INTERVAL = 2.5
_MAX_TRIES = 4
_ddg_rl_count = 0
_RL_THRESHOLD = 4
_ddg_pause_until = 0.0
_RL_PAUSE = 90.0            # pause duration on repeated rate limits


def _ddg_session():
    global _DDG_SESSION
    if _DDG_SESSION is None:
        _DDG_SESSION = DDGS()  # persistent session; avoids new vqd each call
    return _DDG_SESSION


def ddg_image_search(term: str, max_results: int = 8):
    global _last_ddg_t, _ddg_rl_count, _ddg_pause_until
    if not _DDG_OK:
        return []
    term = (term or "").strip()
    if not term:
        return []

    now = time.monotonic()
    if now < _ddg_pause_until:
        raise SearchRateLimited("paused")

    key = (term, int(max_results))
    ts_res = _DDG_CACHE.get(key)
    if ts_res and (now - ts_res[0] < _DDG_CACHE_TTL):
        return ts_res[1]

    last_err = None

    with _DDG_LOCK:  # serialize all DDG calls across the app
        wait = max(0.0, _MIN_INTERVAL - (time.monotonic() - _last_ddg_t))
        if wait:
            print(f"[ddg] q={term!r} wait={wait:.2f}s")
            time.sleep(wait)

        tries = 0
        while tries < _MAX_TRIES:
            try:
                print(f"[ddg] q={term!r} try={tries+1}")
                sess = _ddg_session()
                results = list(sess.images(
                    term,                         # positional (best)
                    max_results=max_results,
                    safesearch="moderate",
                    region="wt-wt",               # neutral region is safest
                ))
                _last_ddg_t = time.monotonic()
                out = [(r.get("image"), r.get("url"))
                       for r in results if r.get("image") and r.get("url")]
                print(f"[ddg] q={term!r} try={tries+1} results={len(out)}")
                _DDG_CACHE[key] = (_last_ddg_t, out)
                _ddg_rl_count = 0
                return out
            except Exception as e:
                tries += 1
                last_err = e
                msg = str(e).lower()
                is_rl = "403" in msg or "rate" in msg
                if is_rl:
                    _ddg_rl_count += 1
                    marker = "rate-limit"
                else:
                    _ddg_rl_count = 0
                    marker = "error"
                backoff = (2 ** tries) + random.random()
                print(
                    f"[ddg] q={term!r} try={tries} {marker} sleep={backoff:.2f}s")
                time.sleep(backoff)
                if _ddg_rl_count >= _RL_THRESHOLD:
                    _ddg_pause_until = time.monotonic() + _RL_PAUSE
                    print(f"[ddg] circuit break {_RL_PAUSE}s")
                    break

    # attempt stale cache on failure
    if key in _DDG_CACHE:
        return _DDG_CACHE[key][1]

    if isinstance(last_err, Exception) and ("403" in str(last_err).lower() or _ddg_rl_count > 0):
        raise SearchRateLimited(str(last_err))
    raise SearchError(str(last_err))
# ---- Composite image search: DDG first, Wikimedia fallback ----


def wikimedia_image_search(query: str, max_results: int = 8):
    """
    Super-simple Wikimedia Commons image lookup.
    Returns [(image_url, page_url), ...]
    """
    try:
        import requests
        # Keep it light and generic: search commons file namespace
        u = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,           # File:
            "gsrlimit": max(1, min(20, max_results * 2)),
            "prop": "imageinfo|info",
            "iiprop": "url",
            "inprop": "url",
            "format": "json",
        }
        r = requests.get(u, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        pages = (j.get("query", {}).get("pages", {}) or {}).values()
        out = []
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            img = ii.get("url") or ""
            pageurl = p.get("fullurl") or ""
            if img:
                out.append((img, pageurl))
                if len(out) >= max_results:
                    break
        return out
    except Exception:
        return []


def image_search(query: str, max_results: int = 8):
    """
    Preferred entry point for reference images.
    1) ddg_image_search
    2) wikimedia_image_search (fallback)
    Raises SearchRateLimited/SearchError only if *both* fail.
    """
    try:
        res = ddg_image_search(query, max_results=max_results)
        if res:
            return res
    except (SearchRateLimited, SearchError):
        pass

    # Fallback
    res2 = wikimedia_image_search(query, max_results=max_results)
    if res2:
        return res2

    # Surface a consistent "no results" signal
    raise SearchError("No results found.")


def _download_image(url: str, dest_path: str, referer: str | None = None) -> bool:
    """Download an image to dest_path. Returns True on success."""
    try:
        ua = (
            _discogs_user_agent()
            if "discogs.com" in (url or "")
            else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        headers = {
            "User-Agent": ua,
            "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        }
        if referer:
            headers["Referer"] = referer

        r = requests.get(url, headers=headers, timeout=20,
                         stream=True, allow_redirects=True)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
            return True

        # Fallback via DuckDuckGo’s hotlink-safe proxy
        from urllib.parse import quote
        proxy = f"https://external-content.duckduckgo.com/iu/?u={quote(url, safe='')}&f=1"
        r2 = requests.get(proxy, headers=headers, timeout=20,
                          stream=True, allow_redirects=True)
        if r2.status_code == 200 and r2.headers.get("content-type", "").startswith("image/"):
            with open(dest_path, "wb") as f:
                for chunk in r2.iter_content(65536):
                    if chunk:
                        f.write(chunk)
            return True
        return False
    except Exception:
        return False


def populate_photo_slots(item_id: int, category: str, name: str, details: dict, uploaded: str) -> dict:
    cat = (category or "").lower()
    item_dir = os.path.join(PHOTO_DIR, cat, str(item_id))
    os.makedirs(item_dir, exist_ok=True)
    meta = {}

    slot = 1
    max_slots = PHOTO_SLOTS.get(cat, 4)

    def _norm_url(u: str) -> str:
        try:
            base = (u or "").split("?")[0].strip().lower()
            # strip common tracking suffixes
            for tail in (".jpeg", ".jpg", ".png", ".webp"):
                if base.endswith(tail):
                    return base
            return base
        except Exception:
            return (u or "").strip().lower()

    seen_urls = set()
    saved_urls = set()

    def _save(url: str) -> bool:
        nonlocal slot
        if slot > max_slots or not url:
            return False
        nu = _norm_url(url)
        if nu in saved_urls:
            return False
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"

        # find the next unused slot/file (keeps existing files intact,
        # regardless of extension already used)
        while slot <= max_slots:
            pattern = os.path.join(item_dir, f"img{slot}.*")
            if not glob.glob(pattern):
                dest = os.path.join(item_dir, f"img{slot}{ext}")
                break
            slot += 1
        else:
            return False

        ok = _download_image(url, dest)
        seen_urls.add(nu)
        if ok:
            meta[f"img{slot}_path"] = dest
            meta[f"img{slot}_src"] = url
            saved_urls.add(nu)
            slot += 1
        return ok

    # copy uploaded image for persistence
    if uploaded and os.path.exists(uploaded):
        up_ext = os.path.splitext(uploaded)[1] or ".jpg"
        up_dest = os.path.join(item_dir, f"upload{up_ext}")
        try:
            if os.path.abspath(uploaded) != os.path.abspath(up_dest):
                shutil.copyfile(uploaded, up_dest)
            meta["upload_path"] = up_dest
        except Exception:
            pass

        # promote the local copy to img1, replacing any existing first slot
        if cat in {"fossil", "shell", "mineral", "zoological"}:
            dest_first = os.path.join(item_dir, f"img1{up_ext}")
            try:
                for f in glob.glob(os.path.join(item_dir, "img1.*")):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                shutil.copyfile(up_dest, dest_first)
                meta["img1_path"] = dest_first
                meta["img1_src"] = "uploaded"
            except Exception:
                pass
            # next web save goes to slot 2
            slot = 2

    q_name = name or ""

    if cat == "vinyl":
        rel_id = details.get("discogs_release_id") or ""
        urls = []
        if rel_id:
            try:
                rel = discogs_get_release(int(rel_id))
                for im in (rel.get("images") or [])[:max_slots]:
                    u = im.get("uri") or im.get("resource_url")
                    if u:
                        urls.append(u)
            except Exception:
                urls = []

        discogs_ok = []
        for u in urls[:max_slots]:
            discogs_ok.append(_save(u))

        needed = max_slots - sum(1 for ok in discogs_ok if ok)
        if needed > 0:
            album = details.get("release_title", "")
            artist = details.get("artist", "")
            queries = [f"{album} – {artist}"] + [artist] * (max_slots - 1)
            for q in queries:
                if needed <= 0:
                    break
                q = q.strip()
                res = image_search(_build_reference_query(
                    cat, q), max_results=1) if q else []
                if _save(res[0][0] if res else ""):
                    needed -= 1
    elif cat == "coin":
        queries = [q_name, q_name]
        for q in queries:
            q = q.strip()
            res = image_search(_build_reference_query(
                cat, q), max_results=1) if q else []
            _save(res[0][0] if res else "")
    elif cat in {"fossil", "shell", "mineral"}:
        # Build a few varied queries to diversify results
        q1 = q_name
        q2 = f"{q_name} specimen"
        q3 = f"{q_name} macro"
        q4 = f"{q_name} museum"
        need = max_slots - (slot - 1)
        all_cands = []

        for q in (q1, q2, q3, q4):
            if need <= 0:
                break
            qq = q.strip()
            if not qq:
                continue
            # ask for several candidates at once
            res = image_search(
                _build_reference_query(cat, qq),
                max_results=min(8, need * 4),
            )
            for (img_url, _page) in res:
                nu = _norm_url(img_url)
                if nu and nu not in seen_urls:
                    all_cands.append(img_url)
                    seen_urls.add(nu)
                    if len(all_cands) >= need:
                        break
            if len(all_cands) >= need:
                break

        for url in all_cands:
            if need <= 0:
                break
            if _save(url):
                need -= 1
    elif cat == "zoological":
        q1 = q_name
        q2 = f"{q_name} specimen"
        q3 = f"{q_name} closeup"
        q4 = f"{q_name} museum"
        need = max_slots - (slot - 1)
        all_cands = []
        for q in (q1, q2, q3, q4):
            if need <= 0:
                break
            qq = q.strip()
            if not qq:
                continue
            res = image_search(
                _build_reference_query(cat, qq),
                max_results=min(12, need * 4),
            )
            for (img_url, _page) in res:
                nu = _norm_url(img_url)
                if nu and nu not in seen_urls:
                    all_cands.append(img_url)
                    seen_urls.add(nu)
                    if len(all_cands) >= need:
                        break
            if len(all_cands) >= need:
                break
        for url in all_cands:
            if need <= 0:
                break
            if _save(url):
                need -= 1

    return meta


# ===================== Upload & Review Dialogs =====================

class UploadDialog(tk.Toplevel):
    """
    Step 1: pick/capture images, basic details.
    Hands off to ReviewDialog via a simple dict payload.
    """
    MAX_IMAGES = 4

    def __init__(self, master, on_continue):
        super().__init__(master)
        self.title("Upload Artifact")
        self.configure(bg=COLORS.get('bg_panel', '#111827'))
        self.on_continue = on_continue
        self.images = []   # list of (PIL.Image, temp_path) for later save
        self.tempfiles = []
        self.cap = None    # OpenCV camera handle

        # --- Top: image pick / capture ---
        frm = tk.Frame(self, bg=COLORS['bg_panel'])
        frm.pack(fill='both', expand=True, padx=12, pady=12)

        left = tk.Frame(frm, bg=COLORS['bg_panel'])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(frm, bg=COLORS['bg_panel'])
        right.grid(row=0, column=1, sticky="nsew")

        for i in range(2):
            frm.grid_columnconfigure(i, weight=1)

        # File picker
        tk.Label(left, text="Upload photos (1–4)",
                 fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).pack(anchor='w')
        tk.Button(left, text="Choose images...", command=self.pick_files).pack(
            anchor='w', pady=(4, 10))
        self.preview = tk.Frame(left, bg=COLORS['bg_panel'])
        self.preview.pack(fill='x')

        # Camera capture
        tk.Label(right, text="Camera",
                 fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).pack(anchor='w')
        tk.Button(right, text="Capture from camera...",
                  command=self._capture_from_ident_camera).pack(
                      anchor='w', pady=(4, 10))

        # --- Details ---
        tk.Label(frm, text="Details", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(12, 4))
        det = tk.Frame(frm, bg=COLORS['bg_panel'])
        det.grid(row=2, column=0, columnspan=2, sticky='ew')
        for i in range(2):
            det.grid_columnconfigure(i, weight=1)

        tk.Label(det, text="Category", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).grid(
            row=0, column=0, sticky='w')
        self.category_var = tk.StringVar(value="minerals")
        cats = ["minerals", "fossils", "shells",
                "coins", "vinyl", "zoological", "other"]
        self.category = ttk.Combobox(
            det, values=cats, textvariable=self.category_var, state="readonly")
        self.category.grid(row=1, column=0, sticky='ew', pady=(0, 8))

        tk.Label(det, text="Title", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).grid(
            row=0, column=1, sticky='w')
        self.title_var = tk.StringVar()
        tk.Entry(det, textvariable=self.title_var).grid(
            row=1, column=1, sticky='ew', pady=(0, 8))

        # --- Actions ---
        act = tk.Frame(self, bg=COLORS['bg_panel'])
        act.pack(fill='x', pady=(10, 6))
        tk.Button(act, text="Cancel", command=self.destroy).pack(side='left')
        tk.Button(act, text="Review →",
                  command=self.go_review).pack(side='right')

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.stop_cam()
        self.destroy()

    def pick_files(self):
        if len(self.images) >= self.MAX_IMAGES:
            messagebox.showinfo("Limit", "Max 4 photos.")
            return
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        for p in paths:
            if len(self.images) >= self.MAX_IMAGES:
                break
            try:
                im = Image.open(p).convert("RGB")
                self.images.append((im, p))
            except Exception:
                pass
        self.render_preview()

    def render_preview(self):
        for w in list(self.preview.children.values()):
            w.destroy()
        for idx, (im, src) in enumerate(self.images):
            thumb = im.copy()
            thumb.thumbnail((140, 140))
            tkimg = ImageTk.PhotoImage(thumb)
            lbl = tk.Label(self.preview, image=tkimg, bg=COLORS['bg_panel'])
            lbl.image = tkimg
            lbl.grid(row=0, column=idx, padx=4, pady=4)
            rm = tk.Button(self.preview, text="×",
                           command=partial(self._remove, idx))
            rm.grid(row=1, column=idx)

    def _remove(self, idx):
        try:
            self.images.pop(idx)
            self.render_preview()
        except Exception:
            pass

    # ---- Camera ----
    def _capture_from_ident_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith(
            "win") else cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            messagebox.showerror("Camera", "Cannot open camera.")
            return
        cv2.namedWindow('Press SPACE to capture', cv2.WINDOW_NORMAL)
        frame = None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imshow('Press SPACE to capture', img)
            key = cv2.waitKey(1)
            if key == 32:  # SPACE
                frame = img.copy()
                break
            elif key == 27:  # ESC
                break
        cap.release()
        cv2.destroyAllWindows()
        if frame is not None and len(self.images) < self.MAX_IMAGES:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self.images.append((pil_img, "<camera>"))
            self.render_preview()

    def stop_cam(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def go_review(self):
        if not self.images:
            messagebox.showinfo("Photos", "Add at least one photo.")
            return
        payload = {
            "images": self.images,  # list of (PIL.Image, src)
            "category": (self.category_var.get() or "other").strip().lower(),
            "title": self.title_var.get().strip(),
            "notes": self.notes.get("1.0", "end").strip()
        }
        self.stop_cam()
        self.withdraw()
        ReviewDialog(self, payload, on_done=self._done)

    def _done(self, saved_id=None):
        # close both on finish
        try:
            self.destroy()
        except Exception:
            pass


class ReviewDialog(tk.Toplevel):
    """
    Step 2: review thumbnails, optional enrichment, Save to library.
    """

    def __init__(self, master, payload, on_done=None):
        super().__init__(master)
        self.title("Review & Save")
        self.configure(bg=COLORS.get('bg_panel', '#111827'))
        self.payload = payload
        self.on_done = on_done

        # Thumbs
        tk.Label(self, text="Images", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).pack(
            anchor='w', padx=12, pady=(12, 4))
        preview = tk.Frame(self, bg=COLORS['bg_panel'])
        preview.pack(fill='x', padx=12)
        for idx, (im, src) in enumerate(payload["images"]):
            th = im.copy()
            th.thumbnail((220, 220))
            tkimg = ImageTk.PhotoImage(th)
            lbl = tk.Label(preview, image=tkimg, bg=COLORS['bg_panel'])
            lbl.image = tkimg
            lbl.grid(row=0, column=idx, padx=4, pady=4)

        # Details (editable)
        frm = tk.Frame(self, bg=COLORS['bg_panel'])
        frm.pack(fill='x', padx=12, pady=10)
        for i in range(2):
            frm.grid_columnconfigure(i, weight=1)

        tk.Label(frm, text="Category", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).grid(
            row=0, column=0, sticky='w')
        self.category_var = tk.StringVar(
            value=payload.get("category", "other"))
        self.category = ttk.Combobox(frm, values=["minerals", "fossils", "shells", "coins",
                                     "vinyl", "zoological", "other"], textvariable=self.category_var, state="readonly")
        self.category.grid(row=1, column=0, sticky='ew', pady=(0, 8))

        tk.Label(frm, text="Title", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).grid(
            row=0, column=1, sticky='w')
        self.title_var = tk.StringVar(value=payload.get("title", ""))
        tk.Entry(frm, textvariable=self.title_var).grid(
            row=1, column=1, sticky='ew', pady=(0, 8))

        tk.Label(frm, text="Notes", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).grid(
            row=2, column=0, columnspan=2, sticky='w')
        self.notes = tk.Text(frm, height=4)
        self.notes.grid(row=3, column=0, columnspan=2, sticky='ew')
        self.notes.insert("1.0", payload.get("notes", ""))

        # Optional enrichment output
        tk.Label(self, text="Enrichment (optional)", fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).pack(
            anchor='w', padx=12, pady=(10, 4))
        enr = tk.Frame(self, bg=COLORS['bg_panel'])
        enr.pack(fill='x', padx=12)
        tk.Button(enr, text="Suggest Identification",
                  command=self.suggest_ident).pack(side='left')
        tk.Button(enr, text="Suggest Traits",
                  command=self.suggest_traits).pack(side='left', padx=6)
        self.ai_out = tk.Text(self, height=6)
        self.ai_out.pack(fill='x', padx=12, pady=6)

        # Actions
        act = tk.Frame(self, bg=COLORS['bg_panel'])
        act.pack(fill='x', padx=12, pady=10)
        tk.Button(act, text="← Back", command=self._back).pack(side='left')
        tk.Button(act, text="Save to Library",
                  command=self.save).pack(side='right')

    def _back(self):
        self.destroy()
        self.master.deiconify()

    def _first_image_path(self):
        # ensure we have a temp path for vision helpers
        im, src = self.payload["images"][0]
        if src != "<camera>" and os.path.exists(src):
            return src
        tmp = os.path.join(tempfile.gettempdir(),
                           f"upload_{int(time.time())}.jpg")
        im.save(tmp, "JPEG", quality=90)
        return tmp

    def suggest_ident(self):
        self.ai_out.delete("1.0", "end")
        self.ai_out.insert("end", "Identifying…\n")
        try:
            p = self._first_image_path()
            cat = (self.category_var.get() or "other").lower()
            # Use your existing classifier stub (returns {'openai': {'guesses': [...]}})
            res = run_classifier(p, cat)
            guesses = (res.get('openai') or {}).get('guesses') or []
            # show top 3 with rationales if present
            for g in guesses[:3]:
                lbl = g.get('label', '').strip()
                conf = g.get('confidence', '')
                why = g.get('why', '') or g.get('explanation', '')
                self.ai_out.insert("end", f"• {lbl}  ({conf})\n")
                if why:
                    self.ai_out.insert("end", f"   – {why}\n")
            if not guesses:
                self.ai_out.insert("end", "No suggestions.\n")
        except Exception as e:
            self.ai_out.insert("end", f"Failed: {e}\n")

    def suggest_traits(self):
        self.ai_out.insert("end", "Suggesting traits…\n")
        # Placeholder hook; you can wire this to your existing enrich_* by category.
        # Keeping text-only so it’s safe if OpenAI client is not configured.
        cat = (self.category_var.get() or "other").lower()
        tips = {
            "minerals": "Typical fields: hardness, luster, cleavage, streak.",
            "fossils": "Typical fields: period, fossil type (ammonite/trilobite/shark tooth), preservation.",
            "shells": "Typical fields: group (whelk/cone/cowrie), patterning, aperture/lip, spire.",
            "zoological": "Typical fields: element (tooth/bone/antler/feather/insect), family/order hints.",
            "coins": "Year, mint marks, composition; avoid grading unless confident.",
            "vinyl": "Artist, title, label, year, RPM/size, catalog no."
        }.get(cat, "Add whatever is true and useful without guessing.")
        self.ai_out.insert("end", f"{tips}\n")

    def save(self):
        cat = (self.category_var.get() or "other").lower()
        title = self.title_var.get().strip() or "Untitled artifact"
        notes = self.notes.get("1.0", "end").strip()

        # Ensure photo dir
        os.makedirs(PHOTO_DIR, exist_ok=True)

        # Insert into DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, category TEXT, notes TEXT,
                created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER, slot INTEGER, path TEXT,
                FOREIGN KEY(item_id) REFERENCES items(id)
            )
        """)
        cur.execute("INSERT INTO items(title, category, notes, created_at) VALUES (?,?,?,?)",
                    (title, cat, notes, datetime.utcnow().isoformat()))
        item_id = cur.lastrowid

        # Save up to 4 images as cover-first
        max_slots = PHOTO_SLOTS.get(cat, 4)
        for slot, (im, src) in enumerate(self.payload["images"][:max_slots], start=1):
            out_dir = os.path.join(PHOTO_DIR, f"{item_id:06d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{slot}.jpg")
            im.save(out_path, "JPEG", quality=90)
            cur.execute("INSERT INTO photos(item_id, slot, path) VALUES (?,?,?)",
                        (item_id, slot, out_path))

        conn.commit()
        conn.close()
        messagebox.showinfo("Saved", f"Saved item #{item_id}")
        if self.on_done:
            self.on_done(saved_id=item_id)
        try:
            self.destroy()
        except Exception:
            pass


# ===================== Front Hub & Carousel =====================

def _get_recent_photo_cards(limit=None):
    """
    Returns [{'item_id': int, 'title': str, 'path': str}, ...] for ALL items
    (or the first N if limit is not None). Uses item_details.img1_path/upload_path
    with photo_path as a fallback.
    """
    rows = []
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        sql = """
            SELECT i.id,
                   i.name,
                   i.photo_path,
                   MAX(CASE WHEN d.key='img1_path'   THEN d.value END) AS img1_path,
                   MAX(CASE WHEN d.key='upload_path' THEN d.value END) AS upload_path
            FROM items i
            LEFT JOIN item_details d ON d.item_id = i.id
            GROUP BY i.id
            ORDER BY i.id DESC
        """
        params = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        cur.execute(sql, params)
        for iid, name, photo_path, img1_path, upload_path in cur.fetchall():
            path = img1_path or upload_path or photo_path
            if path and os.path.exists(path):
                rows.append(
                    {"item_id": iid, "title": name or "Artifact", "path": path})
        con.close()
    except Exception as e:
        print("front hub error:", e)
    return rows


def build_front_hub(parent, on_add_item, on_help_ident, on_open_library):
    """
    parent: a tk.Frame you provide (e.g., your home panel)
    Creates a slim top action bar and a bottom carousel of recent photos.
    """
    # Top mini menu
    top = tk.Frame(parent, bg=COLORS['bg_panel'])
    top.pack(fill='x', pady=(6, 4))
    lbl = tk.Label(top, text="Upload Artifact",
                   fg=COLORS['fg_primary'], bg=COLORS['bg_panel'])
    lbl.pack(side='left', padx=8)
    tk.Button(top, text="Add Item", command=on_add_item).pack(side='left')
    tk.Button(top, text="Help Identify", command=on_help_ident).pack(
        side='left', padx=6)
    tk.Button(top, text="Open Library", command=on_open_library).pack(
        side='right', padx=8)

    # Title for carousel
    tk.Label(parent, text="Your Collection",
             fg=COLORS['fg_primary'], bg=COLORS['bg_panel']).pack(anchor='w', padx=8, pady=(8, 2))


# --- Carousel frame (larger + auto-scroll) ---
    wrap = tk.Frame(parent, bg=COLORS['bg_panel'])
    wrap.pack(fill='x', padx=6, pady=(0, 10))

    canvas = tk.Canvas(
        wrap,
        height=CAROUSEL_CARD_H + 60,
        bg=COLORS['bg_panel'],
        highlightthickness=0
    )
    canvas.pack(side='left', fill='x', expand=True)

    hs = ttk.Scrollbar(wrap, orient='horizontal', command=canvas.xview)
    hs.pack(side='bottom', fill='x')
    canvas.configure(xscrollcommand=hs.set)

    rail = tk.Frame(canvas, bg=COLORS['bg_panel'])
    canvas.create_window((0, 0), window=rail, anchor='nw')
    # keep the scrollregion in sync as cards are added

    def _update_region(_=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    rail.bind("<Configure>", _update_region)

    # Click-drag panning (keep if you already have it)
    def _scan_mark(e): canvas.scan_mark(e.x, e.y)
    def _scan_drag(e): canvas.scan_dragto(e.x, e.y, gain=1)
    canvas.bind("<ButtonPress-1>", _scan_mark)
    canvas.bind("<B1-Motion>", _scan_drag)

# ----- Smooth auto-scroll (time-based easing, uses xview_moveto) -----
    _auto = {"job": None, "dir": 1, "v": 0.0, "t": None}  # v = px/sec

    def _pause_auto(*_):
        if _auto["job"] is not None:
            canvas.after_cancel(_auto["job"])
            _auto["job"] = None
        _auto["v"] = 0.0
        _auto["t"] = None

    def _resume_auto(delay_ms=600):
        if _auto["job"] is None:
            _auto["t"] = time.perf_counter()
            _auto["job"] = canvas.after(
                int(round(1000 / CAROUSEL_SCROLL_FPS)), _auto_step)

    def _auto_step():
        now = time.perf_counter()
        last = _auto["t"] or now
        dt = max(0.0, now - last)    # seconds since last tick
        _auto["t"] = now

        bbox = canvas.bbox("all")
        if not bbox:
            _auto["job"] = canvas.after(
                int(1000 / CAROUSEL_SCROLL_FPS), _auto_step)
            return

        content_w = max(1, bbox[2] - bbox[0])
        first, last = canvas.xview()
        view_frac = last - first

    # ease velocity toward target speed in current direction
        target_v = _auto["dir"] * CAROUSEL_SCROLL_SPEED_PX
        _auto["v"] += (target_v - _auto["v"]) * CAROUSEL_SCROLL_EASE

    # advance by dx pixels -> convert to fraction of total content width
        df = (_auto["v"] * dt) / content_w
        new_first = first + df

    # bounce at edges
        if new_first <= 0:
            new_first = 0
            _auto["dir"] = 1
            _auto["v"] = 0.0
        elif new_first >= 1 - view_frac:
            new_first = 1 - view_frac
            _auto["dir"] = -1
            _auto["v"] = 0.0

        canvas.xview_moveto(new_first)
        _auto["job"] = canvas.after(
            int(1000 / CAROUSEL_SCROLL_FPS), _auto_step)

# Wheel nudges (also pause/resume auto)
    def _nudge_px(px):
        bbox = canvas.bbox("all")
        if not bbox:
            return
        content_w = max(1, bbox[2] - bbox[0])
        first, last = canvas.xview()
        view_frac = last - first
        new_first = min(max(0.0, first + (px / content_w)), 1 - view_frac)
        canvas.xview_moveto(new_first)

    def _wheel_handler(units):
        _pause_auto()
    # ~30px per 'unit' feels right on Windows; adjust if you like
        _nudge_px(units * 30)
        _resume_auto()

    if sys.platform.startswith("win"):
        canvas.bind("<MouseWheel>",
                    lambda e: _wheel_handler(int(-e.delta/120)))
    elif sys.platform == "darwin":
        canvas.bind("<MouseWheel>", lambda e: _wheel_handler(
            int(-e.delta)))  # natural
    else:
        canvas.bind("<Button-4>", lambda e: _wheel_handler(-1))  # Linux
        canvas.bind("<Button-5>", lambda e: _wheel_handler(1))

# Optional: ensure 1-px increments for manual scrolls
    canvas.configure(xscrollincrement=1)

# --- populate cards (all items; batch to keep UI snappy)
    cards = _get_recent_photo_cards()   # no limit → all items
    CAROUSEL_BATCH_SIZE = 24            # adjust if you like

    def _render_cards(idx=0):
        end = min(idx + CAROUSEL_BATCH_SIZE, len(cards))
        for i in range(idx, end):
            card = cards[i]
            frm = tk.Frame(
                rail, bg=COLORS['bg_panel'], bd=0, highlightthickness=0)
            frm.grid(row=0, column=i, padx=CAROUSEL_PAD_X,
                     pady=CAROUSEL_PAD_Y, sticky='n')
            try:
                im = Image.open(card["path"]).convert("RGB")
                im.thumbnail((CAROUSEL_CARD_W, CAROUSEL_CARD_H),
                             Image.Resampling.LANCZOS)
                tkimg = ImageTk.PhotoImage(im)
                lbl = tk.Label(frm, image=tkimg,
                               bg=COLORS['bg_panel'], cursor="hand2")
                lbl.image = tkimg
                lbl.pack()
                title = (card.get("title") or "Artifact")[:60]
                tk.Label(frm, text=title,
                         fg=COLORS['fg_muted'], bg=COLORS['bg_panel']).pack()
                if card.get("item_id"):
                    lbl.bind("<Button-1>", lambda e,
                             iid=card["item_id"]: _open_item_page(iid))
            except Exception:
                tk.Label(frm, text="(image)",
                         fg=COLORS['fg_muted'], bg=COLORS['bg_panel']).pack()

        _update_region()
        if end < len(cards):
            canvas.after(0, lambda: _render_cards(end))

    if cards:
        _render_cards(0)
        # start auto-scroll once the layout settles
        canvas.after(800, _resume_auto)
    else:
        tk.Label(
            rail, text="No photos in your library yet.",
            fg=COLORS['fg_muted'], bg=COLORS['bg_panel']
        ).grid(row=0, column=0, padx=12, pady=12, sticky='w')


def _open_item_page(item_id: int):
    win = LibraryWindow(root)
    # after the tree populates, select the item and show details

    def _select():
        sid = str(item_id)
        if win.tree.exists(sid):
            win.tree.selection_set(sid)
            win.display_item_info(None)
    win.after(300, _select)

# ---------- Detail Window ----------


class ItemDetailWindow(tk.Toplevel):
    def __init__(self, master, initial_name: str, category: str, photo_path: str,
                 last_classification=None, item_id: int | None = None,
                 initial_details: dict | None = None, on_save=None):
        super().__init__(master)
        self.title("Item details")
        self.configure(bg=COLORS['bg_app'])
        self.minsize(900, 650)
        self.photo_path = photo_path
        self.last_classification = last_classification
        self.details_entries = {}
        self.meta_entries = {}
        self.item_id = item_id
        self._on_saved = on_save
        self.category_var = tk.StringVar(value=category)

        # Header
        header = SlateFrame(self)
        header.pack(fill=tk.X, padx=12, pady=(12, 0))
        SlateTitle(header, text=f"Review & Save Item — {category.capitalize()}").pack(
            anchor='w', pady=(4, 6))

        cat_row = SlateFrame(self)
        cat_row.pack(fill=tk.X, padx=12, pady=(0, 4))
        SlateLabel(cat_row, text="Category").grid(
            row=0, column=0, sticky='e', padx=(8, 4))
        self.category_combo = ttk.Combobox(
            cat_row, textvariable=self.category_var, state='readonly',
            values=list(SCHEMAS.keys()), width=18
        )
        self.category_combo.grid(row=0, column=1, sticky='w', padx=(0, 12))

        # Photo
        row_photo = SlateFrame(self)
        row_photo.pack(fill=tk.X, padx=12, pady=8)
        SlateLabel(row_photo, text="Photo").grid(
            row=0, column=0, sticky='ne', padx=(8, 6))
        img_box = SlateFrame(row_photo)
        img_box.grid(row=0, column=1, sticky='w')
        try:
            im = Image.open(photo_path)
            im.thumbnail((300, 300), Image.Resampling.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(im)
            tk.Label(img_box, image=self._tk_img, bg=COLORS['bg_panel']).pack()
        except Exception:
            SlateLabel(img_box, text="(image unavailable)").pack()

        # Scrollable fields
        scroll = ScrollFrame(self)
        scroll.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        body = scroll.inner

        cols = SlateFrame(body)
        cols.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        left = SlateFrame(cols)
        right = SlateFrame(cols)
        left.pack(side='left', fill=tk.BOTH, expand=True, padx=(0, 6))
        right.pack(side='left', fill=tk.BOTH, expand=True, padx=(6, 0))

        SlateTitle(left, text="You enter").pack(
            anchor='w', padx=8, pady=(6, 4))
        self.user_frame = SlateFrame(left)
        self.user_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        SlateTitle(right, text="Auto from API").pack(
            anchor='w', padx=8, pady=(6, 4))
        self.api_frame = SlateFrame(right)
        self.api_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        schema = SCHEMAS.get(self.category_var.get().lower(), SCHEMAS["other"])
        for k in schema["user"]:
            self._add_field(self.user_frame, k, is_api=False)
        for k in schema["api"]:
            if k in HIDDEN_META_KEYS:
                continue
            self._add_field(self.api_frame, k, is_api=True)

        if "name" in self.details_entries:
            e = self.details_entries["name"]
            e.delete(0, tk.END)
            e.insert(0, initial_name)

        # Populate any existing details
        initial_details = initial_details or {}
        for k, v in initial_details.items():
            if k in self.details_entries:
                e = self.details_entries[k]
                e.delete(0, tk.END)
                e.insert(0, v)
            elif k in self.meta_entries:
                e = self.meta_entries[k]
                e.delete(0, tk.END)
                e.insert(0, v)

        # Pinned buttons
        btns = SlateFrame(self)
        btns.pack(fill=tk.X, padx=12, pady=(0, 12))
        self.collect_btn = SlateButton(
            btns, text="Collect Details", command=self.collect_details)
        self.collect_btn.pack(side='left', padx=6)
        SlateButton(btns, text="Save Item", command=self.save_item).pack(
            side='left', padx=6)
        SlateButton(btns, text="Cancel", command=self.destroy).pack(
            side='left', padx=6)
        self.status = SlateLabel(btns, text="")
        self.status.pack(side='left', padx=12)

    def _add_field(self, parent, key: str, is_api: bool):
        row = SlateFrame(parent)
        row.pack(fill=tk.X, pady=3)
        row.grid_columnconfigure(1, weight=1)
        label = key.replace("_", " ")
        SlateLabel(row, text=label).grid(
            row=0, column=0, sticky='e', padx=(8, 6))
        var = tk.StringVar(value="")
        e = tk.Entry(row, textvariable=var, bg=COLORS['bg_panel'], fg=COLORS['fg_primary'],
                     insertbackground=COLORS['accent_a'], relief='flat', font=('Segoe UI', 10))
        e.grid(row=0, column=1, sticky='we')
        (self.meta_entries if is_api else self.details_entries)[key] = e
        if not is_api and key in {"barcode_ean_upc"}:
            try:
                btn = SlateButton(row, text="Scan",
                                  command=lambda k=key: self.scan_barcode(k))
                btn.grid(row=0, column=2, padx=(6, 8))
            except Exception:
                pass

    def scan_barcode(self, field_key: str = "barcode_ean_upc"):
        """Open camera and decode a UPC/EAN; writes to the given field. Threaded (no GUI freeze)."""
        import tkinter as _tk
        import threading
        try:
            import cv2
            from PIL import Image, ImageTk
        except Exception:
            messagebox.showinfo(
                "Barcode scan", "Install deps: pip install opencv-python pillow pyzbar")
            return
        try:
            from pyzbar.pyzbar import decode as _pz_decode
        except Exception:
            _pz_decode = None

        def _digits(s): return "".join(ch for ch in (s or "") if ch.isdigit())

        def _ean13_ok(s):
            if len(s) != 13 or not s.isdigit():
                return False
            d = list(map(int, s))
            chk = d[-1]
            ss = 3*sum(d[0:12:2]) + sum(d[1:12:2])
            return (10 - (ss % 10)) % 10 == chk

        def _upc_ok(s):
            if len(s) != 12 or not s.isdigit():
                return False
            d = list(map(int, s))
            chk = d[-1]
            ss = 3*sum(d[0:11:2]) + sum(d[1:11:2])
            return (10 - (ss % 10)) % 10 == chk

        def _normalize(code):
            s = _digits(code)
            if len(s) == 13 and s.startswith("0") and _ean13_ok(s):
                t = s[1:]
                if _upc_ok(t):
                    return t
            return s

        def _decode_once(img, detector):
            vals = set()
            try:
                res = detector.detectAndDecode(
                    img) if detector is not None else ("", None, None)
                a = res[0] if isinstance(res, tuple) else res
                if isinstance(a, (list, tuple)):
                    for v in a:
                        if v:
                            vals.add(_digits(v))
                else:
                    if a:
                        vals.add(_digits(a))
            except Exception:
                pass
            if _pz_decode is not None:
                try:
                    for d in _pz_decode(img):
                        try:
                            v = d.data.decode("utf-8", errors="ignore")
                        except Exception:
                            v = str(d.data)
                        if v:
                            vals.add(_digits(v))
                except Exception:
                    pass
            vals = [v for v in vals if len(v) >= 8]
            if not vals:
                return ""
            vals.sort(key=lambda s: (1 if (_upc_ok(s) or _ean13_ok(s))
                      else 0, len(s)), reverse=True)
            return vals[0]

        def _heavy(img, detector):
            import cv2
            import numpy as np
            h, w = img.shape[:2]
            for s in (1.0, 1.3, 1.6, 2.0, 2.5):
                try:
                    im = cv2.resize(img, (int(
                        w*s), int(h*s)), interpolation=cv2.INTER_CUBIC) if s != 1.0 else img.copy()
                except Exception:
                    im = img.copy()
                hh, ww = im.shape[:2]
                for frac in (0.98, 0.92, 0.85, 0.75):
                    cw, ch = int(ww*frac), int(hh*frac)
                    cx, cy = (ww-cw)//2, (hh-ch)//2
                    crop = im[cy:cy+ch, cx:cx+cw]
                    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                        try:
                            r = cv2.rotate(
                                crop, rot) if rot is not None else crop
                        except Exception:
                            r = crop
                        try:
                            gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
                        except Exception:
                            gray = r
                        try:
                            blur = cv2.GaussianBlur(gray, (5, 5), 0)
                            th_a = cv2.adaptiveThreshold(
                                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                            _, th_o = cv2.threshold(
                                blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            kernel = np.ones((3, 3), np.uint8)
                            morph = cv2.morphologyEx(
                                th_o, cv2.MORPH_CLOSE, kernel)
                            cands = (r, gray, blur, th_a, th_o, morph)
                        except Exception:
                            cands = (r, gray)
                        for cand in cands:
                            code = _decode_once(cand, detector)
                            if code:
                                return code
            return ""

        win = _tk.Toplevel(self)
        win.title("Scan barcode")
        win.configure(bg=COLORS['bg_panel'])
        win.transient(self)
        win.grab_set()
        preview = _tk.Label(win, bg=COLORS['bg_panel'])
        preview.pack(padx=10, pady=(10, 6))
        status = _tk.Label(win, text="Align barcode in the box. Press Capture (or Space).",
                           bg=COLORS['bg_panel'], fg=COLORS['fg_primary'])
        status.pack(padx=10, pady=(0, 8))
        btns = _tk.Frame(win, bg=COLORS['bg_panel'])
        btns.pack(padx=10, pady=(0, 10))
        cap_btn = SlateButton(btns, text="Capture")
        cap_btn.pack(side='left', padx=6)
        SlateButton(btns, text="Close", command=win.destroy).pack(
            side='left', padx=6)

        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(
            cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            win.destroy()
            messagebox.showerror("Barcode scanner", "No camera found.")
            return
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass
        try:
            detector = cv2.barcode.BarcodeDetector()
        except Exception:
            detector = None

        last = {"frame": None}
        busy = {"flag": False}

        def on_close():
            try:
                cap.release()
            except Exception:
                pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)
        win.bind("<Escape>", lambda e: on_close())

        def commit(val):
            s = _normalize(val)
            e = self.details_entries.get(field_key)
            if e:
                e.delete(0, tk.END)
                e.insert(0, s)
            else:
                messagebox.showinfo("Barcode", s)
            on_close()

        def after(guess):
            busy["flag"] = False
            try:
                cap_btn.config(state="normal", text="Capture")
            except Exception:
                pass
            if guess:
                commit(guess)
            else:
                status.config(text="No barcode found. Try again.")

        def do_capture(event=None):
            if busy["flag"]:
                return
            frame = last["frame"]
            if frame is None:
                return
            busy["flag"] = True
            try:
                cap_btn.config(state="disabled", text="Processing…")
            except Exception:
                pass
            status.config(text="Processing capture…")
            win.update_idletasks()

            img = frame.copy()
            h, w = img.shape[:2]
            rw, rh = int(w*0.96), int(h*0.55)
            rx, ry = (w-rw)//2, (h-rh)//2
            roi = img[ry:ry+rh, rx:rx+rw].copy()

            def worker():
                try:
                    for cand in (roi, img):
                        code = _heavy(cand, detector)
                        if code:
                            self.after(0, lambda g=code: after(g))
                            return
                    self.after(0, lambda: after(""))
                except Exception:
                    self.after(0, lambda: after(""))
            threading.Thread(target=worker, daemon=True).start()

        cap_btn.config(command=do_capture)
        win.bind("<space>", lambda e: do_capture())

        def tick():
            ok, frame = cap.read()
            if not ok:
                win.after(15, tick)
                return
            last["frame"] = frame.copy()
            h, w = frame.shape[:2]
            rw, rh = int(w*0.96), int(h*0.55)
            rx, ry = (w-rw)//2, (h-rh)//2
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame_rgb)
            maxw = 960
            if w > maxw:
                im = im.resize((maxw, int(maxw*h/w)))
            tk_img = ImageTk.PhotoImage(im)
            preview.configure(image=tk_img)
            preview.image = tk_img
            win.after(30, tick)

        tick()

    def _gather_user_helpers(self) -> dict:
        helpers = {}
        for k, e in self.details_entries.items():
            helpers[k] = e.get().strip()
        for k, e in self.meta_entries.items():
            v = e.get().strip()
            if v:
                helpers.setdefault(k, v)
        return helpers

    def _apply_details(self, meta: dict):
        if not meta:
            self.status.config(text="No enrichment found.")
            return
        filled = 0
        for k, v in (meta or {}).items():
            if k in self.meta_entries and v:
                entry = self.meta_entries[k]
                entry.delete(0, tk.END)
                entry.insert(0, v)
                filled += 1
        self.status.config(text=("No confident fields returned." if filled ==
                           0 else f"Details collected. {filled} field(s) populated."))

    def collect_details(self):
        alias_map = {
            'records': 'vinyl', 'record': 'vinyl',
            'shells': 'shell', 'minerals': 'mineral',
            'fossils': 'fossil', 'coins': 'coin'
        }
        self.collect_btn.config(state='disabled')
        self.status.config(text="Collecting details…")
        raw_cat = (self.category_var.get() or '').strip()
        cat = alias_map.get(raw_cat.lower(), raw_cat.lower())
        helpers = self._gather_user_helpers()
        raw_name = helpers.get("name", "").strip()
        name_for_prompt = raw_name or "(unnamed item)"
        schema = SCHEMAS.get(cat, SCHEMAS["other"])
        key_list = list(dict.fromkeys(schema['api']))

        def _work():
            meta, err = None, None
            try:
                if cat == "fossil":
                    lbl, conf = "", 1.0
                    if self.last_classification:
                        lbl = self.last_classification.get("label", "")
                        conf = float(self.last_classification.get(
                            "confidence", 0) or 0)
                    meta = enrich_fossil_two_pass(name_for_prompt, lbl, conf)
                elif cat in ("mineral", "shell"):
                    meta = enrich_generic_openai(
                        name_for_prompt, cat, self.photo_path, helpers, key_list)
                elif cat == "vinyl":
                    meta = enrich_vinyl(
                        name_for_prompt, self.photo_path, helpers)
                    try:
                        self._last_meta_result = dict(meta)
                    except Exception:
                        self._last_meta_result = meta
                elif cat == "coin":
                    meta = enrich_coin(name_for_prompt, helpers)
                elif cat == "zoological":
                    # Use two-pass zoological enrichment with the image so the model
                    # can identify the organism (e.g., "death's-head hawkmoth") rather than the container.
                    lbl = raw_name or (
                        self.last_classification.get(
                            "label", "") if self.last_classification else ""
                    )
                    conf = (
                        float(self.last_classification.get(
                            "confidence", 0) or 0)
                        if self.last_classification else 0.0
                    )
                    meta = enrich_zoological_two_pass(
                        lbl or "zoological specimen", lbl, self.photo_path, conf
                    )
                else:
                    meta = {}
            except Exception as e:
                err = str(e)

            def _ui():
                if err:
                    messagebox.showerror("Enrichment error", err)
                    self.status.config(text="Enrichment failed.")
                else:
                    self._apply_details(meta)
                self.collect_btn.config(state='normal')
            self.after(0, _ui)
        threading.Thread(target=_work, daemon=True).start()

    def save_item(self):
        name = self.details_entries.get("name").get().strip(
        ) if self.details_entries.get("name") else ""
        if not name:
            messagebox.showerror(
                "Missing name", "Please enter a name before saving.")
            return
        details = {}
        for k, e in self.details_entries.items():
            v = e.get().strip()
            if v:
                details[k] = v
        for k, e in self.meta_entries.items():
            v = e.get().strip()
            if v:
                details[k] = v

# Include hidden Discogs fields captured during enrichment
        try:
            if hasattr(self, "_last_meta_result") and isinstance(self._last_meta_result, dict):
                for hk in ("discogs_release_id", "discogs_url", "discogs_median_price_usd", "discogs_low_high_usd"):
                    hv = self._last_meta_result.get(hk)
                    if hv and hk not in details:
                        details[hk] = str(hv)
        except Exception:
            pass

        try:
            cat = self.category_var.get()
            if self.item_id is not None:
                item_id = update_item_full(
                    self.item_id, name, cat, self.photo_path, details)
            else:
                item_id = save_item_full(
                    name, cat, self.photo_path, details)
            messagebox.showinfo("Saved", f"Item saved (id {item_id}).")
            if callable(self._on_saved):
                self._on_saved()
            self.destroy()
        except Exception as e:
            messagebox.showerror("DB error", str(e))

# ---------- Classifier App (pass-in root) ----------


class ClassifierApp:
    def __init__(self, root: tk.Tk):
        self.master = root
        self.master.title('artiFACTS')
        self.master.configure(bg=COLORS['bg_app'])
        try:
            self.master.state('zoomed')
        except tk.TclError:
            self.master.attributes('-zoomed', True)

        banner_frame = SlateFrame(self.master)
        banner_frame.pack(fill=tk.X, padx=12, pady=(12, 6))
        self.banner = tk.Canvas(banner_frame, height=96,
                                bg=COLORS['bg_panel'], highlightthickness=0)
        self.banner.pack(fill=tk.X)
        self.banner.bind('<Configure>', lambda e: draw_banner(
            self.banner, e.width, e.height))

        main = SlateFrame(self.master)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        for r in (0, 1):
            main.grid_rowconfigure(r, weight=1, uniform="rows")
        for c in (0, 1):
            main.grid_columnconfigure(c, weight=1, uniform="cols")

        # Controls
        tl = SlateFrame(main)
        tl.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        SlateTitle(tl, text='Controls').grid(row=0, column=0,
                                             columnspan=6, sticky='w', padx=8, pady=(8, 6))
        SlateLabel(tl, text='Category').grid(
            row=1, column=0, sticky='e', padx=(8, 4))
        self.category_var = tk.StringVar(value='mineral')
        self.category_combo = ttk.Combobox(
            tl, textvariable=self.category_var, state='readonly',
            values=list(SCHEMAS.keys()), width=14
        )
        self.category_combo.grid(
            row=1, column=1, sticky='w', padx=(0, 12), pady=6)
        self.category_combo.bind(
            "<<ComboboxSelected>>", self._on_category_change)
        self.last_classification = None
        self._species_mode = False
        self._busy_count = 0
        self._img_search_epoch = 0
        self._ref_job = None
        self._last_image_path = ""
        SlateButton(tl, text='Select Image', command=self.select_image).grid(
            row=1, column=2, padx=6)
        SlateButton(tl, text='Capture Image', command=self.capture_image).grid(
            row=1, column=3, padx=6)
        SlateButton(tl, text='View Library', command=self.view_library).grid(
            row=1, column=4, padx=6)

        # Selected image
        tr = SlateFrame(main)
        tr.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        SlateTitle(tr, text='Selected Image').pack(
            anchor='w', padx=8, pady=(8, 6))
        self.image_label = SlateLabel(
            tr, text='No image selected', font=('Segoe UI', 11, 'italic'))
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # Results
        bl = SlateFrame(main)
        bl.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        SlateTitle(bl, text='Classification Results').pack(
            anchor='w', padx=8, pady=(8, 6))
        actions = SlateFrame(bl)
        actions.pack(fill=tk.X, padx=8, pady=(0, 6))
        SlateLabel(actions, text="Choose result:").grid(
            row=0, column=0, padx=(8, 6), pady=6, sticky="w")
        self.rb_frame = SlateFrame(actions)
        self.rb_frame.grid(row=0, column=1, padx=(0, 12), pady=6, sticky="w")
        SlateButton(actions, text="Review & Save", command=self.open_detail_window).grid(
            row=0, column=2, padx=6)
        results_wrap = SlateFrame(bl)
        results_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.result_text = SlateText(results_wrap, height=10)
        self.result_text.pack(side='left', fill=tk.BOTH, expand=True)
        res_scroll = tk.Scrollbar(
            results_wrap, orient='vertical', command=self.result_text.yview)
        res_scroll.pack(side='right', fill='y')
        self.result_text.configure(yscrollcommand=res_scroll.set)

        # Reference images
        br = SlateFrame(main)
        br.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))
        SlateTitle(br, text='Reference images').pack(
            anchor='w', padx=8, pady=(8, 6))
        note = "DuckDuckGo previews" if _DDG_OK else "Install: pip install duckduckgo-search"
        self.ref_note = SlateLabel(br, text=note)
        self.ref_note.pack(anchor='w', padx=8, pady=(0, 6))
        self.ref_frame = SlateFrame(br)
        self.ref_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        for c in (0, 1):
            self.ref_frame.grid_columnconfigure(c, weight=1)
        self.ref_imgs = []
        self.ref_widgets = []
        for i in range(4):
            ph = SlateLabel(self.ref_frame, text="—",
                            font=('Segoe UI', 10, 'italic'))
            r, c = divmod(i, 2)
            ph.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            self.ref_widgets.append(ph)

        self._rb_widgets = []
        self._chosen_label = ""
        self._last_element = {"element": "", "confidence": 0.0}

    def _on_category_change(self, event=None):
        self._species_mode = False

    def _clear_reference_images(self, msg=""):
        for w in self.ref_widgets:
            w.config(image='', text="—")
        if msg:
            self.ref_note.config(text=msg)

    def _display_guess_label(self, raw_label: str) -> str:
        """
        For Shells, flip 'Scientific (Common)' -> 'Common (Scientific)'.
        Leave all other categories/labels unchanged.
        """
        try:
            cat = (self.category_var.get() or "").strip().lower()
        except Exception:
            cat = ""
        if cat.startswith("shell"):  # e.g., 'Shells', 'Shell', etc.
            m = re.match(r"\s*([^(]+?)\s*\(([^)]+)\)\s*$", raw_label or "")
            if m:
                scientific = m.group(1).strip()
                common = m.group(2).strip()
                return f"{common} ({scientific})"
        return raw_label

    def _set_reference_images(self, items):
        self.ref_imgs.clear()
        for w in self.ref_widgets:
            w.config(image='', text="—")
        for i, (img_url, page_url) in enumerate(items[:4]):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                r = requests.get(img_url, headers=headers, timeout=10)
                r.raise_for_status()
                im = Image.open(io.BytesIO(r.content))
                im.thumbnail((180, 180), Image.Resampling.LANCZOS)
                tk_im = ImageTk.PhotoImage(im)
                self.ref_widgets[i].config(image=tk_im, text='')
                self.ref_widgets[i].image = tk_im
                self.ref_widgets[i].bind(
                    "<Button-1>", lambda e, u=page_url: webbrowser.open(u))
            except Exception:
                self.ref_widgets[i].config(text="(failed)")

    def _label_hint(self) -> str:
        """Return the best current label hint for species guessing."""
        try:
            if getattr(self, "last_classification", None):
                lab = (self.last_classification.get("label") or "").strip()
                if lab:
                    return lab
        except Exception:
            pass
        return ""

    def _safe_load_refs(self, term: str):
        try:
            if self._ref_job:
                self.master.after_cancel(self._ref_job)
                self._ref_job = None
        except Exception:
            pass
        self._ref_job = self.master.after(
            250, lambda t=term: self.load_reference_images(t))

    def on_guess_selected(self, term: str):
        self._chosen_label = term
        if not _DDG_OK:
            self._clear_reference_images(
                "Install: pip install ddgs")
            return
        cat = (self.category_var.get() or "").lower().strip()
        sel = (term or "").strip()

        if cat == "zoological":
            # Try to recover element + species from the last stored label
            el, sp = ("", "")
            try:
                if getattr(self, "last_classification", None):
                    lbl = self.last_classification.get("label", "")
                    el, sp = _parse_element_species_label(lbl)
            except Exception:
                pass
            # Fall back to the clicked text if species missing
            if not sp:
                sp = sel.split("[", 1)[0].replace(
                    "“", "").replace("”", "").strip()
            # Build element-aware query (whole organisms remain species-only)
            clean = _ref_query_for_zoo(sp, el)
        else:
            clean = _decontainerize_label(sel)

        self._safe_load_refs(clean)

    def load_reference_images(self, term: str | None = None):
        """Fetch and display DuckDuckGo reference images for the current category/term.

        - Runs in a worker thread, marshals UI updates back to the main thread.
        - Uses an epoch guard so older requests can't clobber newer results.
        - Tries a few sensible query variants before giving up.
        """
        self._busy_push()
        # cancel any pending scheduled call
        self._ref_job = None

        cat = (self.category_var.get() or "").lower().strip()
        base = (term or "").strip()

        # If no explicit term passed, fall back to any selected/derived label
        try:
            if not base:
                base = (getattr(self, "_chosen_label", "") or "").strip()
        except Exception:
            pass
        if not base:
            self._clear_reference_images("No query available.")
            self._busy_pop()
            return

        # Zoological: if the label is generic (e.g., "insect specimen"), try to promote to species via enrichment
        if cat == "zoological":
            try:
                norm = normalize_zoo_label(base)
                generic = {
                    "zoological specimen", "insect specimen", "butterfly specimen",
                    "beetle specimen", "moth specimen", "insect", "butterfly",
                    "beetle", "moth"
                }
                search_base = norm or base
                if (not norm) or ("specimen" in norm) or (norm in generic):
                    photo = getattr(self, "_last_image_path", "") or ""
                    meta = enrich_zoological_two_pass(
                        search_base, search_base, photo, 0.7)
                    sp = (meta.get("scientific_name") or "").strip()
                    if sp:
                        base = sp
            except Exception:
                # Keep original base if anything fails
                pass

        # Primary + fallbacks
        def _build_qs(b):
            try:
                return _build_reference_query(cat, b)
            except Exception:
                return b

        queries = [
            _build_qs(base),
            _build_qs(f"{base} specimen"),
            _build_qs(f"{base} macro"),
            _build_qs(f"{base} museum"),
        ]
        if cat == "vinyl":
            queries += [f"{base} album cover", base]

        # De-dupe while preserving order
        seen = set()
        queries = [q for q in queries if not (q in seen or seen.add(q))]

        # Show spinner text, bump epoch
        self._clear_reference_images("Searching…")
        self._img_search_epoch += 1
        epoch = self._img_search_epoch

        def _run(qs, ep, base_term):
            self._busy_push()
            err = None
            results = []
            chosen = base_term
            for q in qs:
                chosen = q
                try:
                    results = image_search(q, max_results=8)
                    if results:
                        break
                except (SearchRateLimited, SearchError) as e:
                    err = e
                    break
            self.master.after(0, lambda: self._image_search_complete(
                ep, chosen, results, err))

        threading.Thread(target=_run, args=(
            queries, epoch, base), daemon=True).start()

    def _image_search_complete(self, epoch: int, term: str, results, err=None):
        try:
            if epoch != self._img_search_epoch:
                return
            if err:
                self.ref_note.config(text="Rate-limited; retrying shortly")
                self.master.after(
                    6000, lambda t=term: self.load_reference_images(t))
            elif results:
                self.ref_note.config(text=f"Results for: {term}")
                self._set_reference_images(results)
            else:
                self._clear_reference_images("No images found.")
        finally:
            # Always unwind both pushes
            self._busy_pop()
            self._busy_pop()

    # --- Busy / reset helpers -------------------------------------------------

    def _busy_push(self):
        self._busy_count = getattr(self, "_busy_count", 0) + 1
        try:
            self.master.config(cursor="watch")
        except Exception:
            pass

    def _busy_pop(self):
        self._busy_count = max(0, getattr(self, "_busy_count", 1) - 1)
        if self._busy_count == 0:
            try:
                self.master.config(cursor="")
            except Exception:
                pass

    def _cancel_ref_work(self):
        """Cancel any scheduled or in-flight reference image lookups."""
        # Cancel any pending .after() call
        try:
            if getattr(self, "_ref_job", None):
                self.master.after_cancel(self._ref_job)
                self._ref_job = None
        except Exception:
            pass
        # Invalidate worker threads by bumping the epoch
        self._img_search_epoch = getattr(self, "_img_search_epoch", 0) + 1

    def _clear_classification(self, msg: str = ""):
        """Wipe the guesses UI and optionally show a placeholder message."""
        # Text panel
        try:
            self.result_text.config(state='normal')
            self.result_text.delete(1.0, tk.END)
            if msg:
                self.result_text.insert(tk.END, f"{msg}\n")
            self.result_text.config(state='disabled')
        except Exception:
            pass
        # Radio buttons
        for w in getattr(self, "_rb_widgets", []):
            try:
                w.destroy()
            except Exception:
                pass
        self._rb_widgets = []
        self._chosen_label = ""

    def _reset_for_new_image(self):
        """Call this whenever a *new* image path/frame is set."""
        self._cancel_ref_work()
        # Clear right away so the user sees things change
        try:
            self._clear_reference_images("—")
        except Exception:
            pass
        self._clear_classification("Classifying…")

    def view_library(self): LibraryWindow(self.master)

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[('Image files', '*.jpg *.jpeg *.png')])
        if not path:
            return
        self._reset_for_new_image()
        self._last_image_path = path
        self.process_image(path)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror('Error', 'Camera not accessible.')
            return
        cv2.namedWindow('Press SPACE to capture', cv2.WINDOW_NORMAL)
        frame = None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imshow('Press SPACE to capture', img)
            key = cv2.waitKey(1)
            if key == 32:
                frame = img.copy()
                break
            elif key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        if frame is not None:
            self._reset_for_new_image()
            self._busy_push()
            self.process_frame(frame)

    def process_image(self, path):
        try:
            img = Image.open(path).resize((300, 300))
            tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=tk_img, text='')
            self.image_label.image = tk_img
            cropped = _autocrop_inside_case(path)
            self._last_image_path = cropped or path
            cat = (self.category_var.get() or "").lower().strip()
            if cat == "zoological":
                self._species_mode = True
                self._busy_push()
                self.result_text.config(state='normal')
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(
                    tk.END, "Finding species candidates…\n")
                self.result_text.config(state='disabled')
                self.master.update_idletasks()

                def _bg_species():
                    try:
                        self._last_element = guess_specimen_element(self._last_image_path) or {
                            "element": "", "confidence": 0.0}
                    except Exception:
                        self._last_element = {"element": "", "confidence": 0.0}
                    try:
                        cands = guess_species_from_image(
                            self._last_image_path, family_hint="", max_results=4)
                        elem = (self._last_element.get(
                            "element") or "").strip().lower()
                        if elem:
                            for c in (cands or []):
                                c["element"] = elem
                    except Exception:
                        cands = []
                    self.master.after(
                        0, lambda: self._show_species_candidates(cands))
                    self.master.after(0, self._busy_pop)
                threading.Thread(target=_bg_species, daemon=True).start()
                return
            self._species_mode = False
            self.classify_and_display(self._last_image_path)
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def process_frame(self, frame):
        try:
            tmp = os.path.join(tempfile.gettempdir(), 'artifacts_capture.jpg')
            if not cv2.imwrite(tmp, frame):
                messagebox.showerror('Error', 'Could not write temp image.')
                return
            self._last_image_path = tmp
            img = Image.fromarray(cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB)).resize((300, 300))
            tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=tk_img, text='')
            self.image_label.image = tk_img
            cropped = _autocrop_inside_case(tmp)
            self._last_image_path = cropped or tmp
            cat = (self.category_var.get() or "").lower().strip()
            if cat == "zoological":
                self._species_mode = True
                self._busy_push()
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(
                    tk.END, "Finding species candidates…\n")
                self.master.update_idletasks()

                def _bg_species():
                    try:
                        self._last_element = guess_specimen_element(self._last_image_path) or {
                            "element": "", "confidence": 0.0}
                    except Exception:
                        self._last_element = {"element": "", "confidence": 0.0}
                    try:
                        cands = guess_species_from_image(
                            self._last_image_path, family_hint="", max_results=4)
                        elem = (self._last_element.get(
                            "element") or "").strip().lower()
                        if elem:
                            for c in (cands or []):
                                c["element"] = elem
                    except Exception:
                        cands = []
                    self.master.after(
                        0, lambda: self._show_species_candidates(cands))
                    self.master.after(0, self._busy_pop)
                threading.Thread(target=_bg_species, daemon=True).start()
                return
            self._species_mode = False
            self.classify_and_display(self._last_image_path)
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def classify_and_display(self, image_path: str):
        cat = (self.category_var.get() or "").strip().lower()
        if cat == "zoological" and self._species_mode:
            return
        # Make sure UI is in a "busy" state while we classify
        self._busy_push()
        # If something loaded us directly (e.g., library), ensure panes are clean
        if not getattr(self, "_rb_widgets", []):
            # Avoid flicker when radio buttons already exist
            self._clear_classification("Classifying…")
        cat = cat or 'other'
        result = run_classifier(image_path, cat)
        openai_block = result.get(
            'openai', {}) if isinstance(result, dict) else {}
        oa_guesses = (openai_block.get('guesses') or [])[:3]
        oa_guesses = _decontainerize_guesses(oa_guesses)
        if cat == "zoological":
            for g in oa_guesses:
                g["label"] = normalize_zoo_label(g.get("label", ""))
        prio_words = [
            "specimen", "butterfly", "moth", "beetle", "insect",
            "lepidoptera", "coleoptera", "arthropod",
            "bone", "tooth", "antler", "horn", "feather", "skin", "taxidermy"
        ]
        good, rest = [], []
        for g in oa_guesses:
            lbl = (g.get("label") or "").lower()
            (good if any(w in lbl for w in prio_words) else rest).append(g)
        oa_guesses = good + rest
        if oa_guesses:
            top = oa_guesses[0]
            self.last_classification = {
                'label': top.get('label', ''),
                'confidence': float(top.get('confidence', 0) or 0)
            }
        else:
            self.last_classification = None

        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        if oa_guesses:
            for i, g in enumerate(oa_guesses, 1):
                label = self._display_guess_label(g.get('label', 'Unknown'))
                conf = float(g.get('confidence', 0) or 0)
                rationale = g.get('rationale', '')
                self.result_text.insert(
                    tk.END, f"  {i}. {label}  [conf {conf:.1%}]\n")
                if rationale:
                    self.result_text.insert(tk.END, f"     ↳ {rationale}\n")
        else:
            self.result_text.insert(tk.END, "  (no guesses)\n")
        self.result_text.config(state='disabled')

        self._last_image_path = image_path
        self._refresh_guess_radios(oa_guesses)
        self._busy_pop()

    def _refresh_guess_radios(self, guesses):
        for w in getattr(self, "_rb_widgets", []):
            try:
                w.destroy()
            except Exception:
                pass
        self._rb_widgets = []
        var = tk.StringVar(value="")
        self._rb_var = var
        for i, g in enumerate(guesses[:3], 1):
            label = self._display_guess_label(g.get("label", f"Option {i}"))
            rb = tk.Radiobutton(
                self.rb_frame, text=label, value=label, variable=var,
                command=lambda val=label: self.on_guess_selected(val),
                bg=COLORS['bg_panel'], fg=COLORS['fg_primary'],
                selectcolor=COLORS['accent_b'], activebackground=COLORS['bg_panel'],
                font=('Segoe UI', 10), anchor='w', indicatoron=1, takefocus=1
            )
            rb.pack(anchor="w", pady=2, fill='x')
            self._rb_widgets.append(rb)
        if self._rb_widgets:
            first = self._rb_widgets[0]
            self.master.after_idle(first.invoke)

    def _format_taxon_label(self, cand: dict, include_conf: bool = True) -> str:
        sci = (cand.get("scientific_name") or "").strip()
        com = (cand.get("common_name") or "").strip()
        elem = (cand.get("element") or "").strip().lower()
        conf = float(cand.get("confidence") or 0.0)

        if elem in WHOLE_ELEMENTS or not elem:
            bits = [sci]
            if com:
                bits.append(f"“{com}”")
            if include_conf:
                bits.append(f"[{int(round(conf*100))}%]")
            return " ".join([b for b in bits if b]).strip()

        prefix = elem.capitalize()
        label = f"{prefix} — {sci}" if sci else prefix
        if com:
            label += f' “{com}”'
        if include_conf:
            label += f" [{int(round(conf*100))}%]"
        return label

    def _show_species_candidates(self, cands):
        cands = list(cands or [])
        top = cands[:3]

        def _score_of(c):
            s = c.get("score") or c.get("confidence") or c.get("prob") or 0
            try:
                s = float(s)
                if 0.0 <= s <= 1.0:
                    s *= 100.0
                return int(round(s))
            except Exception:
                return 0

        def _line_text(c):
            # e.g., "Urania ripheus “Madagascan sunset moth”"
            label = self._format_taxon_label(c)
            pct = _score_of(c)
            return f"{label}  [{pct}%]"

        def _why_text(c):
            rat = (c.get("rationale") or c.get("explanation") or "").strip()
            return f"   ↳ {rat}" if rat else ""

    # ---- Text panel ----
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        if top:
            self.result_text.insert(tk.END, "Top species:\n")
            for i, c in enumerate(top, 1):
                self.result_text.insert(tk.END, f"{i}. { _line_text(c) }\n")
                w = _why_text(c)
                if w:
                    self.result_text.insert(tk.END, w + "\n")
        else:
            self.result_text.insert(
                tk.END, "No species-level match. Try a closer, glare-free photo.\n"
            )
        self.result_text.config(state="disabled")

    # ---- Radios (limit to 3) ----
        try:
            for w in getattr(self, "_rb_widgets", []):
                w.destroy()
        except Exception:
            pass
        self._rb_widgets = []

    # ensure a shared StringVar for the group
        var = getattr(self, "_rb_var", None)
        if not isinstance(var, tk.StringVar):
            var = tk.StringVar(value="")
            self._rb_var = var

        for c in top:
            label = self._format_taxon_label(c)
            value = label  # if on_guess_selected decontainers, this is fine
            rb = tk.Radiobutton(
                self.rb_frame,
                text=label, value=value, variable=var,
                command=lambda v=value: self.on_guess_selected(v),
                bg=COLORS['bg_panel'], fg=COLORS['fg_primary'],
                selectcolor=COLORS['accent_b'], activebackground=COLORS['bg_panel'],
                font=('Segoe UI', 10), anchor='w', indicatoron=1, takefocus=1
            )
            rb.pack(anchor='w', pady=2, fill='x')
            self._rb_widgets.append(rb)

    # auto-select first at idle (prevents busy leak)
        if self._rb_widgets:
            first = self._rb_widgets[0]
            (self.after_idle if hasattr(self, "after_idle")
             else self.master.after_idle)(first.invoke)

    def open_detail_window(self):
        chosen = (self._chosen_label or "").strip()
        if not chosen:
            messagebox.showinfo(
                "Pick one", "Please select one of the guesses first.")
            return
        ItemDetailWindow(self.master, initial_name=chosen,
                         category=(self.category_var.get() or "other").strip(),
                         photo_path=self._last_image_path or "", last_classification=self.last_classification)


# ---------- App entry ----------
if __name__ == '__main__':
    root = tk.Tk()
    # make sure we're not starting maximized/fullscreen
    root.attributes("-fullscreen", False)
    try:
        root.state("normal")           # Windows/Linux
        root.attributes("-zoomed", False)  # some Windows builds
    except Exception:
        pass

    place_window(root, w_ratio=0.5, h_ratio=0.5, anchor="center")
    root.minsize(900, 600)

    root.configure(bg=COLORS['bg_app'])

    home_panel = tk.Frame(root, bg=COLORS['bg_panel'])
    home_panel.pack(fill='both', expand=True)

    def _open_upload():
        # ReviewDialog will be chained inside
        UploadDialog(root, on_continue=None)

    def _help_ident():
        tl = tk.Toplevel(root)
        tl.title("Identify")
        ClassifierApp(tl)

    def _open_library():
        LibraryWindow(root)

    build_front_hub(
        home_panel,
        on_add_item=_open_upload,
        on_help_ident=_help_ident,
        on_open_library=_open_library
    )

    root.mainloop()
