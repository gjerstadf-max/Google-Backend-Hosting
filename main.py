import os
import json
import uuid
import time
import logging
import re
import urllib.parse
from typing import Dict, List, Any, Optional, Literal, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai import OpenAI
from tavily import TavilyClient

# ==================================================
# Environment
# ==================================================
if os.getenv("ENV") != "CLOUD":
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not OPENAI_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY or TAVILY_API_KEY")

# ==================================================
# Logging
# ==================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("api")

# ==================================================
# Clients
# ==================================================
llm = OpenAI(api_key=OPENAI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ==================================================
# App
# ==================================================
app = FastAPI(title="Restaurant Analysis API")

# ==================================================
# Session store (local dev)
# ==================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = 3600


def now() -> float:
    return time.time()


def get_session(sid: Optional[str]) -> str:
    cutoff = now() - SESSION_TTL
    for k in list(SESSIONS.keys()):
        if SESSIONS[k].get("updated", 0) < cutoff:
            del SESSIONS[k]

    if sid and sid in SESSIONS:
        SESSIONS[sid]["updated"] = now()
        return sid

    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"created": now(), "updated": now()}
    return sid


# ==================================================
# Models
# ==================================================
QuestionType = Literal["boolean", "choice", "text"]


class FollowUpQuestion(BaseModel):
    id: str
    label: str
    type: QuestionType
    options: Optional[List[str]] = None
    help: Optional[str] = None
    step: Optional[int] = None


class AnalyzeRequest(BaseModel):
    query: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    answers: Dict[str, Any] = Field(default_factory=dict)


class RestaurantResult(BaseModel):
    name: str
    summary: str
    sources: List[str] = Field(default_factory=list)
    maps_url: str


class AnalyzeResponse(BaseModel):
    status: Literal["complete", "needs_clarification"]
    session_id: str
    headline: str = ""
    restaurants: List[RestaurantResult] = Field(default_factory=list)
    confidence: int = 80
    follow_up: List[FollowUpQuestion] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)


# ==================================================
# Helpers
# ==================================================
def llm_call(system: str, user: str, temperature: float = 0.2) -> str:
    r = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
    )
    return (r.choices[0].message.content or "").strip()


def safe_json(raw: str, stage: str) -> dict:
    if not raw or not raw.strip():
        raise HTTPException(500, f"{stage} returned empty output")
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise HTTPException(500, f"{stage} returned invalid JSON")
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        raise HTTPException(500, f"{stage} JSON parsing failed")


def google_maps_search_url(name: str, query: str) -> str:
    q = f"{name} {query}"
    return "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote(q)


def normalize_preferences(prefs: Dict[str, Any], answers: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(prefs or {})
    for k, v in (answers or {}).items():
        if isinstance(v, str) and v.strip().lower() in ["no preference", "any", ""]:
            continue
        merged[k] = v
    return merged


def required_item_from_query(query: str) -> Optional[str]:
    q = (query or "").lower()
    if "pizza" in q:
        return "pizza"
    if "burger" in q or "hamburger" in q:
        return "burger"
    if "sushi" in q:
        return "sushi"
    return None


def is_broad_city_query(query: str) -> bool:
    q = (query or "").lower()
    food = any(w in q for w in ["pizza", "burger", "sushi", "restaurant", "restaurants", "where to eat", "best"])
    # broad location signal: "in <place>" OR presence of known large places
    loc = (" in " in q) or any(p in q for p in [
        "rome", "italy", "paris", "london", "tokyo", "barcelona", "new york", "nyc", "seattle"
    ])
    return food and loc


def extract_area_options(query: str, context: str) -> List[str]:
    """
    Pulls neighborhoods/landmarks *from the actual search context*.
    If it fails, returns empty list (we fall back to text input).
    """
    try:
        prompt = f"""
User query: {query}

From the text below, extract up to 6 area/neighborhood/landmark names that help narrow where to eat.
Return JSON only:

{{ "areas": ["Trastevere", "Testaccio", "Termini", "Vatican", "Centro Storico"] }}

TEXT:
{context[:7000]}
"""
        raw = llm_call(
            system="Extract area/neighborhood/landmark names. Return JSON only.",
            user=prompt,
            temperature=0.0,
        )
        parsed = safe_json(raw, "Area extraction")
        areas = parsed.get("areas", [])
        if not isinstance(areas, list):
            return []
        out = []
        for a in areas:
            if isinstance(a, str):
                a = a.strip()
                if a and a.lower() not in [x.lower() for x in out]:
                    out.append(a)
        return out[:6]
    except Exception:
        return []


def evidence_mentions_item(text: str, item: str) -> bool:
    blob = (text or "").lower()
    if item == "pizza":
        positives = ["pizza", "pizzeria", "slice", "margherita", "pepperoni", "neapolitan", "roman", "al taglio"]
    elif item == "burger":
        positives = ["burger", "burgers", "hamburger", "cheeseburger", "patty", "smashburger"]
    elif item == "sushi":
        positives = ["sushi", "nigiri", "sashimi", "maki", "roll", "omakase"]
    else:
        positives = [item]
    return any(p in blob for p in positives)


def verify_candidate(name: str, query: str, item: Optional[str]) -> Tuple[bool, List[str], str]:
    """
    Verify: evidence suggests (a) the place exists and (b) serves requested item (if any).
    """
    try:
        vq = f"\"{name}\" {query} menu"
        s = tavily.search(query=vq, max_results=5)
        results = s.get("results", []) or []
        urls = [r.get("url") for r in results if r.get("url")]

        evidence = " ".join(
            ((r.get("title") or "") + " " + (r.get("content") or ""))
            for r in results
        )

        if item:
            ok = evidence_mentions_item(evidence, item)
        else:
            ok = True

        return ok, urls[:3], evidence[:2500]
    except Exception:
        return False, [], ""


# ==================================================
# Routes
# ==================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    sid = get_session(req.session_id)

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "Missing query")

    prefs = normalize_preferences(req.preferences or {}, req.answers or {})
    item = required_item_from_query(query)

    # --------------------------------------------------
    # Follow-up Step 1: deterministic for broad pizza/city queries
    # --------------------------------------------------
    if is_broad_city_query(query) and item == "pizza" and "style" not in prefs:
        q = FollowUpQuestion(
            id="style",
            label="What pizza style are you after?",
            type="choice",
            options=["Roman thin crust", "Neapolitan", "Pizza al taglio (by the slice)", "No preference"],
            help="Rome has distinct styles—this narrows the list fast.",
            step=1,
        )
        return AnalyzeResponse(
            status="needs_clarification",
            session_id=sid,
            preferences=prefs,
            follow_up=[q],
            headline="",
            restaurants=[],
            confidence=80,
        )

    # --------------------------------------------------
    # Stage 1 — Search (item-specific)
    # --------------------------------------------------
    # Use follow-up answers to refine search
    style_hint = ""
    if prefs.get("style") and str(prefs["style"]).lower() != "no preference":
        style_hint = f" {prefs['style']}"

    area_hint = ""
    if prefs.get("area") and str(prefs["area"]).lower() != "no preference":
        area_hint = f" near {prefs['area']}"

    if item:
        search_q = f"best {item}{style_hint} {query}{area_hint}"
    else:
        search_q = f"best restaurants {query}{area_hint}"

    logger.info(f"[{sid}] Search: {search_q}")
    search = tavily.search(query=search_q, max_results=10)
    results = search.get("results", []) or []
    if not results:
        raise HTTPException(400, "No search results")

    content_blocks = [r.get("content") for r in results if r.get("content")]
    if not content_blocks:
        raise HTTPException(400, "No usable content")

    context = "\n\n".join(content_blocks[:7])
    global_urls = [r.get("url") for r in results if r.get("url")]
    global_urls = [u for u in global_urls if u]

    # --------------------------------------------------
    # Follow-up Step 2: “where exactly?” (area/neighborhood)
    # Only ask if still broad and user hasn't narrowed it yet.
    # --------------------------------------------------
    if is_broad_city_query(query) and "area" not in prefs:
        areas = extract_area_options(query, context)
        if areas:
            q = FollowUpQuestion(
                id="area",
                label="Where in the city do you want to eat?",
                type="choice",
                options=areas + ["No preference"],
                help="Picking an area makes recommendations much more actionable.",
                step=2,
            )
        else:
            q = FollowUpQuestion(
                id="area",
                label="Where in the city do you want to eat? (optional)",
                type="text",
                help="Example: Trastevere, Testaccio, Termini, Vatican, Centro Storico",
                step=2,
            )

        return AnalyzeResponse(
            status="needs_clarification",
            session_id=sid,
            preferences=prefs,
            follow_up=[q],
            headline="",
            restaurants=[],
            confidence=80,
        )

    # --------------------------------------------------
    # Stage 2 — Extraction (name + evidence lines)
    # --------------------------------------------------
    extract_prompt = f"""
You are extracting candidates for: {item if item else "restaurants"}.

From the text below, extract up to 12 candidates.
Return JSON ONLY:

{{
  "candidates": [
    {{
      "name": "string",
      "evidence": ["short snippet from the text supporting relevance"]
    }}
  ]
}}

Rules:
- Evidence MUST be derived from the provided text.
- If the query is for "{item}", prefer candidates explicitly connected to "{item}".
- Do NOT invent names.

TEXT:
{context}
"""
    extracted = safe_json(
        llm_call(
            system="Extract candidates grounded in the provided text. Return JSON only.",
            user=extract_prompt,
            temperature=0.0,
        ),
        "Extraction",
    )

    candidates = extracted.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise HTTPException(400, "No candidates extracted")

    # Normalize + de-dup
    norm = []
    seen = set()
    for c in candidates:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ev = c.get("evidence", [])
        if isinstance(ev, str):
            ev = [ev]
        if not isinstance(ev, list):
            ev = []
        ev = [str(x).strip() for x in ev if str(x).strip()]
        norm.append({"name": name, "evidence": ev[:4]})

    if not norm:
        raise HTTPException(400, "No valid candidates extracted")

    # --------------------------------------------------
    # Stage 2.5 — Verification gate (serves item)
    # --------------------------------------------------
    verified = []
    per_sources: Dict[str, List[str]] = {}
    per_evidence: Dict[str, str] = {}

    for c in norm:
        ok, urls, ev_blob = verify_candidate(c["name"], query, item)
        if ok:
            verified.append(c)
            per_sources[c["name"]] = urls
            per_evidence[c["name"]] = ev_blob

    if item and not verified:
        return AnalyzeResponse(
            status="needs_clarification",
            session_id=sid,
            preferences=prefs,
            follow_up=[
                FollowUpQuestion(
                    id="broaden_search",
                    label=f"I couldn’t verify clear {item} options for “{query}”. What should I do?",
                    type="choice",
                    options=["Broaden nearby", "Show general restaurants", "Try a different query"],
                    help="This prevents recommending places that don't serve what you asked for.",
                )
            ],
            headline="",
            restaurants=[],
            confidence=70,
        )

    working = verified if item else norm
    working = working[:8]

    # --------------------------------------------------
    # Stage 3 — Evidence-grounded synthesis
    # --------------------------------------------------
    synth_input = []
    for c in working:
        name = c["name"]
        synth_input.append({
            "name": name,
            "extracted_evidence": c["evidence"],
            "verification_snippet": per_evidence.get(name, ""),
            "sources": per_sources.get(name, global_urls[:2]),
        })

    synth_prompt = f"""
You are a careful restaurant recommender.

User query: {query}
Requested item (if any): {item}
Preferences: {json.dumps(prefs, ensure_ascii=False)}

Rules:
- Only make claims supported by the evidence provided.
- Do not invent menu items, hours, or pricing.
- Keep each summary 2–4 sentences.
- Headline must NOT repeat the same text as a summary.

Return JSON ONLY:
{{
  "headline": "string",
  "restaurants": [
    {{
      "name": "string",
      "summary": "string",
      "sources": ["url1","url2"]
    }}
  ],
  "confidence": 0-100
}}

CANDIDATES WITH EVIDENCE:
{json.dumps(synth_input, ensure_ascii=False, indent=2)}
"""
    final = safe_json(
        llm_call(
            system="Write grounded recommendations from evidence only. Return JSON only.",
            user=synth_prompt,
            temperature=0.2,
        ),
        "Synthesis",
    )

    headline = (final.get("headline") or "Restaurant recommendations").strip()
    out_list = final.get("restaurants", [])
    if not isinstance(out_list, list) or not out_list:
        raise HTTPException(500, "Synthesis produced no restaurants")

    restaurants_out: List[RestaurantResult] = []
    for r in out_list:
        if not isinstance(r, dict):
            continue
        name = (r.get("name") or "").strip()
        summary = (r.get("summary") or "").strip()
        sources = r.get("sources") or per_sources.get(name, []) or global_urls[:2]
        if not name or not summary:
            continue
        if isinstance(sources, str):
            sources = [sources]
        sources = [s for s in sources if isinstance(s, str) and s.startswith("http")]

        restaurants_out.append(
            RestaurantResult(
                name=name,
                summary=summary,
                sources=sources[:3],
                maps_url=google_maps_search_url(name, query if not prefs.get("area") else f"{query} {prefs.get('area')}"),
            )
        )

    conf = final.get("confidence", 80)
    try:
        conf = int(conf)
    except Exception:
        conf = 80
    conf = max(0, min(100, conf))

    return AnalyzeResponse(
        status="complete",
        session_id=sid,
        preferences=prefs,
        headline=headline,
        restaurants=restaurants_out,
        confidence=conf,
        follow_up=[],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=True,
    )
