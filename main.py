import os
import json
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai import OpenAI
from tavily import TavilyClient

# --------------------------------------------------
# Environment
# --------------------------------------------------
if os.getenv("ENV") != "CLOUD":
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not OPENAI_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY or TAVILY_API_KEY")

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("api")

# --------------------------------------------------
# Clients
# --------------------------------------------------
llm = OpenAI(api_key=OPENAI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="AI Restaurant Analysis API")

# --------------------------------------------------
# In-memory session store (local dev)
# NOTE: For production you’d move this to Redis / DB.
# --------------------------------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 60  # 1 hour


# --------------------------------------------------
# Models
# --------------------------------------------------
QuestionType = Literal["boolean", "choice", "multi_choice", "text"]

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

    # multi-turn state
    session_id: Optional[str] = None

    # answers from follow-ups
    answers: Dict[str, Any] = Field(default_factory=dict)


class RestaurantResult(BaseModel):
    name: str
    summary: str
    sources: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    # Backward compatible fields
    headline: str = ""
    restaurants: List[RestaurantResult] = Field(default_factory=list)
    confidence: int = 80
    follow_up_questions: List[str] = Field(default_factory=list)

    # New fields
    status: Literal["complete", "needs_clarification"] = "complete"
    session_id: str
    follow_up: List[FollowUpQuestion] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _now() -> float:
    return time.time()


def cleanup_sessions():
    cutoff = _now() - SESSION_TTL_SECONDS
    to_delete = [sid for sid, s in SESSIONS.items() if s.get("updated_at", 0) < cutoff]
    for sid in to_delete:
        del SESSIONS[sid]


def get_or_create_session(session_id: Optional[str]) -> str:
    cleanup_sessions()
    if session_id and session_id in SESSIONS:
        return session_id
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "created_at": _now(),
        "updated_at": _now(),
        "query": "",
        "preferences": {},
        "answers": {},
    }
    return sid


def llm_call(system: str, user: str, temperature: float = 0.2) -> str:
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def safe_json_from_llm(raw: str, stage: str) -> dict:
    if not raw or not raw.strip():
        raise HTTPException(status_code=500, detail=f"{stage} returned empty output")

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise HTTPException(status_code=500, detail=f"{stage} returned non-JSON output")

    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"{stage} JSON parsing failed")


def normalize_preferences(prefs: Dict[str, Any], answers: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(prefs or {})
    for k, v in (answers or {}).items():
        merged[k] = v
    return merged


def is_broad_restaurant_query(query: str) -> bool:
    q = query.lower()
    broad_food = any(w in q for w in ["pizza", "restaurants", "food", "best places", "where to eat"])
    broad_place = any(w in q for w in ["rome", "italy", "paris", "london", "new york", "seattle", "tokyo", "barcelona"])
    return broad_food and broad_place


def build_search_query(query: str, prefs: Dict[str, Any]) -> str:
    hints = []
    if prefs.get("budget"):
        hints.append(f"price {prefs['budget']}")
    if prefs.get("open_late") is True:
        hints.append("open late")
    if prefs.get("kid_friendly") is True:
        hints.append("kid friendly")
    if prefs.get("style"):
        hints.append(str(prefs["style"]))
    if prefs.get("neighborhood"):
        hints.append(f"near {prefs['neighborhood']}")
    if prefs.get("sit_down") is True:
        hints.append("sit-down")
    if prefs.get("takeout") is True:
        hints.append("takeout")
    return f"{query} " + " ".join(hints) if hints else query


def progressive_questions(query: str, prefs: Dict[str, Any], extracted_count: int) -> List[FollowUpQuestion]:
    qlist: List[FollowUpQuestion] = []
    ql = query.lower()

    if not (is_broad_restaurant_query(query) or extracted_count >= 7):
        return []

    # Step 1: style
    if "style" not in prefs:
        if "pizza" in ql:
            qlist.append(FollowUpQuestion(
                id="style",
                label="What style are you after?",
                type="choice",
                options=["Roman thin crust", "Neapolitan", "Pizza al taglio (by the slice)", "No preference"],
                help="Rome has distinct pizza styles—this is the biggest driver of recommendations.",
                step=1
            ))
        else:
            qlist.append(FollowUpQuestion(
                id="style",
                label="What kind of place do you want?",
                type="choice",
                options=["Local classic", "Trendy/modern", "Quick bite", "No preference"],
                step=1
            ))
        return qlist

    # Step 2: budget
    if "budget" not in prefs:
        qlist.append(FollowUpQuestion(
            id="budget",
            label="What price range matters most?",
            type="choice",
            options=["budget", "mid", "high", "any"],
            help="Helps filter tourist traps vs great value.",
            step=2
        ))
        return qlist

    # Step 2b: open late
    if "open_late" not in prefs:
        qlist.append(FollowUpQuestion(
            id="open_late",
            label="Is being open late important?",
            type="boolean",
            help="Late-night options can change the shortlist a lot.",
            step=2
        ))
        return qlist

    # Step 3: kid friendly
    if "kid_friendly" not in prefs:
        qlist.append(FollowUpQuestion(
            id="kid_friendly",
            label="Does it need to be kid-friendly?",
            type="boolean",
            step=3
        ))
        return qlist

    # Step 3b: neighborhood (optional)
    if "neighborhood" not in prefs:
        qlist.append(FollowUpQuestion(
            id="neighborhood",
            label="Any neighborhood / landmark to stay near?",
            type="text",
            help="e.g., Trastevere, Termini, Vatican, Pantheon. Leave blank if you don’t care.",
            step=3
        ))
        return qlist

    return []


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    sid = get_or_create_session(req.session_id)

    session = SESSIONS[sid]
    session["updated_at"] = _now()
    session["query"] = req.query or session.get("query", "")
    session["preferences"] = dict(req.preferences or session.get("preferences", {}) or {})
    session["answers"] = dict(req.answers or session.get("answers", {}) or {})

    merged_prefs = normalize_preferences(session["preferences"], session["answers"])
    session["preferences"] = merged_prefs

    logger.info(f"[{sid}] Analyze: {session['query']} | prefs={merged_prefs}")

    # -------------------------
    # Stage 1 — Search
    # -------------------------
    search_query = build_search_query(session["query"], merged_prefs)
    search = tavily.search(query=f"best restaurants {search_query}", max_results=10)

    results = search.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="No search results")

    urls = [r.get("url") for r in results if r.get("url")]
    content_blocks = [r.get("content") for r in results if r.get("content")]
    if not content_blocks:
        raise HTTPException(status_code=400, detail="No usable content")

    context = "\n\n".join(content_blocks[:7])

    # -------------------------
    # Stage 2 — Extraction
    # -------------------------
    extract_prompt = f"""
Extract restaurant names from the text below. Include short factual notes.
Return JSON ONLY:

{{
  "restaurants": [
    {{
      "name": "Restaurant name",
      "facts": ["short fact 1", "short fact 2"]
    }}
  ]
}}

TEXT:
{context}
"""
    raw_extract = llm_call(
        system="You extract factual restaurant entities from text. Return JSON only.",
        user=extract_prompt,
        temperature=0.0,
    )
    extracted = safe_json_from_llm(raw_extract, "Extraction")
    extracted_list = extracted.get("restaurants", [])
    if not isinstance(extracted_list, list) or not extracted_list:
        raise HTTPException(status_code=400, detail="No restaurants found")

    normalized = []
    seen = set()
    for item in extracted_list:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        facts = item.get("facts", [])
        if isinstance(facts, str):
            facts = [facts]
        if not isinstance(facts, list):
            facts = []
        facts = [str(f).strip() for f in facts if str(f).strip()]
        normalized.append({"name": name, "facts": facts[:6]})

    if not normalized:
        raise HTTPException(status_code=400, detail="No restaurants found")

    # -------------------------
    # Progressive follow-up
    # -------------------------
    followup_struct = progressive_questions(session["query"], merged_prefs, extracted_count=len(normalized))
    if followup_struct:
        followup_strings = [q.label for q in followup_struct]
        return AnalyzeResponse(
            status="needs_clarification",
            session_id=sid,
            preferences=merged_prefs,
            follow_up=followup_struct,
            follow_up_questions=followup_strings,
            headline="",
            restaurants=[],
            confidence=80,
        )

    # -------------------------
    # Stage 3 — Synthesis
    # -------------------------
    prefs_json = json.dumps(merged_prefs, ensure_ascii=False, indent=2)

    synth_prompt = f"""
You are a professional restaurant critic.

User query:
{session["query"]}

User preferences (JSON):
{prefs_json}

Restaurants & extracted notes:
{json.dumps(normalized, ensure_ascii=False, indent=2)}

Write:
- A strong headline (NOT the same as the summaries)
- A short, high-quality summary for each restaurant (2–4 sentences), tailored to preferences

Return JSON ONLY:
{{
  "headline": "string",
  "restaurants": [
    {{
      "name": "string",
      "summary": "string"
    }}
  ],
  "confidence": 0-100
}}
"""
    raw_synth = llm_call(
        system="You write polished restaurant recommendations grounded in the extracted notes. Return JSON only.",
        user=synth_prompt,
        temperature=0.3,
    )
    final = safe_json_from_llm(raw_synth, "Synthesis")

    headline = (final.get("headline") or "Restaurant analysis").strip()
    restaurants_out = final.get("restaurants", [])
    if not isinstance(restaurants_out, list) or not restaurants_out:
        raise HTTPException(status_code=500, detail="LLM synthesis returned no restaurants")

    confidence = final.get("confidence", 80)
    try:
        confidence = int(confidence)
    except Exception:
        confidence = 80
    confidence = max(0, min(100, confidence))

    top_sources = list(dict.fromkeys([u for u in urls if u]))[:6]

    enriched: List[RestaurantResult] = []
    for r in restaurants_out:
        if not isinstance(r, dict):
            continue
        name = (r.get("name") or "").strip()
        summary = (r.get("summary") or "").strip()
        if not name or not summary:
            continue
        enriched.append(RestaurantResult(
            name=name,
            summary=summary,
            sources=top_sources[:3],
        ))

    if not enriched:
        raise HTTPException(status_code=500, detail="LLM synthesis produced invalid restaurant entries")

    return AnalyzeResponse(
        status="complete",
        session_id=sid,
        preferences=merged_prefs,
        headline=headline,
        restaurants=enriched,
        confidence=confidence,
        follow_up_questions=[],
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
