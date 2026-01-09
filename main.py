import os
import json
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from tavily import TavilyClient

# -------------------------
# Environment
# -------------------------
if os.getenv("ENV") != "CLOUD":
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY missing")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger("api")

# -------------------------
# Clients
# -------------------------
llm = OpenAI(api_key=OPENAI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="Restaurant Analysis API")

# -------------------------
# Models
# -------------------------
class AnalyzeRequest(BaseModel):
    query: str
    preferences: Optional[List[str]] = []


class RestaurantResult(BaseModel):
    name: str
    summary: str


class AnalyzeResponse(BaseModel):
    headline: str
    restaurants: List[RestaurantResult]
    confidence: int
    sources: List[str]


# -------------------------
# Utilities
# -------------------------
def safe_json_from_llm(raw: str, stage: str) -> dict:
    if not raw or not raw.strip():
        raise HTTPException(
            status_code=500,
            detail=f"{stage} returned empty output"
        )

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        raise HTTPException(
            status_code=500,
            detail=f"{stage} returned non-JSON output"
        )

    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"{stage} JSON parsing failed"
        )


def llm_call(system: str, user: str, temperature: float = 0.2) -> str:
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    logger.info(f"Analyze: {req.query}")

    # -------------------------
    # Stage 1 — Search
    # -------------------------
    search = tavily.search(
        query=f"best restaurants {req.query}",
        max_results=6,
    )

    results = search.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="No search results")

    contents = [r["content"] for r in results if r.get("content")]
    if not contents:
        raise HTTPException(status_code=400, detail="No usable content")

    context = "\n\n".join(contents)
    sources = [r["url"] for r in results if r.get("url")]

    # -------------------------
    # Stage 2 — Fact Extraction
    # -------------------------
    extract_prompt = f"""
Extract restaurants from the text below.

Return JSON only:

{{
  "restaurants": [
    {{
      "name": "Restaurant name",
      "facts": "Key facts"
    }}
  ]
}}

TEXT:
{context}
"""

    raw_extract = llm_call(
        system="You extract structured restaurant facts.",
        user=extract_prompt
    )

    extracted = safe_json_from_llm(raw_extract, "Fact extraction")
    restaurants = extracted.get("restaurants", [])

    if not restaurants:
        raise HTTPException(status_code=400, detail="No restaurants identified")

    # -------------------------
    # Stage 3 — Synthesis
    # -------------------------
    prefs = ", ".join(req.preferences) if req.preferences else "general dining"

    synth_prompt = f"""
You are a restaurant critic.

User preferences: {prefs}

Create a clear headline and a short summary for each restaurant.

Return JSON only:

{{
  "headline": "...",
  "restaurants": [
    {{
      "name": "...",
      "summary": "..."
    }}
  ],
  "confidence": 0-100
}}

Restaurants:
{json.dumps(restaurants, indent=2)}
"""

    raw_synth = llm_call(
        system="You write polished restaurant reviews.",
        user=synth_prompt,
        temperature=0.3
    )

    final = safe_json_from_llm(raw_synth, "LLM synthesis")

    return {
        "headline": final.get("headline", "Restaurant analysis"),
        "restaurants": final.get("restaurants", []),
        "confidence": final.get("confidence", 80),
        "sources": sources,
    }
