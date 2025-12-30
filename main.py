import os
import json
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env locally
if os.getenv("ENV") != "CLOUD":
    load_dotenv()

from openai import OpenAI
from tavily import TavilyClient

# ---- logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger("api")

# ---- app ----
app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"→ {response.status_code}")
    return response

# ---- keys ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is missing")

client = OpenAI(api_key=OPENAI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ---- models ----
class SummarizeRequest(BaseModel):
    query: str
    role: str = "financial analyst"

class AnalysisResponse(BaseModel):
    headline: str
    detail: str
    key_points: list[str]
    sentiment: str

# ---- routes ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize", response_model=AnalysisResponse)
def summarize(req: SummarizeRequest):
    try:
        logger.info(f"Searching for: {req.query}")

        search = tavily.search(
            query=req.query,
            max_results=5,
        )

        # Build strong context
        results = search.get("results", [])[:3]  # top 3 only

        context = "\n\n".join(
    f"Source {i+1}: {r['content'][:800]}"
    for i, r in enumerate(results)
)

        if not context.strip():
            raise ValueError("No search content returned")

        prompt = f"""
Return ONLY valid JSON.

STRICT RULE:
- Headline must NOT reuse phrases from Detail 1
- Detail: deeper explanation, NOT repeating headline
- Provide bullet-style key points
- Assign sentiment

JSON format:
{{
  "headline": "...",
  "detail": "...",
  "key_points": ["...", "..."],
  "sentiment": "Positive | Neutral | Negative"
}}

Content:
{context}
"""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a {req.role}."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
        )

        raw = completion.choices[0].message.content.strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]

        result = json.loads(raw)

        logger.info("Summarization successful")
        return result

    except Exception as e:
        logger.exception("Summarization failed")
        raise HTTPException(status_code=500, detail=str(e))
