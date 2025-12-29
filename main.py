import os
import json
import hashlib
import time
import logging
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI()

# -------------------------------------------------
# Local .env (dev only)
# -------------------------------------------------
if os.environ.get("ENV") != "CLOUD":
    from dotenv import load_dotenv
    load_dotenv()

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Models
# -------------------------------------------------
class SummarizeRequest(BaseModel):
    query: str

# -------------------------------------------------
# Cache
# -------------------------------------------------
cache = {}

# -------------------------------------------------
# Middleware
# -------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()

    response = await call_next(request)

    duration = round(time.time() - start, 4)
    response.headers["X-Request-ID"] = request_id

    logging.info(json.dumps({
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration": duration
    }))

    return response

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/summarize")
def summarize(req: SummarizeRequest, request: Request):
    from tavily import TavilyClient
    from openai import OpenAI

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not tavily_key or not openai_key:
        raise HTTPException(status_code=500, detail="Missing API keys")

    cache_key = hashlib.md5(query.encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    try:
        tavily = TavilyClient(api_key=tavily_key)
        client = OpenAI(api_key=openai_key)

        search = tavily.search(query=query, max_results=5)
        content = "\n".join(r["content"] for r in search["results"])

        prompt = f"""
Return ONLY valid JSON:

{{
  "summary": "brief summary",
  "sentiment": {{
    "positive": [],
    "negative": [],
    "risks": [],
    "overall_sentiment": "Positive | Neutral | Negative",
    "numeric_sentiment": 0.0,
    "confidence": 0.0
  }}
}}

Content:
{content}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        raw = response.choices[0].message.content
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        result = json.loads(raw)

        cache[cache_key] = result
        return result

    except Exception as e:
        logging.exception(str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# -------------------------------------------------
# Entry
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
