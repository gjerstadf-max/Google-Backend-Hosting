import os
import json
import hashlib
import time
import logging
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Local .env for dev only
if os.environ.get("ENV") != "CLOUD":
    from dotenv import load_dotenv
    load_dotenv()

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
)

app = FastAPI()

# Optional CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache
cache = {}

# Middleware for request logging, latency, and request ID
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    response = await call_next(request)
    process_time = round(time.time() - start_time, 4)
    log_data = {
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time": process_time
    }
    logging.info(json.dumps(log_data))
    # Attach request_id to response headers
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/summarize")
def summarize(payload: dict, request: Request):
    from tavily import TavilyClient
    from openai import OpenAI

    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not tavily_key or not openai_key:
        logging.error(json.dumps({"request_id": request_id, "error": "Missing API keys"}))
        raise HTTPException(status_code=500, detail="Missing API keys")

    query = payload.get("query")
    if not query:
        logging.warning(json.dumps({"request_id": request_id, "warning": "Missing query"}))
        raise HTTPException(status_code=400, detail="Missing query")

    key = hashlib.md5(query.encode("utf-8")).hexdigest()
    if key in cache:
        logging.info(json.dumps({"request_id": request_id, "cache_hit": True, "query": query}))
        return cache[key]

    logging.info(json.dumps({"request_id": request_id, "cache_hit": False, "query": query}))

    try:
        tavily = TavilyClient(api_key=tavily_key)
        client = OpenAI(api_key=openai_key)

        search = tavily.search(query=query, max_results=5)
        content = "\n".join(r["content"] for r in search["results"])

        prompt = f"""
Return ONLY valid JSON in this format:

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

        raw = response.choices[0].message.content.strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        result = json.loads(raw)

        cache[key] = result

        logging.info(json.dumps({"request_id": request_id, "status": "success", "query": query}))
        return result

    except Exception as e:
        logging.exception(json.dumps({"request_id": request_id, "error": str(e), "query": query}))
        raise HTTPException(status_code=500, detail="Internal server error")

# Cloud Run entry
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
