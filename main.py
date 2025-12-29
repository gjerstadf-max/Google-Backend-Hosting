import os
import json
from fastapi import FastAPI, HTTPException

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/summarize")
def summarize(payload: dict):
    from tavily import TavilyClient
    from openai import OpenAI

    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not tavily_key or not openai_key:
        raise HTTPException(status_code=500, detail="Missing API keys")

    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    tavily = TavilyClient(api_key=tavily_key)
    client = OpenAI(api_key=openai_key)

    search = tavily.search(query=query, max_results=5)
    content = "\n".join(r["content"] for r in search["results"])

    prompt = f"""
Return ONLY valid JSON in this format:

{{
  "summary": "brief summary",
  "sentiment": {{
    "positive": ["..."],
    "negative": ["..."],
    "risks": ["..."],
    "overall_sentiment": "Positive | Neutral | Negative"
    If there are no items, return an empty array.
    "confidence": 0.0  # confidence score between 0 and 1.
    Confidence shoulde be a number between 0 and 1.
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

    return json.loads(raw)
