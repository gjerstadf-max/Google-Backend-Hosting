from fastapi import FastAPI
from pydantic import BaseModel
import os
from tavily import TavilyClient
from openai import OpenAI
import json
import re

app = FastAPI()

# --- API clients ---
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Health endpoint ---
@app.get("/")
def health():
    return {"status": "Cloud Run is alive"}

# --- Request body model ---
class SummarizeRequest(BaseModel):
    query: str

# --- Helper: Chunk text to avoid token overflow ---
def chunk_text(text: str, max_chars: int = 4000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

# --- Summarize endpoint ---
@app.post("/summarize")
def summarize(req: SummarizeRequest):
    query = req.query

    # 1️⃣ Search Tavily
    search_results = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    documents = [
        r.get("content", "")
        for r in search_results.get("results", [])
        if r.get("content")
    ]

    if not documents:
        return {"error": "No content found"}

    combined_text = "\n\n".join(documents)
    chunks = chunk_text(combined_text)

    # 2️⃣ Summarize each chunk
    partial_summaries = []
    for chunk in chunks:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial and public sentiment analyst.\n"
                        "Summarize the following text into concise bullet points, separating positives, negatives, and risks. "
                        "Keep it short and clear, output valid JSON only."
                    )
                },
                {"role": "user", "content": chunk}
            ],
            max_tokens=250
        )
        partial_summaries.append(resp.choices[0].message.content.strip())

    # 3️⃣ Combine partial summaries and generate final JSON
    final_input = "\n\n".join(partial_summaries)

    final_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Combine the following summaries into a single structured JSON object:\n"
                    "- sentiment: Bullish, Bearish, Neutral, or Mixed\n"
                    "- positives: array of concise bullet points\n"
                    "- negatives: array of concise bullet points\n"
                    "- risks: array of concise bullet points\n"
                    "- summary: 3–5 sentence executive summary\n"
                    "Output JSON ONLY inside triple backticks, no extra text."
                )
            },
            {"role": "user", "content": final_input}
        ],
        max_tokens=400
    )

    # 4️⃣ Extract JSON reliably
    raw_output = final_resp.choices[0].message.content
    match = re.search(r"```(?:json)?\s*(.*)```", raw_output, re.DOTALL)_
