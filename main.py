from fastapi import FastAPI, Query
import os
from tavily import TavilyClient
from openai import OpenAI

app = FastAPI()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def health():
    return {"status": "Cloud Run is alive"}

@app.get("/summarize")
def summarize(query: str = Query(..., min_length=3)):
    # 1. Search using Tavily
    search_results = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    # 2. Combine content
    documents = []
    for r in search_results["results"]:
        if "content" in r:
            documents.append(r["content"])

    if not documents:
        return {"error": "No content found"}

    combined_text = "\n\n".join(documents)

    # 3. Summarize with OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following information clearly and concisely."},
            {"role": "user", "content": combined_text}
        ]
    )

    summary = response.choices[0].message.content

    return {
        "query": query,
        "summary": summary
    }
