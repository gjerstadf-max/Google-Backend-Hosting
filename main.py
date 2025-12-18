import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel

# 1. Initialize FastAPI
app = FastAPI()

# Enable CORS so your website can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your website URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define the Search Request Format
class SearchQuery(BaseModel):
    location: str
    cuisine: str
    vibe: str

# 3. The Search Logic
@app.post("/search-restaurants")
async def find_places(query: SearchQuery):
    # Load keys from Environment Variables (Set these in Google Cloud Console)
    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Initialize the Search Tool
    search = TavilySearchResults(api_key=tavily_key, max_results=5)
    
    # Construct a smart search query for the Agent
    search_prompt = f"Best {query.cuisine} restaurants in {query.location} with a {query.vibe} vibe. Find honest reviews from Reddit and food blogs."
    
    # Get raw web results
    web_results = search.run(search_prompt)

    # Use LLM to summarize the findings
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    
    summary_prompt = f"""
    Based on these web results: {web_results}
    
    Provide a list of the 3 best restaurants matching: {query.cuisine} in {query.location}.
    For each, include:
    1. Name
    2. Why it matches the '{query.vibe}' vibe
    3. One 'Pro' and one 'Con' found in reviews.
    """
    
    ai_response = llm.predict(summary_prompt)

    return {"results": ai_response}

@app.get("/")
def health_check():
    return {"status": "alive"}
