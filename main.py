from fastapi import FastAPI
import os

app = FastAPI()

import os

tavily_key = os.environ.get("TAVILY_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

@app.get("/")
def health():
    return {"status": "If you see this, the cloud is finally working!"}
