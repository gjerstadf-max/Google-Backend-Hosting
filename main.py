from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def health():
    return {"status": "The server is finally alive!"}

@app.post("/search-restaurants")
def mock_search(data: dict):
    return {"message": "Server works, AI logic was the problem."}
