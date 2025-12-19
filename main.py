from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def health():
    return {"status": "If you see this, the cloud is finally working!"}
