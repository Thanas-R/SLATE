"""
server.py
FastAPI backend for SLATE.
Serves the minimal monochrome Obsidian-style interface and handles API requests.
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from core.pipeline import SLATEPipeline

app = FastAPI(title="SLATE API", version="2.0.0")

# Global pipeline instance cached for quick responses
pipeline = SLATEPipeline(hidden_dim=8)
cached_results = None


class ExperimentParams(BaseModel):
    hidden_dim: int = 8
    recurrent_scale: float = 1.0


@app.on_event("startup")
def startup_event():
    global cached_results
    if cached_results is None:
        cached_results = pipeline.run_experiment()


@app.get("/api/pipeline")
def get_pipeline_data():
    global cached_results
    if cached_results is None:
        cached_results = pipeline.run_experiment()
    return JSONResponse(content=cached_results)


@app.post("/api/run-experiment")
def run_experiment(params: ExperimentParams):
    global pipeline, cached_results
    pipeline = SLATEPipeline(hidden_dim=params.hidden_dim)
    cached_results = pipeline.run_experiment(recurrent_scale=params.recurrent_scale)
    return JSONResponse(content=cached_results)


# Serve static assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SLATE: Web assets not found.</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=True)
