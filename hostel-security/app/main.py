"""
main.py — FastAPI app.

GET  /                serves static/index.html (dashboard)
POST /api/process      accepts a video upload (+ optional zone/threshold form
                        fields), runs the full pipeline, returns URLs to the
                        annotated video + alerts JSON + summary counts.
GET  /outputs/*         serves generated videos + alert JSON
GET  /static/*          serves dashboard assets

Run with:
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import json
import os
import shutil
import time
import traceback
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import run_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Hostel Security Monitoring")

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _safe_json(raw, default):
    if raw is None or raw.strip() == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    detector: str = Form("yolo"),
    restricted_zones: str = Form(None),
    entry_zone: str = Form(None),
    curfew_active: bool = Form(False),
    loiter_seconds: float = Form(7.0),
    loiter_radius_px: float = Form(60.0),
    tailgate_window_sec: float = Form(2.0),
    crowd_min_people: int = Form(3),
    crowd_radius_px: float = Form(120.0),
):
    job_id = uuid.uuid4().hex[:10]
    ext = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    input_path = os.path.join(DATA_DIR, f"{job_id}{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_annotated.mp4")
    alerts_path = os.path.join(OUTPUT_DIR, f"{job_id}_alerts.json")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    config = {
        "restricted_zones": _safe_json(restricted_zones, []),
        "entry_zone": _safe_json(entry_zone, None),
        "curfew_active": curfew_active,
        "loiter_time_thresh": loiter_seconds,
        "loiter_radius_px": loiter_radius_px,
        "tailgate_window_sec": tailgate_window_sec,
        "crowd_min_people": crowd_min_people,
        "crowd_radius_px": crowd_radius_px,
    }

    try:
        t0 = time.time()
        summary = run_pipeline(input_path, output_path, alerts_path, detector, config)
        summary["elapsed_sec"] = round(time.time() - t0, 2)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()[-2000:]},
        )

    return {
        "job_id": job_id,
        "video_url": f"/outputs/{os.path.basename(output_path)}",
        "alerts_url": f"/outputs/{os.path.basename(alerts_path)}",
        **summary,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
