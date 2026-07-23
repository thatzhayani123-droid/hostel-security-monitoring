"""
pipeline.py — ties detection + tracking + scenarios + drawing together.

Runnable as a library function (run_pipeline) or as a CLI:
    python -m app.pipeline --input data/clip.mp4 --output outputs/annotated.mp4 \
        --alerts outputs/alerts.json

Detector backends (--detector):
    yolo   (default) — ultralytics YOLOv8n, person class only. Requires
                        `pip install ultralytics` (auto-downloads yolov8n.pt
                        on first run — needs internet once).
    motion — OpenCV background-subtraction + contour blob detector. Zero
                        extra installs, zero downloads. Weaker than YOLO but
                        works fully offline — use this as a demo-reliability
                        fallback if venue wifi is bad (see README).
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from app.tracker import CentroidTracker
from app.scenarios import ScenarioEngine

SEVERITY_COLOR = {
    "low": (0, 200, 255),      # amber (BGR)
    "medium": (0, 140, 255),   # orange
    "high": (0, 0, 255),       # red
}


# ------------------------------------------------------------------ detectors

class YoloPersonDetector:
    """Wraps ultralytics YOLOv8n, restricted to class 0 (person)."""

    def __init__(self, model_path="yolov8n.pt", conf=0.35, imgsz=640):
        from ultralytics import YOLO  # imported lazily so `motion` backend needs no torch
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        results = self.model.predict(
            frame, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        boxes = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0]) if b.conf is not None else 0.5
                boxes.append((x1, y1, x2 - x1, y2 - y1, conf))
        return boxes


class MotionBlobDetector:
    """
    Background-subtraction based person-shaped blob detector. No downloads,
    no torch/ultralytics required — works fully offline. Intended as a
    demo-reliability fallback and for pipeline self-testing; YOLO is the
    primary/production detector per spec.
    """

    def __init__(self, min_area=700, max_area=60000, min_aspect=1.1, history=200):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=25, detectShadows=True
        )
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect  # loosely favor person-ish (taller than wide) blobs
        self._warmed = 0

    def detect(self, frame):
        fgmask = self.bg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fgmask = cv2.dilate(fgmask, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._warmed += 1
        boxes = []
        if self._warmed < 15:  # let the background model warm up first
            return boxes
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((float(x), float(y), float(w), float(h), 0.5))
        return boxes


def build_detector(name, **kwargs):
    if name == "yolo":
        return YoloPersonDetector(**kwargs)
    if name == "motion":
        return MotionBlobDetector(**kwargs)
    raise ValueError(f"Unknown detector backend: {name}")


# ------------------------------------------------------------------ drawing

def draw_polygon(frame, polygon, color, label=None):
    if not polygon:
        return
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    if label:
        x, y = polygon[0]
        cv2.putText(frame, label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)


def draw_tracks(frame, tracks):
    for tid, track in tracks.items():
        x, y, w, h = [int(v) for v in track.bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2, cv2.LINE_AA)


def draw_alert_banners(frame, active_banners):
    """active_banners: list of (text, severity) to render at top of frame this frame."""
    y = 28
    for text, severity in active_banners:
        color = SEVERITY_COLOR.get(severity, (0, 200, 255))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (8, y - th - 8), (16 + tw, y + 6), color, -1)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += th + 16


# ------------------------------------------------------------------ pipeline

def run_pipeline(
    input_path,
    output_path,
    alerts_path,
    detector_name="yolo",
    config=None,
    banner_hold_frames=None,
    progress_cb=None,
):
    """
    Runs the full detect -> track -> score-scenarios -> annotate loop over
    `input_path` (file path or RTSP url), writes an annotated video to
    `output_path` and the alert log to `alerts_path`.

    Returns: dict summary {alert_count, scenario_counts, frames_processed, fps, duration_sec}
    """
    config = config or {}
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-2:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    banner_hold_frames = banner_hold_frames or int(fps * 2.5)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(alerts_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = build_detector(detector_name)
    tracker = CentroidTracker(
        max_disappeared=config.get("max_disappeared", 20),
        max_distance=config.get("max_distance", 90),
        history_seconds=max(20, config.get("loiter_time_thresh", 7) + 5),
        fps=fps,
    )
    engine = ScenarioEngine(config)

    all_alerts = []
    banner_queue = []  # list of [text, severity, frames_left]
    frame_num = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_num / fps

        raw_boxes = detector.detect(frame)
        # tracker expects (x, y, w, h) without confidence
        det_boxes = [(b[0], b[1], b[2], b[3]) for b in raw_boxes]
        tracks = tracker.update(det_boxes, frame_num, timestamp)

        alerts = engine.check_all(tracks, frame_num, timestamp)
        for a in alerts:
            all_alerts.append(a.to_dict())
            banner_queue.append([
                f"[{a.severity.upper()}] {a.scenario}: {a.note}", a.severity, banner_hold_frames
            ])

        # draw
        for zone in engine.restricted_zones:
            draw_polygon(frame, zone.get("polygon"), (0, 0, 255), zone.get("name", "restricted"))
        if engine.entry_zone:
            draw_polygon(frame, engine.entry_zone, (255, 200, 0), "entry")
        draw_tracks(frame, tracks)

        banner_queue = [b for b in banner_queue if b[2] > 0]
        for b in banner_queue:
            b[2] -= 1
        draw_alert_banners(frame, [(b[0], b[1]) for b in banner_queue[-4:]])

        writer.write(frame)
        frame_num += 1
        if progress_cb and frame_num % 30 == 0:
            progress_cb(frame_num)

    cap.release()
    writer.release()

    with open(alerts_path, "w") as f:
        json.dump(all_alerts, f, indent=2)

    scenario_counts = {}
    for a in all_alerts:
        scenario_counts[a["scenario"]] = scenario_counts.get(a["scenario"], 0) + 1

    return {
        "alert_count": len(all_alerts),
        "scenario_counts": scenario_counts,
        "frames_processed": frame_num,
        "fps": fps,
        "duration_sec": round(frame_num / fps, 2),
    }


# ------------------------------------------------------------------ CLI

def _parse_zone_arg(raw):
    """Accepts a JSON string or a path to a JSON file describing a polygon list."""
    if raw is None:
        return None
    if os.path.isfile(raw):
        with open(raw) as f:
            return json.load(f)
    return json.loads(raw)


def main():
    p = argparse.ArgumentParser(description="Hostel Security Monitoring pipeline")
    p.add_argument("--input", required=True, help="Video file path or RTSP URL")
    p.add_argument("--output", default="outputs/annotated.mp4")
    p.add_argument("--alerts", default="outputs/alerts.json")
    p.add_argument("--detector", choices=["yolo", "motion"], default="yolo")
    p.add_argument("--restricted-zones", default=None,
                    help='JSON: [{"name":"Roof","polygon":[[x,y],...],"curfew_only":true}]')
    p.add_argument("--entry-zone", default=None, help='JSON polygon: [[x,y],[x,y],...]')
    p.add_argument("--curfew-active", action="store_true",
                    help="Force curfew-hour severity boost on for this run (demo control)")
    p.add_argument("--loiter-seconds", type=float, default=7.0)
    p.add_argument("--loiter-radius-px", type=float, default=60)
    p.add_argument("--tailgate-window-sec", type=float, default=2.0)
    p.add_argument("--crowd-min-people", type=int, default=3)
    p.add_argument("--crowd-radius-px", type=float, default=120)
    args = p.parse_args()

    config = {
        "restricted_zones": _parse_zone_arg(args.restricted_zones) or [],
        "entry_zone": _parse_zone_arg(args.entry_zone),
        "curfew_active": args.curfew_active,
        "loiter_time_thresh": args.loiter_seconds,
        "loiter_radius_px": args.loiter_radius_px,
        "tailgate_window_sec": args.tailgate_window_sec,
        "crowd_min_people": args.crowd_min_people,
        "crowd_radius_px": args.crowd_radius_px,
    }

    t0 = time.time()
    summary = run_pipeline(args.input, args.output, args.alerts, args.detector, config)
    elapsed = time.time() - t0

    print(json.dumps(summary, indent=2))
    print(f"\nProcessed {summary['frames_processed']} frames in {elapsed:.1f}s "
          f"({summary['frames_processed'] / max(elapsed, 1e-6):.1f} fps)")
    print(f"Annotated video -> {args.output}")
    print(f"Alerts JSON     -> {args.alerts}")


if __name__ == "__main__":
    main()
