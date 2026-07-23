# Hostel Watch — Hostel Security Monitoring

Rule-based adverse-scenario detection from hostel CCTV footage. YOLOv8n person
detection → lightweight numpy centroid tracker → a `ScenarioEngine` with one
method per scenario → annotated video + JSON alert log, served from a small
FastAPI dashboard.

## ⚠️ Test-environment note (read this first)

This was built and unit-tested in a sandboxed environment **with no internet
access** — `pip install`, PyPI, and GitHub were all network-blocked, so I
could not install `ultralytics`/`torch`/`fastapi`/`uvicorn`, download YOLOv8n
weights, or pull a real pedestrian clip there.

What I *did* verify in that sandbox, for real, with real output you can check:
- **All 6 scenarios' decision logic**, directly against `tracker.py` +
  `scenarios.py`, via `tests/test_logic.py` — 6/6 pass, including the crowd
  dedup check (one alert per cluster, not one per person). Full output further
  down in this README.
- **The full video pipeline** (`app/pipeline.py`: video I/O → detect → track →
  score all 6 scenarios → draw overlays → write annotated `.mp4` → write
  `alerts.json`) end-to-end on a real generated video file, using the
  zero-download `motion` detector backend (see below) as a stand-in for
  YOLOv8n. 3 of 6 scenarios fired with real, non-duplicated alerts.
- **`main.py` was NOT boot-tested** — FastAPI/uvicorn aren't installable in
  that sandbox. The code follows the same patterns as `pipeline.py`'s CLI and
  is straightforward FastAPI, but boot it yourself first thing (30 seconds)
  before you demo, per the steps below.

None of this is a substitute for you running it once with real YOLOv8n and a
real clip before the hackathon judging — budget 5 minutes for that. Everything
below is written for that normal, internet-connected run.

## Setup

```bash
cd hostel-security
python3 -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

First run of anything using `--detector yolo` (the default) auto-downloads
`yolov8n.pt` (~6MB) via `ultralytics` — needs internet once, then it's cached.

## Run the dashboard

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Upload a clip, optionally expand "Zones &
thresholds" to paste in polygons for your camera layout (see below), click
**Run Detection**. You'll get an annotated video player, a stat row, and a
live color-coded alert feed (amber = low, orange = medium, red = high).

## Run via CLI only

```bash
python3 -m app.pipeline \
  --input data/your_clip.mp4 \
  --output outputs/annotated.mp4 \
  --alerts outputs/alerts.json \
  --detector yolo \
  --restricted-zones '[{"name":"RoofAccess","polygon":[[380,60],[640,60],[640,240],[380,240]],"curfew_only":true}]' \
  --entry-zone '[[0,190],[260,190],[260,480],[0,480]]' \
  --curfew-active \
  --loiter-seconds 7 \
  --tailgate-window-sec 2.0 \
  --crowd-min-people 3
```

Prints a JSON summary (`alert_count`, `scenario_counts`, frames processed) and
writes the annotated video + alert log.

No real clip handy? OpenCV's own `vtest.avi` pedestrian sample (from the
opencv/opencv GitHub repo, `samples/data/vtest.avi`) is a good generic test —
download it and pass it as `--input`. It won't have curfew/tailgating/fall
moments built in, but it'll exercise crowd formation and general tracking on
real people.

## Editing zone polygons for a different camera layout

Both `--restricted-zones` and `--entry-zone` (CLI), or the matching dashboard
fields, take pixel-coordinate polygons in the source video's native
resolution — **not** normalized 0–1 coordinates.

1. Grab one frame from your camera (`ffmpeg -i clip.mp4 -vframes 1 frame.png`,
   or the first frame the dashboard shows once you upload).
2. Open it in any image viewer that shows pixel coordinates on hover (e.g.
   GIMP, Preview + cursor readout, or just `matplotlib.pyplot.imshow`).
3. Click around the region you want to fence off (a doorway, roof access,
   restricted corridor) and note each corner's `(x, y)`.
4. `--restricted-zones` is a JSON **list** — each entry is
   `{"name": "...", "polygon": [[x,y], ...], "curfew_only": true|false}`.
   `curfew_only: true` zones only escalate to `high` severity when
   `--curfew-active` (CLI) / the "Curfew currently active" checkbox
   (dashboard) is set — see the demo tips below for why that's a flag rather
   than a real clock.
5. `--entry-zone` is a single polygon (`[[x,y], ...]`) — the doorway/turnstile
   area tailgating is measured against.

Points must be given in order (clockwise or counter-clockwise) — don't
scramble the corner order.

## Demoing reliably — 2-3 tips

1. **Pre-record and pre-process your clips before you're on stage.** Run each
   demo clip through the CLI ahead of time, confirm the alert counts look
   right, and have the annotated `.mp4` + `alerts.json` already sitting in
   `outputs/` as a backup. If venue wifi kills your `ultralytics` first-run
   weight download or judges' laptop has no GPU, live-uploading through the
   dashboard becomes a gamble — a pre-baked example isn't.
2. **Use `--detector motion` as a zero-dependency fallback**, not just for
   testing. It needs no model weights and no internet, at the cost of being
   noisier and losing/re-issuing track IDs more than YOLOv8n. If your venue's
   network is unreliable, script your demo around `motion` rather than
   discovering mid-demo that `yolov8n.pt` never finished downloading.
3. **Don't tune thresholds live in front of judges.** `curfew_active`,
   `loiter_seconds`, `crowd_radius_px` etc. are all exposed as flags/form
   fields on purpose — for you to dial in against your actual demo clip
   beforehand, not to explain and adjust in real time. Pick values that make
   your prepared clip trigger cleanly, then leave them alone.

## Scenario reference

| Scenario | Trigger | Default threshold |
|---|---|---|
| `restricted_zone_intrusion` | track centroid enters a restricted polygon | `high` if curfew-only zone + curfew active, else `medium` |
| `loitering` | track's movement stays within a small radius for too long | 7s / 60px |
| `tailgating` | 2+ distinct tracks enter the entry-zone polygon within a short window | 2.0s |
| `fall_detection` | bbox aspect ratio (h/w) flips from >1.4 to <0.8 within a short window | 0.6s |
| `possible_altercation` | 2 tracks close together while both moving fast | 90px / 140px/s |
| `crowd_formation` | 3+ tracks cluster within a radius — one alert per cluster, cooldown-deduped | 3 people / 120px / 8s cooldown |

Every alert (`app/scenarios.py::Alert`) carries `scenario`, `track_id`
(int, or a list for group scenarios), `confidence`, `severity`
(`low`/`medium`/`high`), a human-readable `note`, `frame`, and `timestamp`
(seconds). All 6 have independent per-(scenario, track/cluster) cooldowns so a
persistent condition doesn't spam an alert every frame — see
`ScenarioEngine.alert_cooldowns`.

`ScenarioEngine._vlm_verify(frame, alert)` is a deliberate no-op extension
point — wire a VLM re-check or a LangGraph agentic router there later without
touching any of the 6 `_check_*` methods.

## Actual test run output (from this sandbox, no internet)

### 1. Pure decision-logic test — `python3 tests/test_logic.py` — 6/6 PASS

```
=== restricted_zone_intrusion (curfew) ===
  restricted_zone_intrusion: 1 alert(s)
    - track=0 sev=high conf=0.9 note=Person entered restricted zone 'Roof' during curfew hours
  RESULT: PASS

=== loitering ===
  loitering: 1 alert(s)
    - track=0 sev=low conf=0.8 note=Person stationary in a small area for ~3s (movement radius 9px)
  RESULT: PASS

=== tailgating ===
  tailgating: 1 alert(s)
    - track=[0, 1] sev=medium conf=0.75 note=2 people entered the entry zone within 2.0s of each other
  RESULT: PASS

=== fall_detection ===
  fall_detection: 1 alert(s)
    - track=0 sev=high conf=0.7 note=Bounding box flipped from standing to horizontal within 0.6s
  RESULT: PASS

=== possible_altercation ===
  possible_altercation: 1 alert(s)
    - track=[0, 1] sev=high conf=0.65 note=Persons 0 & 1 in close proximity (29px) while both moving erratically
  RESULT: PASS

=== crowd_formation ===
  crowd_formation: 1 alert(s)
    - track=[0, 1, 2, 3] sev=medium conf=0.85 note=4 people clustered within 150px
  DEDUP CHECK: 1 crowd alert(s) fired for a 4-person cluster over 4s (want exactly 1) -> PASS

TOTAL: 6/6 scenario logic tests passed
```

### 2. Full pipeline, real video file, offline `motion` detector

`python3 -m app.pipeline --input data/synthetic_test.mp4 --detector motion ...`
(see `tests/make_synthetic_video.py` for how that clip was generated, since a
real pedestrian download wasn't reachable from this sandbox):

```json
{
  "alert_count": 5,
  "scenario_counts": {
    "tailgating": 2,
    "restricted_zone_intrusion": 2,
    "crowd_formation": 1
  },
  "frames_processed": 350,
  "fps": 25.0,
  "duration_sec": 14.0
}
```

3 of 6 scenarios (tailgating, intrusion, crowd) fired with real, deduped
alerts from an actual annotated-video run — loitering/fall/altercation weren't
staged into that particular clip's motion but are covered by test #1 above.

**Run these two yourself** — `python3 tests/test_logic.py` and the pipeline
command above — to reproduce, and then run the real `--detector yolo` path
once you have internet, before the actual demo.
