"""
Offline logic test for tracker.py + scenarios.py.

This bypasses video/detection entirely and feeds hand-crafted per-frame
bounding-box sequences straight into CentroidTracker + ScenarioEngine,
engineered so each of the 6 scenarios should fire exactly once (or a known
small number of times) after dedup/cooldown. This proves the *decision
logic* end-to-end without requiring YOLO weights or a video file.

Run: python3 tests/test_logic.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tracker import CentroidTracker
from app.scenarios import ScenarioEngine

FPS = 25.0


def run_scenario(name, frames_fn, config, n_frames, expect_scenarios):
    """frames_fn(frame_num) -> list of (x,y,w,h) detections for that frame."""
    tracker = CentroidTracker(max_disappeared=20, max_distance=120, history_seconds=20, fps=FPS)
    engine = ScenarioEngine(config)
    fired = {}
    for f in range(n_frames):
        ts = f / FPS
        dets = frames_fn(f)
        tracks = tracker.update(dets, f, ts)
        alerts = engine.check_all(tracks, f, ts)
        for a in alerts:
            fired.setdefault(a.scenario, []).append(a)

    print(f"\n=== {name} ===")
    for scen, alerts in fired.items():
        print(f"  {scen}: {len(alerts)} alert(s)")
        for a in alerts[:3]:
            print(f"    - track={a.track_id} sev={a.severity} conf={a.confidence} note={a.note}")
    ok = all(s in fired for s in expect_scenarios)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} (expected to see: {expect_scenarios})")
    return ok, fired


results = []

# 1) Restricted zone intrusion (with curfew boost)
zone = [[100, 100], [300, 100], [300, 300], [100, 300]]
def f_intrusion(f):
    # person walks from outside straight into the zone and stays
    x = 20 + f * 6
    return [(x, 150, 40, 90)]
ok, _ = run_scenario(
    "restricted_zone_intrusion (curfew)", f_intrusion,
    {"restricted_zones": [{"name": "Roof", "polygon": zone, "curfew_only": True}], "curfew_active": True},
    60, {"restricted_zone_intrusion"}
)
results.append(ok)

# 2) Loitering — person stays within a small radius for > threshold
def f_loiter(f):
    import math
    cx, cy = 400, 400
    jitter = 8 * math.sin(f * 0.3)
    return [(cx + jitter, cy, 40, 90)]
ok, _ = run_scenario(
    "loitering", f_loiter,
    {"loiter_time_thresh": 3.0, "loiter_radius_px": 40},
    int(FPS * 5), {"loitering"}
)
results.append(ok)

# 3) Tailgating — two distinct IDs enter the entry zone within ~2s
entry_zone = [[0, 0], [200, 0], [200, 200], [0, 200]]
def f_tailgate(f):
    dets = []
    # person A enters at frame 10
    if f >= 10:
        dets.append((50 + min(f - 10, 40) * 1.0, 50, 30, 70))
    # person B enters at frame 25 (0.6s later @25fps) -> within 2s window
    if f >= 25:
        dets.append((120 + min(f - 25, 40) * 1.0, 90, 30, 70))
    return dets
ok, _ = run_scenario(
    "tailgating", f_tailgate,
    {"entry_zone": entry_zone, "tailgate_window_sec": 2.0},
    80, {"tailgating"}
)
results.append(ok)

# 4) Fall detection — bbox flips from tall to wide within ~0.5s
def f_fall(f):
    if f < 20:
        return [(300, 300, 40, 100)]   # standing: h/w = 2.5 (tall)
    else:
        return [(280, 370, 100, 35)]   # fallen: h/w = 0.35 (wide)
ok, _ = run_scenario(
    "fall_detection", f_fall,
    {"fall_window_sec": 0.6},
    40, {"fall_detection"}
)
results.append(ok)

# 5) Possible altercation — two people close together, both moving fast/erratically
def f_altercation(f):
    import math
    # both oscillate rapidly near each other -> high instantaneous speed + close distance
    ax = 500 + 30 * math.sin(f * 1.1)
    ay = 500 + 30 * math.cos(f * 1.3)
    bx = 540 + 30 * math.cos(f * 1.2)
    by = 500 + 30 * math.sin(f * 0.9)
    return [(ax, ay, 35, 80), (bx, by, 35, 80)]
ok, _ = run_scenario(
    "possible_altercation", f_altercation,
    {"altercation_distance_px": 140, "altercation_speed_px_s": 80},
    30, {"possible_altercation"}
)
results.append(ok)

# 6) Crowd formation — 3+ people cluster; must be ONE alert per cluster, not one per person
def f_crowd(f):
    # 4 people converge into a tight cluster and stay there -> should be exactly
    # one crowd alert (deduped), not 4, and not re-fired every frame (cooldown)
    base = 700
    return [
        (base, 700, 30, 70),
        (base + 20, 705, 30, 70),
        (base + 40, 695, 30, 70),
        (base + 10, 720, 30, 70),
    ]
ok, fired = run_scenario(
    "crowd_formation", f_crowd,
    {"crowd_min_people": 3, "crowd_radius_px": 150, "crowd_cooldown_sec": 8.0},
    int(FPS * 4), {"crowd_formation"}
)
n_crowd_alerts = len(fired.get("crowd_formation", []))
dedup_ok = n_crowd_alerts == 1  # 4 people held in cluster for 4s w/ 8s cooldown -> exactly 1 alert
print(f"  DEDUP CHECK: {n_crowd_alerts} crowd alert(s) fired for a 4-person cluster over 4s "
      f"(want exactly 1, NOT one-per-person) -> {'PASS' if dedup_ok else 'FAIL'}")
results.append(ok and dedup_ok)

print("\n" + "=" * 60)
print(f"TOTAL: {sum(results)}/{len(results)} scenario logic tests passed")
print("=" * 60)
sys.exit(0 if all(results) else 1)
