"""
scenarios.py — ScenarioEngine

One _check_* method per adverse scenario. Each method reads only from the
small per-track history buffers maintained by tracker.Track — no scenario
method re-reads raw video frames. All methods are pluggable / independently
callable so scoring on "completeness per scenario" can be judged in isolation.

Extension point for a future VLM verification / LangGraph agentic layer:
see `_vlm_verify()` at the bottom — currently a no-op passthrough hook.
"""

import itertools
from dataclasses import dataclass, field, asdict
from datetime import datetime


def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test. polygon: list of (x, y). No extra deps."""
    if not polygon or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    else:
                        xinters = x1
                    if x1 == x2 or x <= xinters:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


@dataclass
class Alert:
    scenario: str
    track_id: object          # int for single-track alerts, list[int] for group alerts (crowd)
    confidence: float
    severity: str              # low | medium | high
    note: str
    frame: int
    timestamp: float

    def to_dict(self):
        d = asdict(self)
        d["timestamp"] = round(d["timestamp"], 2)
        d["confidence"] = round(d["confidence"], 2)
        return d


class ScenarioEngine:
    def __init__(self, config=None):
        cfg = config or {}

        # Restricted-zone intrusion
        self.restricted_zones = cfg.get("restricted_zones", [])  # [{"name","polygon","curfew_only":bool}]
        self.curfew_active = cfg.get("curfew_active", False)     # demo-friendly override; see README

        # Entry zone for tailgating
        self.entry_zone = cfg.get("entry_zone", None)            # polygon or None
        self.tailgate_window_sec = cfg.get("tailgate_window_sec", 2.0)

        # Loitering
        self.loiter_time_thresh = cfg.get("loiter_time_thresh", 7.0)   # seconds, default of 6-8s range
        self.loiter_radius_thresh = cfg.get("loiter_radius_px", 60)    # px spread considered "staying put"

        # Fall detection
        self.fall_ratio_tall = cfg.get("fall_ratio_tall", 1.4)
        self.fall_ratio_wide = cfg.get("fall_ratio_wide", 0.8)
        self.fall_window_sec = cfg.get("fall_window_sec", 0.6)

        # Altercation
        self.altercation_distance_px = cfg.get("altercation_distance_px", 90)
        self.altercation_speed_px_s = cfg.get("altercation_speed_px_s", 140)

        # Crowd formation
        self.crowd_min_people = cfg.get("crowd_min_people", 3)
        self.crowd_radius_px = cfg.get("crowd_radius_px", 120)
        self.crowd_cooldown_sec = cfg.get("crowd_cooldown_sec", 8.0)

        # generic per-(scenario, key) cooldown so a persistent condition
        # doesn't spam an alert every single frame
        self.alert_cooldowns = {
            "intrusion": cfg.get("intrusion_cooldown_sec", 6.0),
            "loitering": cfg.get("loiter_cooldown_sec", 10.0),
            "tailgating": cfg.get("tailgate_cooldown_sec", 4.0),
            "fall": cfg.get("fall_cooldown_sec", 5.0),
            "altercation": cfg.get("altercation_cooldown_sec", 6.0),
            "crowd": self.crowd_cooldown_sec,
        }
        self._last_alert_ts = {}          # (scenario, key) -> last timestamp fired
        self._entry_zone_log = []         # (track_id, timestamp) for tailgating window
        self._fall_flagged = set()        # track_ids already alerted for the current fall event

    # ---------------------------------------------------------------- utils

    def _cooldown_ok(self, scenario, key, now_ts):
        last = self._last_alert_ts.get((scenario, key))
        if last is None or (now_ts - last) >= self.alert_cooldowns[scenario]:
            self._last_alert_ts[(scenario, key)] = now_ts
            return True
        return False

    # ------------------------------------------------------------ scenario 1

    def _check_intrusion(self, track, frame_num, now_ts):
        alerts = []
        if not self.restricted_zones:
            return alerts
        cx, cy = track.centroid
        for zone in self.restricted_zones:
            poly = zone.get("polygon", [])
            if point_in_polygon((cx, cy), poly):
                curfew_only = zone.get("curfew_only", False)
                if curfew_only and not self.curfew_active:
                    continue
                key = f"{track.track_id}:{zone.get('name', 'zone')}"
                if not self._cooldown_ok("intrusion", key, now_ts):
                    continue
                severity = "high" if (curfew_only and self.curfew_active) else "medium"
                note = f"Person entered restricted zone '{zone.get('name', 'unnamed')}'"
                if curfew_only and self.curfew_active:
                    note += " during curfew hours"
                alerts.append(Alert(
                    scenario="restricted_zone_intrusion",
                    track_id=track.track_id,
                    confidence=0.9,
                    severity=severity,
                    note=note,
                    frame=frame_num,
                    timestamp=now_ts,
                ))
        return alerts

    # ------------------------------------------------------------ scenario 2

    def _check_loitering(self, track, frame_num, now_ts):
        alerts = []
        elapsed = now_ts - track.first_seen_ts
        if elapsed < self.loiter_time_thresh:
            return alerts
        radius, pts = track.movement_spread(self.loiter_time_thresh, now_ts)
        # need enough samples across the full window to be confident
        span = pts[-1][1] - pts[0][1] if len(pts) >= 2 else 0
        if span < self.loiter_time_thresh * 0.8:
            return alerts
        if radius <= self.loiter_radius_thresh:
            if not self._cooldown_ok("loitering", track.track_id, now_ts):
                return alerts
            alerts.append(Alert(
                scenario="loitering",
                track_id=track.track_id,
                confidence=0.8,
                severity="low",
                note=f"Person stationary in a small area for ~{self.loiter_time_thresh:.0f}s "
                     f"(movement radius {radius:.0f}px)",
                frame=frame_num,
                timestamp=now_ts,
            ))
        return alerts

    # ------------------------------------------------------------ scenario 3

    def _check_tailgating(self, tracks, frame_num, now_ts):
        alerts = []
        if not self.entry_zone:
            return alerts

        # log any track whose centroid is newly inside the entry zone this frame
        for tid, track in tracks.items():
            if point_in_polygon(track.centroid, self.entry_zone):
                if not self._entry_zone_log or self._entry_zone_log[-1][0] != tid:
                    # avoid re-logging the same track every single frame it's inside
                    already_logged_recently = any(
                        t == tid and (now_ts - ts) < 1.0 for t, ts in self._entry_zone_log[-10:]
                    )
                    if not already_logged_recently:
                        self._entry_zone_log.append((tid, now_ts))

        # prune old entries outside the window
        self._entry_zone_log = [
            (t, ts) for t, ts in self._entry_zone_log if now_ts - ts <= self.tailgate_window_sec + 1.0
        ]

        # look for 2+ distinct ids within the window
        recent = [(t, ts) for t, ts in self._entry_zone_log if now_ts - ts <= self.tailgate_window_sec]
        distinct_ids = sorted(set(t for t, _ in recent))
        if len(distinct_ids) >= 2:
            key = ",".join(str(i) for i in distinct_ids)
            if self._cooldown_ok("tailgating", key, now_ts):
                alerts.append(Alert(
                    scenario="tailgating",
                    track_id=distinct_ids,
                    confidence=0.75,
                    severity="medium",
                    note=f"{len(distinct_ids)} people entered the entry zone within "
                         f"{self.tailgate_window_sec:.1f}s of each other (ids {distinct_ids})",
                    frame=frame_num,
                    timestamp=now_ts,
                ))
        return alerts

    # ------------------------------------------------------------ scenario 4

    def _check_fall(self, track, frame_num, now_ts):
        alerts = []
        hist = list(track.bbox_history)
        if len(hist) < 2:
            return alerts

        # find the most recent moment the box was "tall" within the fall window,
        # and check whether it is now "wide"
        current_ratio = track.aspect_ratio()
        if current_ratio > self.fall_ratio_wide:
            return alerts  # not currently horizontal

        window_start = now_ts - self.fall_window_sec
        was_tall = any(
            (h / max(w, 1e-3)) > self.fall_ratio_tall
            for (f, t, x, y, w, h) in hist
            if t >= window_start
        )
        if not was_tall:
            return alerts

        if not self._cooldown_ok("fall", track.track_id, now_ts):
            return alerts

        alerts.append(Alert(
            scenario="fall_detection",
            track_id=track.track_id,
            confidence=0.7,
            severity="high",
            note=f"Bounding box flipped from standing to horizontal within "
                 f"{self.fall_window_sec:.1f}s — possible fall",
            frame=frame_num,
            timestamp=now_ts,
        ))
        return alerts

    # ------------------------------------------------------------ scenario 5

    def _check_altercation(self, tracks, frame_num, now_ts):
        alerts = []
        items = list(tracks.items())
        for (id_a, ta), (id_b, tb) in itertools.combinations(items, 2):
            dist = ((ta.centroid[0] - tb.centroid[0]) ** 2 + (ta.centroid[1] - tb.centroid[1]) ** 2) ** 0.5
            if dist > self.altercation_distance_px:
                continue
            speed_a = ta.speed()
            speed_b = tb.speed()
            if speed_a >= self.altercation_speed_px_s and speed_b >= self.altercation_speed_px_s:
                key = f"{min(id_a, id_b)}-{max(id_a, id_b)}"
                if not self._cooldown_ok("altercation", key, now_ts):
                    continue
                alerts.append(Alert(
                    scenario="possible_altercation",
                    track_id=[id_a, id_b],
                    confidence=0.65,
                    severity="high",
                    note=f"Persons {id_a} & {id_b} in close proximity ({dist:.0f}px) while both "
                         f"moving erratically (~{speed_a:.0f}/{speed_b:.0f} px/s)",
                    frame=frame_num,
                    timestamp=now_ts,
                ))
        return alerts

    # ------------------------------------------------------------ scenario 6

    def _check_crowd(self, tracks, frame_num, now_ts):
        """3+ people clustered tightly -> ONE alert per cluster, deduped."""
        alerts = []
        items = list(tracks.items())
        n = len(items)
        if n < self.crowd_min_people:
            return alerts

        # union-find style clustering by proximity graph
        parent = {tid: tid for tid, _ in items}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (id_a, ta), (id_b, tb) in itertools.combinations(items, 2):
            dist = ((ta.centroid[0] - tb.centroid[0]) ** 2 + (ta.centroid[1] - tb.centroid[1]) ** 2) ** 0.5
            if dist <= self.crowd_radius_px:
                union(id_a, id_b)

        clusters = {}
        for tid, _ in items:
            root = find(tid)
            clusters.setdefault(root, []).append(tid)

        for members in clusters.values():
            if len(members) < self.crowd_min_people:
                continue
            members_sorted = sorted(members)
            # dedup key = the exact cluster membership; cooldown prevents
            # re-firing every frame while the same group stays clustered
            key = ",".join(str(m) for m in members_sorted)
            if not self._cooldown_ok("crowd", key, now_ts):
                continue
            alerts.append(Alert(
                scenario="crowd_formation",
                track_id=members_sorted,
                confidence=0.85,
                severity="medium",
                note=f"{len(members_sorted)} people clustered within {self.crowd_radius_px}px "
                     f"(ids {members_sorted})",
                frame=frame_num,
                timestamp=now_ts,
            ))
        return alerts

    # -------------------------------------------------------------- driver

    def check_all(self, tracks, frame_num, now_ts):
        """tracks: dict[track_id -> Track] of tracks active this frame."""
        alerts = []
        for tid, track in tracks.items():
            alerts.extend(self._check_intrusion(track, frame_num, now_ts))
            alerts.extend(self._check_loitering(track, frame_num, now_ts))
            alerts.extend(self._check_fall(track, frame_num, now_ts))
        alerts.extend(self._check_tailgating(tracks, frame_num, now_ts))
        alerts.extend(self._check_altercation(tracks, frame_num, now_ts))
        alerts.extend(self._check_crowd(tracks, frame_num, now_ts))
        for a in alerts:
            self._vlm_verify(None, a)
        return alerts

    # --------------------------------------------------- future extension

    def _vlm_verify(self, frame, alert):
        """
        Extension point for a future VLM verification / LangGraph agentic
        router layer (e.g. re-check an ambiguous 'possible_altercation' alert
        against the actual frame crop with a vision model before escalating).
        Currently a no-op passthrough — intentionally left unimplemented per
        spec ("leave a clear extension point, don't implement now").
        """
        return alert
