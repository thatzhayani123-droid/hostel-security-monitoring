"""
tracker.py — CentroidTracker

A dependency-free (numpy only) multi-object tracker based on nearest-centroid
matching between frames. No deep_sort / motpy / any heavyweight tracking lib.

Each track carries small rolling history buffers (positions + bboxes, each
timestamped by frame number and seconds) which the ScenarioEngine reads from
to compute movement spread, speed, aspect-ratio changes, dwell time, etc.

Usage:
    tracker = CentroidTracker(max_disappeared=15, max_distance=80, history_seconds=12, fps=25)
    tracks = tracker.update(detections, frame_num, timestamp_sec)
    # tracks: dict[track_id -> Track]
"""

from collections import deque, OrderedDict
import numpy as np


class Track:
    """Holds the rolling state for a single tracked person."""

    __slots__ = (
        "track_id", "centroid", "bbox", "disappeared",
        "position_history", "bbox_history", "first_seen_frame", "first_seen_ts",
    )

    def __init__(self, track_id, centroid, bbox, frame_num, timestamp, history_len):
        self.track_id = track_id
        self.centroid = centroid          # (cx, cy)
        self.bbox = bbox                  # (x, y, w, h)
        self.disappeared = 0
        self.first_seen_frame = frame_num
        self.first_seen_ts = timestamp
        # each entry: (frame_num, timestamp, cx, cy)
        self.position_history = deque(maxlen=history_len)
        # each entry: (frame_num, timestamp, x, y, w, h)
        self.bbox_history = deque(maxlen=history_len)
        self._push(frame_num, timestamp, centroid, bbox)

    def _push(self, frame_num, timestamp, centroid, bbox):
        self.position_history.append((frame_num, timestamp, centroid[0], centroid[1]))
        self.bbox_history.append((frame_num, timestamp, bbox[0], bbox[1], bbox[2], bbox[3]))

    def update_state(self, centroid, bbox, frame_num, timestamp):
        self.centroid = centroid
        self.bbox = bbox
        self.disappeared = 0
        self._push(frame_num, timestamp, centroid, bbox)

    # ---- convenience accessors used heavily by ScenarioEngine ----

    def aspect_ratio(self):
        _, _, x, y, w, h = self.bbox_history[-1]
        if w <= 0:
            return 0.0
        return h / float(w)

    def recent_positions(self, seconds, now_ts):
        """Return position history entries within the last `seconds`."""
        return [p for p in self.position_history if now_ts - p[1] <= seconds]

    def speed(self, window=5):
        """Approx instantaneous speed in px/sec using the last `window` samples."""
        pts = list(self.position_history)[-window:]
        if len(pts) < 2:
            return 0.0
        (f0, t0, x0, y0), (f1, t1, x1, y1) = pts[0], pts[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        dist = float(np.hypot(x1 - x0, y1 - y0))
        return dist / dt

    def movement_spread(self, seconds, now_ts):
        """Radius (px) of the bounding circle of recent positions — used for loitering."""
        pts = self.recent_positions(seconds, now_ts)
        if len(pts) < 2:
            return 0.0, pts
        xs = np.array([p[2] for p in pts])
        ys = np.array([p[3] for p in pts])
        cx, cy = xs.mean(), ys.mean()
        radius = float(np.max(np.hypot(xs - cx, ys - cy)))
        return radius, pts


class CentroidTracker:
    """
    Nearest-centroid multi-object tracker.

    - New detections are matched to existing tracks by minimizing total
      centroid distance (greedy row/col elimination on the distance matrix).
    - A track that goes unmatched for > max_disappeared frames is deregistered.
    - A detection that can't be matched within max_distance px spawns a new track.
    """

    def __init__(self, max_disappeared=20, max_distance=90, history_seconds=15, fps=25):
        self.next_id = 0
        self.tracks = OrderedDict()  # track_id -> Track
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_len = max(30, int(history_seconds * fps))
        self.fps = fps

    def _register(self, centroid, bbox, frame_num, timestamp):
        t = Track(self.next_id, centroid, bbox, frame_num, timestamp, self.history_len)
        self.tracks[self.next_id] = t
        self.next_id += 1
        return t.track_id

    def _deregister(self, track_id):
        del self.tracks[track_id]

    @staticmethod
    def _centroid_of(bbox):
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    def update(self, detections, frame_num, timestamp):
        """
        detections: list of (x, y, w, h) in pixel coords (top-left x,y + width,height)
        Returns: dict[track_id -> Track] of currently-active tracks (this frame).
        """
        if len(detections) == 0:
            # mark everyone as disappeared once; drop stale tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    self._deregister(track_id)
            return {}

        input_centroids = np.array([self._centroid_of(b) for b in detections])

        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self._register(tuple(input_centroids[i]), tuple(detections[i]), frame_num, timestamp)
        else:
            track_ids = list(self.tracks.keys())
            existing_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])

            # distance matrix: existing tracks (rows) vs new detections (cols)
            diff = existing_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :]
            dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

            # greedy assignment: smallest distances first
            rows_sorted = dist_matrix.min(axis=1).argsort()
            used_rows, used_cols = set(), set()

            for row in rows_sorted:
                if row in used_rows:
                    continue
                col = dist_matrix[row].argmin()
                if col in used_cols:
                    # find next best unused column for this row
                    order = np.argsort(dist_matrix[row])
                    col = None
                    for c in order:
                        if c not in used_cols:
                            col = c
                            break
                    if col is None:
                        continue
                if dist_matrix[row, col] > self.max_distance:
                    continue

                track_id = track_ids[row]
                self.tracks[track_id].update_state(
                    tuple(input_centroids[col]), tuple(detections[col]), frame_num, timestamp
                )
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(track_ids))) - used_rows
            unused_cols = set(range(len(detections))) - used_cols

            for row in unused_rows:
                track_id = track_ids[row]
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    self._deregister(track_id)

            for col in unused_cols:
                self._register(tuple(input_centroids[col]), tuple(detections[col]), frame_num, timestamp)

        # return only tracks seen this frame (disappeared == 0)
        return {tid: t for tid, t in self.tracks.items() if t.disappeared == 0}
