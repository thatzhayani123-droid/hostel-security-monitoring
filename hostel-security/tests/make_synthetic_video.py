"""
Generates data/synthetic_test.mp4 — a synthetic clip with moving person-shaped
blobs, used to exercise the *actual* pipeline.py (video I/O, motion-blob
detector, tracker, scenario engine, overlay drawing, JSON writer) end-to-end
in this offline sandbox where a real pedestrian clip / YOLO weights can't be
downloaded. In an internet-connected environment, swap --detector motion for
--detector yolo and point --input at a real CCTV clip instead.
"""
import os
import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 480, 25, 14
N_FRAMES = FPS * SECONDS
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic_test.mp4")


def person(frame, cx, cy, color=(210, 210, 210)):
    cv2.ellipse(frame, (int(cx), int(cy - 25)), (14, 30), 0, 0, 360, color, -1)   # torso
    cv2.circle(frame, (int(cx), int(cy - 62)), 12, color, -1)                     # head


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

    for f in range(N_FRAMES):
        frame = np.full((H, W, 3), (40, 40, 42), dtype=np.uint8)
        # static floor line for visual context only
        cv2.line(frame, (0, H - 60), (W, H - 60), (55, 55, 58), 2)
        t = f / FPS

        # --- Person A: walks straight into the restricted zone (top-right box) and loiters there
        ax = 60 + min(f, 150) * 2.4
        ay = 130
        if f > 150:
            ax = 60 + 150 * 2.4 + 6 * np.sin(f * 0.15)  # loiter with tiny jitter once inside
        person(frame, ax, ay)

        # --- Person B & C: enter the entry zone (bottom-left box) ~0.5s apart -> tailgating
        # kept well apart vertically so the motion-blob fallback detector sees two
        # separate blobs instead of merging them into one contour
        if f >= 60:
            bx = 40 + min(f - 60, 60) * 2.0
            person(frame, bx, 270)
        if f >= 72:
            cx_ = 40 + min(f - 72, 60) * 2.0
            person(frame, cx_, 410)

        # --- Persons D, E, F: converge into a tight crowd cluster mid-frame from frame 180
        if f >= 180:
            k = min(f - 180, 40)
            person(frame, 420 - k * 1.5, 380 - k * 0.5)
            person(frame, 460 + k * 0.2, 385)
            person(frame, 440, 410 + k * 0.3)

        writer.write(frame)

    writer.release()
    print(f"Wrote {N_FRAMES} frames -> {OUT}")


if __name__ == "__main__":
    main()
