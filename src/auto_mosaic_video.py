"""High-recall local video mosaic pipeline."""

import argparse
import os
import subprocess
import time

import cv2
import numpy as np
from nudenet import NudeDetector
from ultralytics import YOLO


TARGET_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED",
    "ANUS_COVERED",
    "BUTTOCKS_COVERED",
}


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0


def center_distance(a, b):
    acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
    bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
    scale = max(a[2], a[3], b[2], b[3], 1)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale


def update_tracks(tracks, detections, hold_frames=5, alpha=0.82):
    for track in tracks:
        track["matched"] = False

    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        best_index, best_metric = None, -999
        for index, track in enumerate(tracks):
            if track["matched"] or track["class"] != detection["class"]:
                continue
            overlap = iou(track["box"], detection["box"])
            distance = center_distance(track["box"], detection["box"])
            metric = overlap - 0.15 * distance
            if (overlap > 0.03 or distance < 1.1) and metric > best_metric:
                best_index, best_metric = index, metric

        if best_index is None:
            tracks.append(
                {
                    "class": detection["class"],
                    "box": [float(value) for value in detection["box"]],
                    "missed": 0,
                    "matched": True,
                }
            )
            continue

        track = tracks[best_index]
        track["box"] = [
            alpha * float(new) + (1 - alpha) * old
            for old, new in zip(track["box"], detection["box"])
        ]
        track["missed"] = 0
        track["matched"] = True

    active = []
    for track in tracks:
        if not track["matched"]:
            track["missed"] += 1
        if track["missed"] <= hold_frames:
            active.append(track)
    return active


def merge_nudenet_detections(detector, frame, threshold=0.18):
    detections = []
    for candidate, flipped in ((frame, False), (cv2.flip(frame, 1), True)):
        for detection in detector.detect(candidate):
            if (
                detection["class"] not in TARGET_CLASSES
                or detection["score"] < threshold
            ):
                continue
            x, y, width, height = detection["box"]
            if flipped:
                x = frame.shape[1] - x - width
            detections.append(
                {
                    "class": detection["class"],
                    "score": detection["score"],
                    "box": [x, y, width, height],
                }
            )
    return detections


def box_from_points(points, frame_width, frame_height, expand_x, expand_y):
    points = np.asarray(points, dtype=np.float32)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1, x2 = x1 - width * expand_x, x2 + width * expand_x
    y1, y2 = y1 - height * expand_y, y2 + height * expand_y
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)
    return [x1, y1, x2 - x1, y2 - y1]


def pose_guardrails(model, frame):
    """Return per-person bounds used to keep mosaics away from faces.

    Pose results are guardrails only in precision mode. They do not create
    censorship regions by themselves.
    """
    frame_height, frame_width = frame.shape[:2]
    result = model.predict(
        frame, imgsz=512, conf=0.12, iou=0.6, verbose=False
    )[0]
    if result.boxes is None:
        return []

    people = result.boxes.xyxy.cpu().numpy()
    keypoints = (
        result.keypoints.data.cpu().numpy()
        if result.keypoints is not None
        else []
    )
    people_bounds = []

    for index, person in enumerate(people):
        x1, y1, x2, y2 = person
        person_width, person_height = x2 - x1, y2 - y1
        if person_width < 25 or person_height < 50:
            continue

        keypoint = keypoints[index] if index < len(keypoints) else None
        shoulder_line = y1 + person_height * 0.18
        hip_line = y1 + person_height * 0.62
        if keypoint is not None:
            left_shoulder, right_shoulder = keypoint[5], keypoint[6]
            left_hip, right_hip = keypoint[11], keypoint[12]
            if left_shoulder[2] > 0.15 and right_shoulder[2] > 0.15:
                shoulder_line = float(
                    min(left_shoulder[1], right_shoulder[1])
                )
            if left_hip[2] > 0.12 and right_hip[2] > 0.12:
                hip_line = float((left_hip[1] + right_hip[1]) / 2)

        people_bounds.append(
            {
                "person": [x1, y1, person_width, person_height],
                "shoulder_line": shoulder_line,
                "hip_line": hip_line,
            }
        )
    return people_bounds


def constrain_sensitive_box(box, class_name, people_bounds):
    """Clip a detected sensitive box to plausible torso/pelvis limits."""
    x, y, width, height = [float(value) for value in box]
    center_x, center_y = x + width / 2, y + height / 2
    matching_people = []
    for bounds in people_bounds:
        px, py, pw, ph = bounds["person"]
        if px <= center_x <= px + pw and py <= center_y <= py + ph:
            matching_people.append(bounds)
    if not matching_people:
        return [x, y, width, height]

    bounds = min(
        matching_people,
        key=lambda item: item["person"][2] * item["person"][3],
    )
    px, py, pw, ph = bounds["person"]

    if "BREAST" in class_name:
        min_y = bounds["shoulder_line"] + ph * 0.02
        max_y = bounds["hip_line"] - ph * 0.06
    else:
        min_y = bounds["hip_line"] - ph * 0.10
        max_y = py + ph * 0.96

    x1 = max(x, px)
    y1 = max(y, min_y)
    x2 = min(x + width, px + pw)
    y2 = min(y + height, max_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def mosaic_region(frame, box, block_size=28, padding_ratio=0.06):
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = box
    padding_x = max(8, int(width * padding_ratio))
    padding_y = max(8, int(height * padding_ratio))
    x1, y1 = max(0, int(x) - padding_x), max(0, int(y) - padding_y)
    x2 = min(frame_width, int(x + width) + padding_x)
    y2 = min(frame_height, int(y + height) + padding_y)
    if x2 <= x1 or y2 <= y1:
        return

    region = frame[y1:y2, x1:x2]
    small_width = max(1, (x2 - x1) // block_size)
    small_height = max(1, (y2 - y1) // block_size)
    reduced = cv2.resize(
        region, (small_width, small_height), interpolation=cv2.INTER_AREA
    )
    frame[y1:y2, x1:x2] = cv2.resize(
        reduced, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
    )


def process_video(input_path, output_path, ffmpeg_path, pose_model_path):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = NudeDetector()
    pose_model = YOLO(pose_model_path)

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "pipe:0",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-shortest",
        output_path,
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    tracks = []
    frame_index = 0
    started = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = merge_nudenet_detections(detector, frame)
            tracks = update_tracks(tracks, detections)
            guardrails = pose_guardrails(pose_model, frame)
            for track in tracks:
                box = constrain_sensitive_box(
                    track["box"], track["class"], guardrails
                )
                if box is not None:
                    mosaic_region(frame, box)

            encoder.stdin.write(frame.tobytes())
            frame_index += 1
            if frame_index % 100 == 0 or frame_index == total_frames:
                elapsed = time.time() - started
                speed = frame_index / elapsed if elapsed else 0
                print(
                    f"{frame_index}/{total_frames} "
                    f"({frame_index / total_frames * 100:.1f}%) "
                    f"{speed:.1f} fps",
                    flush=True,
                )
    finally:
        capture.release()
        if encoder.stdin:
            encoder.stdin.close()

    return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg exited with code {return_code}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--pose-model", required=True)
    arguments = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
    process_video(
        arguments.input,
        arguments.output,
        arguments.ffmpeg,
        arguments.pose_model,
    )


if __name__ == "__main__":
    main()
