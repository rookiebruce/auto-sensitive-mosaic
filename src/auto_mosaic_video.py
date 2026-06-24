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


def update_tracks(tracks, detections, hold_frames=8, alpha=0.80):
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


def merge_nudenet_detections(detector, frame, threshold=0.10):
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
                "keypoint": keypoint,
            }
        )
    return people_bounds


def pose_safety_detections(people_bounds):
    """Add small torso/pelvis safety boxes when the detector misses.

    These boxes intentionally do not cover the face: the upper fallback starts
    below the shoulder line, and the lower fallback is anchored around the hip
    line. They trade some extra torso/pelvis censorship for fewer exposed
    frames.
    """
    detections = []
    for bounds in people_bounds:
        px, py, pw, ph = [float(value) for value in bounds["person"]]
        shoulder_line = float(bounds["shoulder_line"])
        hip_line = float(bounds["hip_line"])
        keypoint = bounds.get("keypoint")
        fallback_added = False

        torso_top = max(py + ph * 0.20, shoulder_line + ph * 0.04)
        torso_bottom = min(hip_line - ph * 0.04, py + ph * 0.58)
        if torso_bottom > torso_top:
            if keypoint is not None and keypoint[5][2] > 0.12 and keypoint[6][2] > 0.12:
                left_shoulder, right_shoulder = keypoint[5], keypoint[6]
                shoulder_x1 = min(left_shoulder[0], right_shoulder[0])
                shoulder_x2 = max(left_shoulder[0], right_shoulder[0])
                shoulder_width = max(shoulder_x2 - shoulder_x1, pw * 0.35)
                x1 = max(px, shoulder_x1 - shoulder_width * 0.35)
                x2 = min(px + pw, shoulder_x2 + shoulder_width * 0.35)
            else:
                x1 = px + pw * 0.12
                x2 = px + pw * 0.88
            if x2 > x1:
                detections.append(
                    {
                        "class": "BREAST_FALLBACK",
                        "score": 0.11,
                        "box": [x1, torso_top, x2 - x1, torso_bottom - torso_top],
                    }
                )
                fallback_added = True

        if keypoint is not None:
            extra_box = abnormal_pose_breast_box(keypoint, px, py, pw, ph)
            if extra_box is not None:
                detections.append(
                    {
                        "class": "BREAST_FALLBACK",
                        "score": 0.12 if fallback_added else 0.13,
                        "box": extra_box,
                    }
                )

        pelvis_top = max(py + ph * 0.50, hip_line - ph * 0.10)
        pelvis_bottom = min(py + ph * 0.94, hip_line + ph * 0.30)
        if pelvis_bottom > pelvis_top:
            if keypoint is not None and keypoint[11][2] > 0.10 and keypoint[12][2] > 0.10:
                left_hip, right_hip = keypoint[11], keypoint[12]
                hip_x1 = min(left_hip[0], right_hip[0])
                hip_x2 = max(left_hip[0], right_hip[0])
                hip_width = max(hip_x2 - hip_x1, pw * 0.28)
                x1 = max(px, hip_x1 - hip_width * 0.55)
                x2 = min(px + pw, hip_x2 + hip_width * 0.55)
            else:
                x1 = px + pw * 0.18
                x2 = px + pw * 0.82
            if x2 > x1:
                detections.append(
                    {
                        "class": "FEMALE_GENITALIA_EXPOSED",
                        "score": 0.11,
                        "box": [x1, pelvis_top, x2 - x1, pelvis_bottom - pelvis_top],
                    }
                )
    return detections


def visible_points(keypoint, indexes, threshold=0.10):
    points = []
    for index in indexes:
        point = keypoint[index]
        if point[2] > threshold:
            points.append((float(point[0]), float(point[1])))
    return points


def abnormal_pose_breast_box(keypoint, px, py, pw, ph):
    """Return a higher chest box for lying/back-arched pose failures."""
    shoulders = visible_points(keypoint, [5, 6], 0.12)
    hips = visible_points(keypoint, [11, 12], 0.10)
    if len(shoulders) < 1:
        return None

    torso_points = shoulders + hips
    xs = [point[0] for point in torso_points]
    ys = [point[1] for point in torso_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    shoulder_mid = (
        sum(point[0] for point in shoulders) / len(shoulders),
        sum(point[1] for point in shoulders) / len(shoulders),
    )

    has_hips = len(hips) >= 1
    if has_hips:
        hip_mid = (
            sum(point[0] for point in hips) / len(hips),
            sum(point[1] for point in hips) / len(hips),
        )
        torso_dx = abs(shoulder_mid[0] - hip_mid[0])
        torso_dy = abs(shoulder_mid[1] - hip_mid[1])
    else:
        hip_mid = None
        torso_dx = 0
        torso_dy = 0

    shoulders_low = shoulder_mid[1] > py + ph * 0.42
    torso_flat = has_hips and torso_dy < ph * 0.18
    torso_horizontal = has_hips and torso_dx > max(80, torso_dy * 1.25)
    hips_missing = not has_hips
    if not (shoulders_low or torso_flat or torso_horizontal or hips_missing):
        return None

    face_points = visible_points(keypoint, [0, 1, 2, 3, 4], 0.12)
    face_x = (
        sum(point[0] for point in face_points) / len(face_points)
        if face_points
        else None
    )
    face_y = (
        sum(point[1] for point in face_points) / len(face_points)
        if face_points
        else None
    )

    if has_hips and torso_horizontal:
        pad_left = pw * 0.10
        pad_right = pw * 0.05
        if face_x is not None and face_x < min_x:
            pad_left, pad_right = pad_right, pad_left
        x1 = min_x - pad_left
        x2 = max_x + pad_right
        y1 = min_y - ph * 0.32
        y2 = max_y + ph * 0.12
    elif hips_missing:
        shoulder_span = max(max_x - min_x, pw * 0.28)
        x1 = min_x - shoulder_span * 0.35
        x2 = max_x + shoulder_span * 0.18
        if face_x is not None and face_x > max_x:
            x2 = min(x2, max_x + shoulder_span * 0.25)
        elif face_x is not None and face_x < min_x:
            x1 = max(x1, min_x - shoulder_span * 0.25)
        y1 = min_y - max(80, ph * 0.28)
        y2 = max_y + ph * 0.12
    else:
        center_x = (min_x + max_x) / 2
        width = max(max_x - min_x, pw * 0.58)
        x1 = center_x - width * 0.55
        x2 = center_x + width * 0.55
        y1 = min_y - ph * 0.08
        y2 = min(py + ph * 0.98, max_y + ph * 0.32)

    overlaps_face_x = (
        face_x is not None and x1 <= face_x <= x2
    )
    if overlaps_face_x and face_y is not None and y1 < face_y + ph * 0.04:
        y1 = max(y1, face_y + ph * 0.04)

    x1 = max(px, x1)
    y1 = max(py, y1)
    x2 = min(px + pw, x2)
    y2 = min(py + ph, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def constrain_sensitive_box(box, class_name, people_bounds):
    """Clip a detected sensitive box to plausible torso/pelvis limits."""
    x, y, width, height = [float(value) for value in box]
    if class_name == "BREAST_FALLBACK":
        return [x, y, width, height]

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


def mosaic_region(frame, box, block_size=34, padding_ratio=0.10):
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
            guardrails = pose_guardrails(pose_model, frame)
            detections.extend(pose_safety_detections(guardrails))
            tracks = update_tracks(tracks, detections)
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
