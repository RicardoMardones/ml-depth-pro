from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import math
import os
from typing import Optional

import cv2
import depth_pro
import numpy as np
import torch
from ultralytics import YOLO, SAM

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ============================================================
# Configuracion stream
# ============================================================

USE_RTSP = os.getenv("USE_RTSP", "1") == "1"

RTSP_USER = os.getenv("RTSP_USER", "admin")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", "itg24chile")
RTSP_IP = os.getenv("RTSP_IP", "10.22.100.22")
RTSP_PORT = int(os.getenv("RTSP_PORT", "554"))
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "802")
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()

RTSP_URL = (
    f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}"
    f"@{RTSP_IP}:{RTSP_PORT}/Streaming/Channels/{RTSP_CHANNEL}"
)

VIDEO_SOURCE = RTSP_URL if USE_RTSP else int(os.getenv("CAMERA_INDEX", "0"))


# ============================================================
# Configuracion modelos
# ============================================================

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s-worldv2.pt")
SAM_WEIGHTS = os.getenv("SAM_WEIGHTS", "mobile_sam.pt")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
TARGET_CONFIDENCE_THRESHOLD = float(os.getenv("TARGET_CONFIDENCE_THRESHOLD", "0.55"))

TARGET_CLASSES = [
    "fish",
    "salmon fish",
    "trout fish",
]
TARGET_CLASS_SET = set(TARGET_CLASSES)
TARGET_LABEL = "pez"


# ============================================================
# Geometria / lente / salida
# ============================================================

LENS_HFOV_DEG = float(os.getenv("LENS_HFOV_DEG", "127.3"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs_live_track_segm"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_WINDOWS = os.getenv("SHOW_WINDOWS", "1") == "1"
WINDOW_NAME = "YOLO live track segm"
KEEP_RESULTS_WINDOW_ON_EXIT = os.getenv("KEEP_RESULTS_WINDOW_ON_EXIT", "1") == "1"

ENABLE_3D_VIEWER = os.getenv("ENABLE_3D_VIEWER", "1") == "1"
SHOW_3D_AXIS = os.getenv("SHOW_3D_AXIS", "0") == "1"
SAVE_POINT_CLOUD_COLOR = os.getenv("SAVE_POINT_CLOUD_COLOR", "1") == "1"
SAVE_FULL_DEPTH_POINT_CLOUD = os.getenv("SAVE_FULL_DEPTH_POINT_CLOUD", "0") == "1"
SAVE_TARGET_SESSION_ARTIFACTS = os.getenv("SAVE_TARGET_SESSION_ARTIFACTS", "1") == "1"

MAX_POINT_CLOUD_POINTS = int(os.getenv("MAX_POINT_CLOUD_POINTS", "60000"))
MAX_DEPTH_VIEWER_POINTS = int(os.getenv("MAX_DEPTH_VIEWER_POINTS", "220000"))
MAX_STREAM_READ_FAILURES = int(os.getenv("MAX_STREAM_READ_FAILURES", "8"))


# ============================================================
# ROI de entrada y ROI de medicion
# ============================================================

# ENTRY_SIDE = os.getenv("ENTRY_SIDE", "left").strip().lower()
ENTRY_SIDE = os.getenv("ENTRY_SIDE", "right").strip().lower()

if ENTRY_SIDE not in {"left", "right"}:
    ENTRY_SIDE = "left"

ENTRY_MAX_X_RATIO = float(os.getenv("ENTRY_MAX_X_RATIO", "0.45"))
ENTRY_MIN_Y_RATIO = float(os.getenv("ENTRY_MIN_Y_RATIO", "0.10"))
ENTRY_MAX_Y_RATIO = float(os.getenv("ENTRY_MAX_Y_RATIO", "0.90"))

MEASURE_ROI = (
    float(os.getenv("MEASURE_ROI_X_MIN", "0.15")),
    float(os.getenv("MEASURE_ROI_Y_MIN", "0.15")),
    float(os.getenv("MEASURE_ROI_X_MAX", "0.85")),
    float(os.getenv("MEASURE_ROI_Y_MAX", "0.85")),
)


# ============================================================
# Tracking
# ============================================================

PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "5"))
DISPLAY_FPS = float(os.getenv("DISPLAY_FPS", "15"))

TRACKER_IOU_WEIGHT = float(os.getenv("TRACKER_IOU_WEIGHT", "0.70"))
TRACKER_CENTER_WEIGHT = float(os.getenv("TRACKER_CENTER_WEIGHT", "0.30"))
TRACKER_SIZE_WEIGHT = float(os.getenv("TRACKER_SIZE_WEIGHT", "0.20"))
TRACKER_DIRECTION_WEIGHT = float(os.getenv("TRACKER_DIRECTION_WEIGHT", "0.35"))
TRACKER_MIN_SCORE = float(os.getenv("TRACKER_MIN_SCORE", "0.20"))
TRACKER_MAX_MISSING = int(os.getenv("TRACKER_MAX_MISSING", "10"))
TRACKER_MAX_CENTER_DIST_RATIO = float(os.getenv("TRACKER_MAX_CENTER_DIST_RATIO", "0.18"))
TRACKER_MIN_SIZE_SIMILARITY = float(os.getenv("TRACKER_MIN_SIZE_SIMILARITY", "0.35"))
TRACKER_MAX_BACKTRACK_X_RATIO = float(os.getenv("TRACKER_MAX_BACKTRACK_X_RATIO", "0.04"))
TRAIL_MAX_POINTS = int(os.getenv("TRAIL_MAX_POINTS", "80"))

TARGET_MIN_HITS = int(os.getenv("TARGET_MIN_HITS", "2"))
MAX_TARGET_OBJECTS = int(os.getenv("MAX_TARGET_OBJECTS", "3"))


# ============================================================
# Filtros medicion
# ============================================================

MIN_MASK_AREA_PX = int(os.getenv("MIN_MASK_AREA_PX", "1500"))
MIN_DISTANCE_M = float(os.getenv("MIN_DISTANCE_M", "0.20"))
MAX_DISTANCE_M = float(os.getenv("MAX_DISTANCE_M", "4.00"))
MIN_FISH_LENGTH_M = float(os.getenv("MIN_FISH_LENGTH_M", "0.05"))
MAX_FISH_LENGTH_M = float(os.getenv("MAX_FISH_LENGTH_M", "1.50"))

FISH_THICKNESS_RATIO = float(os.getenv("FISH_THICKNESS_RATIO", "0.45"))
FISH_VOLUME_SHAPE_FACTOR = float(os.getenv("FISH_VOLUME_SHAPE_FACTOR", "0.55"))


# ============================================================
# Modelos de datos
# ============================================================

@dataclass
class Detection:
    box: list[int]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class Track:
    track_id: int
    detection: Detection
    missing: int = 0
    hits: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    trail: deque[tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=TRAIL_MAX_POINTS)
    )

    def __post_init__(self) -> None:
        self.detection.track_id = self.track_id
        self.trail.append(bbox_center(self.detection.box))

    def update(self, detection: Detection) -> None:
        previous_center = bbox_center(self.detection.box)
        new_center = bbox_center(detection.box)
        delta_x = float(new_center[0] - previous_center[0])
        delta_y = float(new_center[1] - previous_center[1])

        self.velocity = (
            0.65 * delta_x + 0.35 * self.velocity[0],
            0.65 * delta_y + 0.35 * self.velocity[1],
        )

        detection.track_id = self.track_id
        self.detection = detection
        self.missing = 0
        self.hits += 1
        self.trail.append(new_center)

    def mark_missing(self) -> None:
        self.missing += 1

    @property
    def is_alive(self) -> bool:
        return self.missing <= TRACKER_MAX_MISSING

    @property
    def is_active(self) -> bool:
        return self.missing == 0

    def predicted_box(self) -> list[int]:
        if self.hits < 2:
            return self.detection.box

        steps_ahead = max(1, self.missing + 1)
        return shift_box(
            self.detection.box,
            self.velocity[0] * steps_ahead,
            self.velocity[1] * steps_ahead,
        )


@dataclass
class TrackedFrame:
    frame_index: int
    timestamp: str
    frame_bgr: np.ndarray
    detection: Detection


@dataclass
class TargetSession:
    track_id: int
    started_timestamp: str
    frames: list[TrackedFrame] = field(default_factory=list)
    best_quality_score: float = -1.0
    best_result_info: dict | None = None
    best_annotated: np.ndarray | None = None
    measurements_ok: int = 0
    final_reason: str = ""

    def append_frame(
        self,
        frame_index: int,
        timestamp: str,
        frame_bgr: np.ndarray,
        detection: Detection,
    ) -> None:
        self.frames.append(
            TrackedFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                frame_bgr=frame_bgr.copy(),
                detection=clone_detection(detection),
            )
        )

    def update_best(self, annotated: np.ndarray, result_info: dict) -> None:
        closest_obj = result_info.get("closest_obj")

        if closest_obj is None or not closest_obj.get("filter_passed", False):
            return

        self.measurements_ok += 1
        quality_score = float(closest_obj.get("quality_score", 0.0))

        if quality_score <= self.best_quality_score:
            return

        self.best_quality_score = quality_score
        self.best_result_info = result_info
        self.best_annotated = annotated.copy()


# ============================================================
# Utilidades generales
# ============================================================

def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def clone_detection(detection: Detection) -> Detection:
    return Detection(
        box=list(detection.box),
        confidence=float(detection.confidence),
        class_id=int(detection.class_id),
        class_name=str(detection.class_name),
        track_id=detection.track_id,
    )


def open_video_capture():
    if USE_RTSP:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap


def bbox_center(box: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bbox_area(box: list[int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def shift_box(box: list[int], delta_x: float, delta_y: float) -> list[int]:
    x1, y1, x2, y2 = box
    dx = int(round(delta_x))
    dy = int(round(delta_y))
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def center_similarity(
    box_a: list[int],
    box_b: list[int],
    frame_w: int,
    frame_h: int,
) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)

    diagonal = math.sqrt(frame_w * frame_w + frame_h * frame_h)
    if diagonal <= 0:
        return 0.0

    distance = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    similarity = 1.0 - distance / diagonal
    return float(np.clip(similarity, 0.0, 1.0))


def center_distance(box_a: list[int], box_b: list[int]) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def bbox_size_similarity(box_a: list[int], box_b: list[int]) -> float:
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    return float(min(area_a, area_b) / max(area_a, area_b))


def direction_similarity(previous_box: list[int], current_box: list[int], frame_w: int) -> float:
    previous_x, _ = bbox_center(previous_box)
    current_x, _ = bbox_center(current_box)

    if frame_w <= 0:
        return 0.0

    max_backtrack_px = max(1.0, frame_w * TRACKER_MAX_BACKTRACK_X_RATIO)
    delta_x = current_x - previous_x

    if ENTRY_SIDE == "right":
        if delta_x <= 0:
            return 1.0
        return float(np.clip(1.0 - abs(delta_x) / max_backtrack_px, 0.0, 1.0))

    if delta_x >= 0:
        return 1.0

    return float(np.clip(1.0 - abs(delta_x) / max_backtrack_px, 0.0, 1.0))


def tracking_score(
    track: Track,
    current: Detection,
    frame_w: int,
    frame_h: int,
) -> float | None:
    reference_box = track.predicted_box()
    iou = bbox_iou(reference_box, current.box)
    center_sim = center_similarity(reference_box, current.box, frame_w, frame_h)
    size_sim = bbox_size_similarity(reference_box, current.box)
    direction_sim = direction_similarity(reference_box, current.box, frame_w)

    diagonal = math.sqrt(frame_w * frame_w + frame_h * frame_h)
    max_center_dist_px = max(1.0, diagonal * TRACKER_MAX_CENTER_DIST_RATIO)

    if center_distance(reference_box, current.box) > max_center_dist_px:
        return None

    if size_sim < TRACKER_MIN_SIZE_SIMILARITY:
        return None

    if track.hits >= 2 and direction_sim <= 0.0:
        return None

    total_weight = (
        TRACKER_IOU_WEIGHT
        + TRACKER_CENTER_WEIGHT
        + TRACKER_SIZE_WEIGHT
        + TRACKER_DIRECTION_WEIGHT
    )

    if total_weight <= 0:
        return None

    return float(
        (
            TRACKER_IOU_WEIGHT * iou
            + TRACKER_CENTER_WEIGHT * center_sim
            + TRACKER_SIZE_WEIGHT * size_sim
            + TRACKER_DIRECTION_WEIGHT * direction_sim
        ) / total_weight
    )


def draw_multiline_text(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
    line_height: int = 26,
) -> None:
    x, y = origin

    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def overlay_mask(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask_bool] = color_bgr
    return cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)


def resize_mask_to_shape(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mask_u8 = mask_bool.astype(np.uint8) * 255
    resized = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def draw_normalized_roi(
    image_bgr: np.ndarray,
    roi: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = image_bgr.shape[:2]
    x_min, y_min, x_max, y_max = roi

    x1 = int(x_min * w)
    y1 = int(y_min * h)
    x2 = int(x_max * w)
    y2 = int(y_max * h)

    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(
        image_bgr,
        label,
        (x1 + 10, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def focal_px_from_hfov(image_width_px: int, hfov_deg: float) -> float:
    hfov_rad = math.radians(hfov_deg)
    return image_width_px / (2.0 * math.tan(hfov_rad / 2.0))


def get_depthpro_focal_px(prediction: dict) -> float | None:
    focallength_px = prediction.get("focallength_px", None)

    if focallength_px is None:
        return None

    if isinstance(focallength_px, torch.Tensor):
        return float(focallength_px.detach().cpu().item())

    return float(focallength_px)


# ============================================================
# Medicion y nubes de puntos
# ============================================================

def estimate_size_from_point_cloud(points_xyz: np.ndarray) -> dict:
    if points_xyz.size == 0 or len(points_xyz) < 4:
        return {
            "x_size_m": 0.0,
            "y_size_m": 0.0,
            "z_size_m": 0.0,
            "bbox_volume_m3": 0.0,
            "pca_length_m": 0.0,
            "pca_width_m": 0.0,
            "pca_thickness_m": 0.0,
            "pca_bbox_volume_m3": 0.0,
            "ellipsoid_volume_m3": 0.0,
        }

    mins = np.percentile(points_xyz, 2, axis=0)
    maxs = np.percentile(points_xyz, 98, axis=0)
    sizes = maxs - mins

    x_size_m = float(sizes[0])
    y_size_m = float(sizes[1])
    z_size_m = float(sizes[2])

    bbox_volume_m3 = float(max(0.0, x_size_m) * max(0.0, y_size_m) * max(0.0, z_size_m))

    centered = points_xyz - np.mean(points_xyz, axis=0, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh.T
        pca_mins = np.percentile(projected, 2, axis=0)
        pca_maxs = np.percentile(projected, 98, axis=0)
        pca_sizes = np.maximum(pca_maxs - pca_mins, 0.0)
        pca_sizes = np.sort(pca_sizes)[::-1]
    except np.linalg.LinAlgError:
        pca_sizes = np.array([x_size_m, y_size_m, z_size_m], dtype=np.float32)

    pca_length_m = float(pca_sizes[0])
    pca_width_m = float(pca_sizes[1])
    pca_thickness_m = float(pca_sizes[2])
    pca_bbox_volume_m3 = float(pca_length_m * pca_width_m * pca_thickness_m)
    ellipsoid_volume_m3 = float((math.pi / 6.0) * pca_bbox_volume_m3)

    return {
        "x_size_m": x_size_m,
        "y_size_m": y_size_m,
        "z_size_m": z_size_m,
        "bbox_volume_m3": bbox_volume_m3,
        "pca_length_m": pca_length_m,
        "pca_width_m": pca_width_m,
        "pca_thickness_m": pca_thickness_m,
        "pca_bbox_volume_m3": pca_bbox_volume_m3,
        "ellipsoid_volume_m3": ellipsoid_volume_m3,
    }


def estimate_oriented_size_from_mask_2d(
    mask_bool: np.ndarray,
    depth_m: float,
    focal_px: float,
) -> dict:
    ys, xs = np.where(mask_bool)

    if xs.size < 10 or focal_px <= 0 or depth_m <= 0:
        return {
            "mask_length_px": 0.0,
            "mask_height_px": 0.0,
            "mask_length_m": 0.0,
            "mask_height_m": 0.0,
        }

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords_centered = coords - np.mean(coords, axis=0, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
        projected = coords_centered @ vh.T
        p2 = np.percentile(projected, 2, axis=0)
        p98 = np.percentile(projected, 98, axis=0)
        sizes_px = np.maximum(p98 - p2, 0.0)
        sizes_px = np.sort(sizes_px)[::-1]
        length_px = float(sizes_px[0])
        height_px = float(sizes_px[1])
    except np.linalg.LinAlgError:
        x_min, x_max = np.percentile(xs, [2, 98])
        y_min, y_max = np.percentile(ys, [2, 98])
        length_px = float(max(x_max - x_min, y_max - y_min))
        height_px = float(min(x_max - x_min, y_max - y_min))

    length_m = length_px * depth_m / focal_px
    height_m = height_px * depth_m / focal_px

    return {
        "mask_length_px": length_px,
        "mask_height_px": height_px,
        "mask_length_m": float(length_m),
        "mask_height_m": float(height_m),
    }


def estimate_fish_empirical_volume(
    length_m: float,
    height_m: float,
    thickness_ratio: float = FISH_THICKNESS_RATIO,
    shape_factor: float = FISH_VOLUME_SHAPE_FACTOR,
) -> dict:
    if length_m <= 0 or height_m <= 0:
        return {
            "fish_thickness_m": 0.0,
            "fish_volume_m3": 0.0,
            "fish_volume_liters": 0.0,
        }

    thickness_m = thickness_ratio * height_m
    volume_m3 = shape_factor * length_m * height_m * thickness_m

    return {
        "fish_thickness_m": float(thickness_m),
        "fish_volume_m3": float(volume_m3),
        "fish_volume_liters": float(volume_m3 * 1000.0),
    }


def passes_detection_filters(detection: dict) -> tuple[bool, str]:
    mask_area_px = detection.get("mask_area_px", 0)
    distance_m = detection.get("distance_median", 0.0)
    length_m = detection.get("mask_length_m", 0.0)

    if mask_area_px < MIN_MASK_AREA_PX:
        return False, f"mask_area_px {mask_area_px} < {MIN_MASK_AREA_PX}"

    if not (MIN_DISTANCE_M <= distance_m <= MAX_DISTANCE_M):
        return False, f"distance {distance_m:.2f} fuera de rango"

    if not (MIN_FISH_LENGTH_M <= length_m <= MAX_FISH_LENGTH_M):
        return False, f"length {length_m:.2f} fuera de rango"

    return True, "ok"


def create_point_cloud_from_mask(
    depth_map: np.ndarray,
    mask_depth: np.ndarray,
    frame_bgr: np.ndarray,
    focal_px: float,
    max_points: int = 60000,
) -> tuple[np.ndarray, np.ndarray | None]:
    depth_h, depth_w = depth_map.shape[:2]

    if focal_px <= 0:
        raise ValueError("focal_px debe ser mayor que cero")

    valid = mask_depth & np.isfinite(depth_map) & (depth_map > 0)
    ys, xs = np.where(valid)

    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32), None

    if xs.size > max_points:
        idx = np.random.choice(xs.size, size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    z = depth_map[ys, xs].astype(np.float32)

    cx = depth_w / 2.0
    cy = depth_h / 2.0
    x = ((xs.astype(np.float32) - cx) * z) / focal_px
    y = ((ys.astype(np.float32) - cy) * z) / focal_px

    points_xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    colors_rgb = None

    if SAVE_POINT_CLOUD_COLOR:
        frame_h, frame_w = frame_bgr.shape[:2]
        scale_x = frame_w / depth_w
        scale_y = frame_h / depth_h
        img_xs = np.clip(np.round(xs * scale_x).astype(int), 0, frame_w - 1)
        img_ys = np.clip(np.round(ys * scale_y).astype(int), 0, frame_h - 1)
        colors_bgr = frame_bgr[img_ys, img_xs]
        colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)

    return points_xyz, colors_rgb


def depth_values_to_rgb(depth_values: np.ndarray) -> np.ndarray:
    if depth_values.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    z_min = float(np.percentile(depth_values, 2))
    z_max = float(np.percentile(depth_values, 98))

    if z_max <= z_min:
        z_max = z_min + 1e-6

    normalized = (depth_values - z_min) / (z_max - z_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    gray = ((1.0 - normalized) * 255).astype(np.uint8)
    colors_bgr = cv2.applyColorMap(gray.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return colors_bgr[:, ::-1].astype(np.uint8)


def create_point_cloud_from_depth_map(
    depth_map: np.ndarray,
    frame_bgr: np.ndarray,
    focal_px: float,
    highlight_mask_depth: np.ndarray | None = None,
    max_points: int = 220000,
) -> tuple[np.ndarray, np.ndarray]:
    depth_h, depth_w = depth_map.shape[:2]

    if focal_px <= 0:
        raise ValueError("focal_px debe ser mayor que cero")

    valid = np.isfinite(depth_map) & (depth_map > 0)
    ys, xs = np.where(valid)

    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    if xs.size > max_points:
        idx = np.random.choice(xs.size, size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    z = depth_map[ys, xs].astype(np.float32)
    cx = depth_w / 2.0
    cy = depth_h / 2.0
    x = ((xs.astype(np.float32) - cx) * z) / focal_px
    y = ((ys.astype(np.float32) - cy) * z) / focal_px

    points_xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    colors_rgb = depth_values_to_rgb(z)

    if highlight_mask_depth is not None:
        mask_hit = highlight_mask_depth[ys, xs]
        colors_rgb[mask_hit] = np.array([0, 255, 40], dtype=np.uint8)

    return points_xyz, colors_rgb


def save_ply(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors_rgb is not None and len(colors_rgb) == len(points_xyz)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        if has_color:
            for point, color in zip(points_xyz, colors_rgb):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        else:
            for point in points_xyz:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


class PointCloudWindow:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and o3d is not None
        self.visualizer = None
        self.point_cloud = None
        self.frame = None
        self.has_geometry = False

        if enabled and o3d is None:
            print("Open3D no esta disponible. Se omitira la ventana 3D.")

    def close(self) -> None:
        if self.visualizer is not None:
            self.visualizer.destroy_window()
            self.visualizer = None
            self.point_cloud = None
            self.frame = None
            self.has_geometry = False

    def poll(self) -> None:
        if self.visualizer is not None:
            self.visualizer.poll_events()
            self.visualizer.update_renderer()

    def update(
        self,
        points_xyz: np.ndarray,
        colors_rgb: np.ndarray | None = None,
        title: str = "Nube de puntos 3D - target track",
    ) -> None:
        if not self.enabled or points_xyz.ndim != 2 or points_xyz.shape[0] == 0:
            return

        if self.visualizer is None:
            self.visualizer = o3d.visualization.Visualizer()
            self.visualizer.create_window(window_name=title, width=960, height=720)

            self.point_cloud = o3d.geometry.PointCloud()
            self.visualizer.add_geometry(self.point_cloud)

            if SHOW_3D_AXIS:
                self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
                self.visualizer.add_geometry(self.frame)

            render_option = self.visualizer.get_render_option()
            render_option.point_size = 1.4
            render_option.background_color = np.asarray([0.02, 0.02, 0.02])

        points_view = points_xyz.copy()
        points_view[:, 1] *= -1.0
        self.point_cloud.points = o3d.utility.Vector3dVector(points_view.astype(np.float64))

        if colors_rgb is not None and len(colors_rgb) == len(points_xyz):
            self.point_cloud.colors = o3d.utility.Vector3dVector(
                colors_rgb.astype(np.float64) / 255.0
            )
        else:
            self.point_cloud.paint_uniform_color([0.1, 0.8, 0.2])

        if self.has_geometry:
            self.visualizer.update_geometry(self.point_cloud)
        else:
            self.has_geometry = True

        if not self.point_cloud.has_points():
            return

        bounding_box = self.point_cloud.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()
        self.visualizer.reset_view_point(True)
        view_control = self.visualizer.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0.0, -0.35, -1.0])
        view_control.set_up([0.0, -1.0, 0.0])
        view_control.set_zoom(0.55)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()


class PointCloudViewerManager:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and o3d is not None
        self.windows: dict[str, PointCloudWindow] = {}

        if enabled and o3d is None:
            print("Open3D no esta disponible. Se omitiran las ventanas 3D.")

    def poll(self) -> None:
        for window in self.windows.values():
            window.poll()

    def update(
        self,
        window_key: str,
        points_xyz: np.ndarray,
        colors_rgb: np.ndarray | None = None,
        title: str = "Nube de puntos 3D",
    ) -> None:
        if not self.enabled:
            return

        window = self.windows.get(window_key)

        if window is None:
            window = PointCloudWindow(enabled=self.enabled)
            self.windows[window_key] = window

        window.update(points_xyz, colors_rgb, title)

    def has_open_windows(self) -> bool:
        return any(window.visualizer is not None for window in self.windows.values())

    def close(self) -> None:
        for window in self.windows.values():
            window.close()
        self.windows.clear()


# ============================================================
# Zonas y seleccion de target
# ============================================================

def is_in_entry_zone(detection: Detection, frame_w: int, frame_h: int) -> bool:
    center_x, center_y = bbox_center(detection.box)

    if frame_w <= 0 or frame_h <= 0:
        return False

    x_ratio = center_x / frame_w
    y_ratio = center_y / frame_h
    in_y = ENTRY_MIN_Y_RATIO <= y_ratio <= ENTRY_MAX_Y_RATIO

    if ENTRY_SIDE == "right":
        in_x = (1.0 - ENTRY_MAX_X_RATIO) <= x_ratio <= 1.0
    else:
        in_x = 0.0 <= x_ratio <= ENTRY_MAX_X_RATIO

    return in_x and in_y


def is_in_measurement_roi(detection: Detection, frame_w: int, frame_h: int) -> bool:
    center_x, center_y = bbox_center(detection.box)

    if frame_w <= 0 or frame_h <= 0:
        return False

    x_ratio = center_x / frame_w
    y_ratio = center_y / frame_h
    x_min, y_min, x_max, y_max = MEASURE_ROI

    return x_min <= x_ratio <= x_max and y_min <= y_ratio <= y_max


def has_track_passed_measurement_roi(track: Track, frame_w: int) -> bool:
    if frame_w <= 0:
        return False

    center_x, _ = bbox_center(track.detection.box)
    x_ratio = center_x / frame_w
    x_min, _, x_max, _ = MEASURE_ROI

    if ENTRY_SIDE == "right":
        return x_ratio < x_min

    return x_ratio > x_max


def pick_target_tracks(
    tracks: list[Track],
    active_session_ids: set[int],
    frame_w: int,
    frame_h: int,
    max_targets: int,
) -> list[Track]:
    eligible = [
        track
        for track in tracks
        if track.track_id not in active_session_ids
        and track.is_active
        and track.hits >= TARGET_MIN_HITS
        and is_in_measurement_roi(track.detection, frame_w, frame_h)
    ]

    def sort_key(track: Track) -> tuple[float, float, float]:
        center_x, _ = bbox_center(track.detection.box)
        directional_x = -center_x if ENTRY_SIDE == "right" else center_x
        return (-track.hits, -track.detection.confidence, directional_x)

    eligible.sort(key=sort_key)
    return eligible[:max_targets]


def find_track_by_id(tracks: list[Track], track_id: int | None) -> Track | None:
    if track_id is None:
        return None

    for track in tracks:
        if track.track_id == track_id:
            return track

    return None


# ============================================================
# YOLO / SAM / Depth Pro
# ============================================================

def extract_sam_masks(
    frame_bgr: np.ndarray,
    sam_model: SAM,
    bboxes: list[list[int]],
    yolo_device,
) -> list[np.ndarray]:
    if not bboxes:
        return []

    results = sam_model.predict(
        source=frame_bgr,
        bboxes=bboxes,
        device=yolo_device,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]

    if result.masks is None or result.masks.data is None:
        return []

    masks_tensor = result.masks.data.detach().cpu()
    masks_np = masks_tensor.numpy()
    frame_h, frame_w = frame_bgr.shape[:2]
    masks_bool: list[np.ndarray] = []

    for mask in masks_np:
        mask_bool = mask > 0.5

        if mask_bool.shape[:2] != (frame_h, frame_w):
            mask_bool = resize_mask_to_shape(mask_bool, frame_h, frame_w)

        masks_bool.append(mask_bool)

    return masks_bool


def load_models(device: torch.device, precision: torch.dtype):
    yolo_device = 0 if device.type == "cuda" else "cpu"

    print(f"Cargando YOLO: {YOLO_WEIGHTS}")
    yolo_model = YOLO(YOLO_WEIGHTS)

    if TARGET_CLASSES and "world" in YOLO_WEIGHTS.lower():
        yolo_model.set_classes(TARGET_CLASSES)
        print(f"Clases YOLO-World configuradas: {TARGET_CLASSES}")

    print(f"Cargando SAM: {SAM_WEIGHTS}")
    sam_model = SAM(SAM_WEIGHTS)

    print("Cargando Depth Pro...")
    depth_model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=precision,
    )
    depth_model.eval()

    return yolo_model, sam_model, depth_model, transform, yolo_device


def detect_objects(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    yolo_device: int | str,
) -> list[Detection]:
    frame_h, frame_w = frame_bgr.shape[:2]

    results = yolo_model.predict(
        source=frame_bgr,
        conf=CONFIDENCE_THRESHOLD,
        device=yolo_device,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return []

    detections: list[Detection] = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()

        x1i = clamp(int(round(x1)), 0, frame_w - 1)
        y1i = clamp(int(round(y1)), 0, frame_h - 1)
        x2i = clamp(int(round(x2)), 0, frame_w - 1)
        y2i = clamp(int(round(y2)), 0, frame_h - 1)

        if x2i <= x1i or y2i <= y1i:
            continue

        class_id = int(box.cls.item())
        class_name = yolo_model.names.get(class_id, str(class_id))
        confidence = float(box.conf.item())

        if TARGET_CLASS_SET and class_name not in TARGET_CLASS_SET:
            continue

        detections.append(
            Detection(
                box=[x1i, y1i, x2i, y2i],
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
        )

    return detections


class EntryZoneMultiTracker:
    def __init__(self) -> None:
        self.next_track_id = 1
        self.tracks: list[Track] = []

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def update(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
    ) -> list[Track]:
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))
        candidates: list[tuple[float, int, int]] = []

        for track_idx, track in enumerate(self.tracks):
            for det_idx, detection in enumerate(detections):
                score = tracking_score(track, detection, frame_w, frame_h)

                if score is not None and score >= TRACKER_MIN_SCORE:
                    candidates.append((score, track_idx, det_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])

        for score, track_idx, det_idx in candidates:
            if track_idx not in unmatched_tracks:
                continue

            if det_idx not in unmatched_detections:
                continue

            self.tracks[track_idx].update(detections[det_idx])
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missing()

        for det_idx in unmatched_detections:
            detection = detections[det_idx]

            if detection.confidence < TARGET_CONFIDENCE_THRESHOLD:
                continue

            if not is_in_entry_zone(detection, frame_w, frame_h):
                continue

            track = Track(track_id=self.next_track_id, detection=detection)
            self.next_track_id += 1
            self.tracks.append(track)

            print(
                f"Nuevo track desde zona {ENTRY_SIDE}: "
                f"ID {track.track_id} | {detection.class_name} | conf={detection.confidence:.2f}"
            )

        self.tracks = [track for track in self.tracks if track.is_alive]
        return self.tracks


# ============================================================
# Segmentacion y medicion del target bloqueado
# ============================================================

def process_target_detection(
    frame_bgr: np.ndarray,
    detection: Detection,
    sam_model: SAM,
    depth_model,
    transform,
    yolo_device,
) -> tuple[np.ndarray, dict | None]:
    annotated = frame_bgr.copy()
    draw_normalized_roi(annotated, MEASURE_ROI, "ROI medicion", (0, 255, 255), 2)

    masks = extract_sam_masks(
        frame_bgr=frame_bgr,
        sam_model=sam_model,
        bboxes=[detection.box],
        yolo_device=yolo_device,
    )

    if not masks:
        cv2.putText(
            annotated,
            "SAM no genero mascara para target",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, None

    mask_orig = masks[0]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_input = transform(frame_rgb)

    with torch.inference_mode():
        prediction = depth_model.infer(depth_input, f_px=None)

    depth_map = prediction["depth"].detach().cpu().numpy().squeeze()
    depth_h, depth_w = depth_map.shape[:2]
    focal_px_lens = focal_px_from_hfov(frame_bgr.shape[1], LENS_HFOV_DEG)
    focal_px_depthpro = get_depthpro_focal_px(prediction)

    if focal_px_depthpro is None or focal_px_depthpro <= 0:
        focal_px_used = focal_px_lens
        focal_source = "HFOV"
    else:
        focal_px_used = focal_px_depthpro
        focal_source = "Depth Pro"

    mask_depth = resize_mask_to_shape(mask_orig, depth_h, depth_w)
    valid_depths = depth_map[mask_depth & np.isfinite(depth_map) & (depth_map > 0)]

    if valid_depths.size == 0:
        cv2.putText(
            annotated,
            "Target sin profundidad valida",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, None

    distance_p10 = float(np.percentile(valid_depths, 10))
    distance_p25 = float(np.percentile(valid_depths, 25))
    distance_median = float(np.median(valid_depths))
    distance_p75 = float(np.percentile(valid_depths, 75))
    distance_p90 = float(np.percentile(valid_depths, 90))
    distance_mean = float(np.mean(valid_depths))
    distance_min = float(np.min(valid_depths))
    distance_max = float(np.max(valid_depths))

    points_xyz, colors_rgb = create_point_cloud_from_mask(
        depth_map=depth_map,
        mask_depth=mask_depth,
        frame_bgr=frame_bgr,
        focal_px=focal_px_used,
        max_points=MAX_POINT_CLOUD_POINTS,
    )

    if points_xyz.size == 0:
        return annotated, None

    size_info_3d = estimate_size_from_point_cloud(points_xyz)
    size_info_mask = estimate_oriented_size_from_mask_2d(
        mask_bool=mask_orig,
        depth_m=distance_median,
        focal_px=focal_px_used,
    )
    volume_info = estimate_fish_empirical_volume(
        length_m=size_info_mask["mask_length_m"],
        height_m=size_info_mask["mask_height_m"],
    )

    detection_info = {
        "track_id": detection.track_id,
        "box_int": detection.box,
        "conf": detection.confidence,
        "name": detection.class_name,
        "mask_orig": mask_orig,
        "mask_depth": mask_depth,
        "mask_area_px": int(np.count_nonzero(mask_orig)),
        "distance_p10": distance_p10,
        "distance_p25": distance_p25,
        "distance_median": distance_median,
        "distance_p75": distance_p75,
        "distance_p90": distance_p90,
        "distance_mean": distance_mean,
        "distance_min": distance_min,
        "distance_max": distance_max,
        "points_xyz": points_xyz,
        "colors_rgb": colors_rgb,
    }
    detection_info.update(size_info_3d)
    detection_info.update(size_info_mask)
    detection_info.update(volume_info)

    passed, reason = passes_detection_filters(detection_info)
    detection_info["filter_passed"] = passed
    detection_info["filter_reason"] = reason

    depth_iqr = max(1e-6, distance_p75 - distance_p25)
    detection_info["depth_iqr"] = depth_iqr
    detection_info["quality_score"] = float(
        detection.confidence * math.log1p(detection_info["mask_area_px"]) / (1.0 + depth_iqr)
    )

    scene_points_xyz = None
    scene_colors_rgb = None

    if SAVE_FULL_DEPTH_POINT_CLOUD:
        scene_points_xyz, scene_colors_rgb = create_point_cloud_from_depth_map(
            depth_map=depth_map,
            frame_bgr=frame_bgr,
            focal_px=focal_px_used,
            highlight_mask_depth=mask_depth,
            max_points=MAX_DEPTH_VIEWER_POINTS,
        )

    color = (0, 255, 0) if passed else (0, 0, 255)
    alpha = 0.25 if passed else 0.12
    annotated = overlay_mask(annotated, mask_orig, color, alpha)

    x1, y1, x2, y2 = detection.box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    label = (
        f"target ID {detection.track_id} {detection.class_name} "
        f"{distance_median:.2f}m L={detection_info['mask_length_m']:.2f}m"
    )
    cv2.putText(
        annotated,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    summary_lines = [
        f"Target track {detection.track_id}",
        f"Dist p25/med: {distance_p25:.2f}/{distance_median:.2f} m",
        f"L/H mask: {detection_info['mask_length_m']:.2f} x {detection_info['mask_height_m']:.2f} m",
        f"Esp est: {detection_info['fish_thickness_m']:.2f} m",
        f"Vol emp: {detection_info['fish_volume_liters']:.2f} L",
        f"Pts: {len(points_xyz)}",
        f"fx: {focal_px_used:.1f}px ({focal_source})",
        f"Filtro: {reason}",
    ]
    draw_multiline_text(annotated, summary_lines, (20, 40), color, 0.72, 2, 29)

    result_info = {
        "closest_obj": detection_info,
        "focal_px_used": focal_px_used,
        "focal_source": focal_source,
        "scene_points_xyz": scene_points_xyz,
        "scene_colors_rgb": scene_colors_rgb,
    }
    return annotated, result_info


def process_target_session(
    session: TargetSession,
    sam_model: SAM,
    depth_model,
    transform,
    yolo_device,
) -> TargetSession:
    print()
    print(
        f"Procesando target session track {session.track_id} "
        f"con {len(session.frames)} frames trackeados..."
    )

    for tracked_frame in session.frames:
        annotated, result_info = process_target_detection(
            frame_bgr=tracked_frame.frame_bgr,
            detection=tracked_frame.detection,
            sam_model=sam_model,
            depth_model=depth_model,
            transform=transform,
            yolo_device=yolo_device,
        )

        if result_info is None:
            continue

        session.update_best(annotated, result_info)

    if session.best_result_info is None:
        print(f"Track {session.track_id}: no hubo mediciones validas tras el postproceso.")
    else:
        best_obj = session.best_result_info["closest_obj"]
        print(
            f"Track {session.track_id}: mejor medicion "
            f"dist={best_obj['distance_median']:.2f}m "
            f"L={best_obj['mask_length_m']:.2f}m "
            f"Vol={best_obj['fish_volume_liters']:.2f}L "
            f"score={best_obj.get('quality_score', 0.0):.3f}"
        )

    return session


def save_target_session_artifacts(session: TargetSession) -> None:
    if not SAVE_TARGET_SESSION_ARTIFACTS:
        return

    if session.best_result_info is None or session.best_annotated is None:
        print(f"Track {session.track_id}: sin medicion valida para guardar.")
        return

    closest_obj = session.best_result_info["closest_obj"]
    base_name = f"track_{session.track_id}_{session.started_timestamp}"

    result_path = OUTPUT_DIR / f"{base_name}_best.jpg"
    mask_path = OUTPUT_DIR / f"{base_name}_mask.png"
    npy_path = OUTPUT_DIR / f"{base_name}_pointcloud.npy"
    ply_path = OUTPUT_DIR / f"{base_name}_pointcloud.ply"

    cv2.imwrite(str(result_path), session.best_annotated)
    cv2.imwrite(str(mask_path), closest_obj["mask_orig"].astype(np.uint8) * 255)
    np.save(str(npy_path), closest_obj["points_xyz"])
    save_ply(ply_path, closest_obj["points_xyz"], closest_obj["colors_rgb"])

    scene_points_xyz = session.best_result_info.get("scene_points_xyz")
    scene_colors_rgb = session.best_result_info.get("scene_colors_rgb")

    if scene_points_xyz is not None and scene_colors_rgb is not None:
        scene_ply_path = OUTPUT_DIR / f"{base_name}_scene.ply"
        save_ply(scene_ply_path, scene_points_xyz, scene_colors_rgb)

    print()
    print(f"Track {session.track_id}: mejor nube guardada.")
    print(f"Resultado: {result_path}")
    print(f"Mascara:   {mask_path}")
    print(f"NPY:       {npy_path}")
    print(f"PLY:       {ply_path}")


# ============================================================
# Visualizacion
# ============================================================

def track_color(track_id: int) -> tuple[int, int, int]:
    colors = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (128, 255, 0),
        (255, 128, 0),
    ]
    return colors[(track_id - 1) % len(colors)]


def draw_entry_zone(frame_bgr: np.ndarray) -> None:
    frame_h, frame_w = frame_bgr.shape[:2]
    x_width = int(frame_w * float(np.clip(ENTRY_MAX_X_RATIO, 0.05, 1.0)))
    y1 = int(frame_h * float(np.clip(ENTRY_MIN_Y_RATIO, 0.0, 1.0)))
    y2 = int(frame_h * float(np.clip(ENTRY_MAX_Y_RATIO, 0.0, 1.0)))

    if ENTRY_SIDE == "right":
        x1 = max(0, frame_w - x_width)
        x2 = frame_w
        zone_label = "zona entrada derecha"
    else:
        x1 = 0
        x2 = x_width
        zone_label = "zona entrada izquierda"

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        zone_label,
        (x1 + 10, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_trail(
    frame_bgr: np.ndarray,
    trail_points: deque[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    points = list(trail_points)

    if len(points) < 2:
        return

    for idx in range(1, len(points)):
        p1 = points[idx - 1]
        p2 = points[idx]
        thickness = max(1, int(1 + 5 * idx / len(points)))
        cv2.line(frame_bgr, p1, p2, color, thickness, cv2.LINE_AA)

    cv2.circle(frame_bgr, points[-1], 6, color, -1)


def draw_track(frame_bgr: np.ndarray, track: Track, active_session_ids: set[int]) -> None:
    is_selected = track.track_id in active_session_ids
    color = (255, 255, 255) if is_selected else track_color(track.track_id)
    x1, y1, x2, y2 = track.detection.box

    draw_trail(frame_bgr, track.trail, color)
    thickness = 4 if is_selected else (3 if track.is_active else 1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    label = (
        f"ID {track.track_id} | {track.detection.class_name} | "
        f"conf={track.detection.confidence:.2f} | hits={track.hits} | miss={track.missing}"
    )
    cv2.putText(
        frame_bgr,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    tracks: list[Track],
    active_sessions: dict[int, TargetSession],
    num_targets_started: int,
) -> None:
    draw_entry_zone(frame_bgr)
    draw_normalized_roi(frame_bgr, MEASURE_ROI, "ROI medicion", (0, 255, 255), 2)
    active_session_ids = set(active_sessions.keys())

    for track in tracks:
        draw_track(frame_bgr, track, active_session_ids)

    active_tracks = sum(1 for track in tracks if track.is_active)
    session_text = (
        "targets activos: " + ", ".join(str(track_id) for track_id in sorted(active_session_ids))
        if active_session_ids
        else "targets activos: ninguno"
    )

    cv2.putText(
        frame_bgr,
        f"tracks activos: {active_tracks}",
        (20, frame_bgr.shape[0] - 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        session_text,
        (20, frame_bgr.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_bgr,
        f"objetivos seleccionados: {num_targets_started}/{MAX_TARGET_OBJECTS}",
        (20, frame_bgr.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_bgr,
        "q: salir | r: reset tracks",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def update_3d_viewer(
    viewer_manager: PointCloudViewerManager,
    result_info: dict | None,
    track_id: int | None = None,
) -> None:
    if not result_info:
        return

    closest_obj = result_info.get("closest_obj")
    if closest_obj is None:
        return

    points_xyz = closest_obj["points_xyz"]
    if points_xyz.size == 0:
        return

    colors_rgb = depth_values_to_rgb(points_xyz[:, 2])
    effective_track_id = track_id if track_id is not None else closest_obj.get("track_id")
    window_key = f"track_{effective_track_id}" if effective_track_id is not None else "target"
    title = (
        f"Nube 3D - track {effective_track_id}"
        if effective_track_id is not None
        else "Nube 3D - target"
    )
    viewer_manager.update(window_key, points_xyz, colors_rgb, title)


def finalize_target_session(
    session: TargetSession,
    reason: str,
    sam_model: SAM,
    depth_model,
    transform,
    yolo_device,
    viewer_manager: PointCloudViewerManager,
) -> np.ndarray | None:
    session.final_reason = reason
    print()
    print(f"Track {session.track_id}: cerrando target session. Motivo: {reason}")
    process_target_session(session, sam_model, depth_model, transform, yolo_device)
    save_target_session_artifacts(session)
    update_3d_viewer(viewer_manager, session.best_result_info, session.track_id)
    return session.best_annotated


def hold_results_window(
    last_result_image: np.ndarray | None,
    viewer_manager: PointCloudViewerManager,
) -> None:
    if not KEEP_RESULTS_WINDOW_ON_EXIT:
        return

    if not SHOW_WINDOWS and not viewer_manager.has_open_windows():
        return

    if last_result_image is not None and SHOW_WINDOWS:
        preview = last_result_image.copy()
        cv2.putText(
            preview,
            "Resultados finales - q/ESC para cerrar",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, preview)

    while True:
        viewer_manager.poll()
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break


# ============================================================
# Programa principal
# ============================================================

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.half if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    print(f"Fuente video: {VIDEO_SOURCE}")
    print(f"ENTRY_SIDE: {ENTRY_SIDE}")
    print(f"ROI medicion: {MEASURE_ROI}")
    print(f"RTSP transport: {RTSP_TRANSPORT}")

    yolo_model, sam_model, depth_model, transform, yolo_device = load_models(device, precision)
    viewer_manager = PointCloudViewerManager(enabled=ENABLE_3D_VIEWER)

    cap = open_video_capture()
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de video")

    tracker = EntryZoneMultiTracker()
    tracks: list[Track] = []
    active_sessions: dict[int, TargetSession] = {}
    num_targets_started = 0
    frame_counter = 0
    display_delay_ms = int(1000 / max(DISPLAY_FPS, 1.0))
    read_failures = 0
    last_result_image: np.ndarray | None = None

    try:
        while True:
            viewer_manager.poll()

            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                print(
                    f"No se pudo leer frame del stream "
                    f"({read_failures}/{MAX_STREAM_READ_FAILURES})."
                )

                if USE_RTSP and read_failures < MAX_STREAM_READ_FAILURES:
                    cap.release()
                    cap = open_video_capture()

                    if cap.isOpened():
                        print("Reconexion RTSP exitosa.")
                        continue

                    print("Fallo la reconexion RTSP.")
                    continue

                print("Se alcanzo el maximo de fallos de lectura. Finalizando stream.")
                break

            read_failures = 0

            frame_counter += 1
            frame_h, frame_w = frame.shape[:2]

            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                detections = detect_objects(frame, yolo_model, yolo_device)
                tracks = tracker.update(detections, frame_w, frame_h)

                finished_session_ids: list[int] = []

                for track_id, session in active_sessions.items():
                    track = find_track_by_id(tracks, track_id)

                    if track is None or not track.is_alive:
                        result_image = finalize_target_session(
                            session=session,
                            reason="track perdido",
                            sam_model=sam_model,
                            depth_model=depth_model,
                            transform=transform,
                            yolo_device=yolo_device,
                            viewer_manager=viewer_manager,
                        )
                        if result_image is not None:
                            last_result_image = result_image
                        finished_session_ids.append(track_id)
                        continue

                    if has_track_passed_measurement_roi(track, frame_w):
                        result_image = finalize_target_session(
                            session=session,
                            reason="track salio de ROI/visualizacion util",
                            sam_model=sam_model,
                            depth_model=depth_model,
                            transform=transform,
                            yolo_device=yolo_device,
                            viewer_manager=viewer_manager,
                        )
                        if result_image is not None:
                            last_result_image = result_image
                        finished_session_ids.append(track_id)
                        continue

                    if track.is_active and is_in_measurement_roi(track.detection, frame_w, frame_h):
                        session.append_frame(
                            frame_index=frame_counter,
                            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                            frame_bgr=frame,
                            detection=track.detection,
                        )

                for track_id in finished_session_ids:
                    active_sessions.pop(track_id, None)

                remaining_slots = max(0, MAX_TARGET_OBJECTS - num_targets_started)

                if remaining_slots > 0:
                    new_targets = pick_target_tracks(
                        tracks=tracks,
                        active_session_ids=set(active_sessions.keys()),
                        frame_w=frame_w,
                        frame_h=frame_h,
                        max_targets=remaining_slots,
                    )

                    for track in new_targets:
                        session = TargetSession(
                            track_id=track.track_id,
                            started_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        )
                        session.append_frame(
                            frame_index=frame_counter,
                            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                            frame_bgr=frame,
                            detection=track.detection,
                        )
                        active_sessions[track.track_id] = session
                        num_targets_started += 1
                        print(
                            f"Track {track.track_id}: target session creada "
                            f"({num_targets_started}/{MAX_TARGET_OBJECTS})."
                        )

            display = frame.copy()
            draw_overlay(display, tracks, active_sessions, num_targets_started)

            if SHOW_WINDOWS:
                cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(display_delay_ms) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                for session in list(active_sessions.values()):
                    result_image = finalize_target_session(
                        session=session,
                        reason="reset manual",
                        sam_model=sam_model,
                        depth_model=depth_model,
                        transform=transform,
                        yolo_device=yolo_device,
                        viewer_manager=viewer_manager,
                    )
                    if result_image is not None:
                        last_result_image = result_image
                tracker.reset()
                tracks = []
                active_sessions.clear()
                num_targets_started = 0
                print("Tracks y target session reiniciados.")

    finally:
        for session in list(active_sessions.values()):
            result_image = finalize_target_session(
                session=session,
                reason="cierre del programa",
                sam_model=sam_model,
                depth_model=depth_model,
                transform=transform,
                yolo_device=yolo_device,
                viewer_manager=viewer_manager,
            )
            if result_image is not None:
                last_result_image = result_image
        active_sessions.clear()

        hold_results_window(last_result_image, viewer_manager)
        cap.release()
        viewer_manager.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
