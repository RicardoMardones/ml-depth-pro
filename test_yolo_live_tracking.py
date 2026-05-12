from datetime import datetime
from pathlib import Path
from collections import deque
import math
import os
import queue
import threading

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
# Configuracion RTSP
# ============================================================

USER = os.getenv("RTSP_USER", "admin")
PASSWORD = os.getenv("RTSP_PASSWORD", "itg24chile")
NVR_IP = os.getenv("RTSP_IP", "10.22.100.22")
PORT = int(os.getenv("RTSP_PORT", "554"))
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "202")

RTSP_URL = f"rtsp://{USER}:{PASSWORD}@{NVR_IP}:{PORT}/Streaming/Channels/{RTSP_CHANNEL}"

USE_RTSP = os.getenv("USE_RTSP", "1") == "1"
TEST_IMAGE_PATH = os.getenv("TEST_IMAGE_PATH", "huelmo_cap2_small.png")


# ============================================================
# Configuracion lente SL-0041
# ============================================================

LENS_HFOV_DEG = 127.3


# ============================================================
# Configuracion YOLO-World / SAM / Depth Pro
# ============================================================

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s-worldv2.pt")
SAM_WEIGHTS = os.getenv("SAM_WEIGHTS", "mobile_sam.pt")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.20"))

TARGET_CLASSES = [
    "fish",
    "salmon fish",
    "trout fish",
]

TARGET_CLASS_SET = set(TARGET_CLASSES)
TARGET_LABEL = "pez"


# ============================================================
# Configuracion tracker
# ============================================================

ENABLE_TRACKING = os.getenv("ENABLE_TRACKING", "1") == "1"

TRACKER_IOU_WEIGHT = float(os.getenv("TRACKER_IOU_WEIGHT", "0.55"))
TRACKER_CENTER_WEIGHT = float(os.getenv("TRACKER_CENTER_WEIGHT", "0.30"))
TRACKER_DEPTH_WEIGHT = float(os.getenv("TRACKER_DEPTH_WEIGHT", "0.15"))

TRACKER_MIN_SCORE = float(os.getenv("TRACKER_MIN_SCORE", "0.25"))
TRACKER_MAX_MISSING = int(os.getenv("TRACKER_MAX_MISSING", "5"))


# ============================================================
# Configuracion seguimiento en vivo
# ============================================================

LIVE_TRACKING_ENABLED = os.getenv("LIVE_TRACKING_ENABLED", "1") == "1"

# Procesa 1 frame cada N frames para no congelar el stream.
LIVE_PROCESS_EVERY_N_FRAMES = int(os.getenv("LIVE_PROCESS_EVERY_N_FRAMES", "8"))

# Longitud de la estela.
TRAIL_MAX_POINTS = int(os.getenv("TRAIL_MAX_POINTS", "80"))

# Si True, el procesamiento corre en segundo plano.
BACKGROUND_PROCESSING = os.getenv("BACKGROUND_PROCESSING", "1") == "1"


# ============================================================
# Configuracion de nubes de puntos / visualizacion
# ============================================================

MAX_POINT_CLOUD_POINTS = int(os.getenv("MAX_POINT_CLOUD_POINTS", "60000"))
MAX_DEPTH_VIEWER_POINTS = int(os.getenv("MAX_DEPTH_VIEWER_POINTS", "220000"))

SAVE_POINT_CLOUD_COLOR = os.getenv("SAVE_POINT_CLOUD_COLOR", "1") == "1"
SAVE_FULL_DEPTH_POINT_CLOUD = os.getenv("SAVE_FULL_DEPTH_POINT_CLOUD", "0") == "1"

ENABLE_3D_VIEWER = os.getenv("ENABLE_3D_VIEWER", "1") == "1"
SHOW_WINDOWS = os.getenv("SHOW_WINDOWS", "1") == "1"
SHOW_3D_AXIS = os.getenv("SHOW_3D_AXIS", "0") == "1"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs_stream"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Filtros de calidad para detecciones / mascaras
# ============================================================

MIN_MASK_AREA_PX = int(os.getenv("MIN_MASK_AREA_PX", "1500"))
MIN_DISTANCE_M = float(os.getenv("MIN_DISTANCE_M", "0.20"))
MAX_DISTANCE_M = float(os.getenv("MAX_DISTANCE_M", "4.00"))

MIN_FISH_LENGTH_M = float(os.getenv("MIN_FISH_LENGTH_M", "0.05"))
MAX_FISH_LENGTH_M = float(os.getenv("MAX_FISH_LENGTH_M", "1.50"))

FISH_THICKNESS_RATIO = float(os.getenv("FISH_THICKNESS_RATIO", "0.45"))
FISH_VOLUME_SHAPE_FACTOR = float(os.getenv("FISH_VOLUME_SHAPE_FACTOR", "0.55"))


# ============================================================
# Funciones auxiliares generales
# ============================================================

def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


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


# ============================================================
# Medicion / volumen / nube de puntos
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

    bbox_volume_m3 = float(
        max(0.0, x_size_m) * max(0.0, y_size_m) * max(0.0, z_size_m)
    )

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
    fx = focal_px
    fy = focal_px

    x = ((xs.astype(np.float32) - cx) * z) / fx
    y = ((ys.astype(np.float32) - cy) * z) / fy

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


def save_ply(
    path: Path,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
) -> None:
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


# ============================================================
# Open3D Viewer
# ============================================================

class PointCloudViewer:
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
        title: str = "Nube de puntos 3D - objeto seguido",
    ) -> None:
        if not self.enabled or points_xyz.size == 0:
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


# ============================================================
# YOLO / SAM / modelos
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

    masks_bool = []

    for mask in masks_np:
        mask_bool = mask > 0.5

        if mask_bool.shape[:2] != (frame_h, frame_w):
            mask_bool = resize_mask_to_shape(mask_bool, frame_h, frame_w)

        masks_bool.append(mask_bool)

    return masks_bool


def load_models(device: torch.device, precision: torch.dtype):
    print("Cargando YOLO-World...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.set_classes(TARGET_CLASSES)

    print(f"Clases configuradas: {TARGET_CLASSES}")
    print(f"Clases del modelo YOLO: {yolo_model.names}")

    print("Cargando SAM/MobileSAM...")
    sam_model = SAM(SAM_WEIGHTS)

    print("Cargando Depth Pro...")
    depth_model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=precision,
    )
    depth_model.eval()

    return yolo_model, sam_model, depth_model, transform


# ============================================================
# Tracking simple por IoU + centro + profundidad
# ============================================================

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


def bbox_center(box: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def normalized_center_similarity(
    box_a: list[int],
    box_b: list[int],
    frame_w: int,
    frame_h: int,
) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)

    diag = math.sqrt(frame_w * frame_w + frame_h * frame_h)

    if diag <= 0:
        return 0.0

    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    sim = 1.0 - dist / diag

    return float(np.clip(sim, 0.0, 1.0))


def depth_similarity(depth_a: float, depth_b: float, max_delta_m: float = 1.0) -> float:
    if not np.isfinite(depth_a) or not np.isfinite(depth_b):
        return 0.0

    delta = abs(depth_a - depth_b)
    sim = 1.0 - delta / max_delta_m

    return float(np.clip(sim, 0.0, 1.0))


class FishTrack:
    def __init__(self, track_id: int, detection: dict, frame_idx: int) -> None:
        self.track_id = track_id
        self.last_detection = detection
        self.first_frame_idx = frame_idx
        self.last_frame_idx = frame_idx
        self.hits = 1
        self.missing = 0
        self.history = [detection]

    def update(self, detection: dict, frame_idx: int) -> None:
        detection["track_id"] = self.track_id

        self.last_detection = detection
        self.last_frame_idx = frame_idx
        self.hits += 1
        self.missing = 0
        self.history.append(detection)

    def mark_missing(self) -> None:
        self.missing += 1

    @property
    def is_active(self) -> bool:
        return self.missing == 0


class FishTracker:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        min_score: float = TRACKER_MIN_SCORE,
        max_missing: int = TRACKER_MAX_MISSING,
    ) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.min_score = min_score
        self.max_missing = max_missing

        self.next_track_id = 1
        self.tracks: list[FishTrack] = []

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def association_score(self, track: FishTrack, detection: dict) -> float:
        prev = track.last_detection

        iou = bbox_iou(prev["box_int"], detection["box_int"])

        center_sim = normalized_center_similarity(
            prev["box_int"],
            detection["box_int"],
            self.frame_w,
            self.frame_h,
        )

        depth_sim = depth_similarity(
            prev.get("distance_median", 0.0),
            detection.get("distance_median", 0.0),
            max_delta_m=1.0,
        )

        score = (
            TRACKER_IOU_WEIGHT * iou
            + TRACKER_CENTER_WEIGHT * center_sim
            + TRACKER_DEPTH_WEIGHT * depth_sim
        )

        return float(score)

    def update(self, detections: list[dict], frame_idx: int) -> list[FishTrack]:
        for detection in detections:
            detection["track_id"] = None

        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        candidates = []

        for track_idx, track in enumerate(self.tracks):
            for det_idx, detection in enumerate(detections):
                score = self.association_score(track, detection)

                if score >= self.min_score:
                    candidates.append((score, track_idx, det_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])

        for score, track_idx, det_idx in candidates:
            if track_idx not in unmatched_tracks:
                continue

            if det_idx not in unmatched_detections:
                continue

            track = self.tracks[track_idx]
            track.update(detections[det_idx], frame_idx)

            detections[det_idx]["track_score"] = score

            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missing()

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track = FishTrack(
                track_id=self.next_track_id,
                detection=detection,
                frame_idx=frame_idx,
            )

            detection["track_id"] = self.next_track_id
            detection["track_score"] = 1.0

            self.next_track_id += 1
            self.tracks.append(track)

        self.tracks = [
            track for track in self.tracks
            if track.missing <= self.max_missing
        ]

        return self.tracks

    def get_track_by_id(self, track_id: int | None) -> FishTrack | None:
        if track_id is None:
            return None

        for track in self.tracks:
            if track.track_id == track_id:
                return track

        return None


# ============================================================
# Procesamiento de frame pesado: YOLO + SAM + Depth
# ============================================================

def process_frame(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    sam_model: SAM,
    depth_model,
    transform,
    yolo_device,
    timestamp: str,
    save_artifacts: bool = False,
    tracker: FishTracker | None = None,
    frame_idx: int = 0,
    target_track_id: int | None = None,
) -> tuple[np.ndarray, dict | None]:
    orig_h, orig_w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    focal_px_lens = focal_px_from_hfov(orig_w, LENS_HFOV_DEG)

    # ========================================================
    # 1. Deteccion YOLO-World
    # ========================================================

    yolo_results = yolo_model.predict(
        source=frame_bgr,
        conf=CONFIDENCE_THRESHOLD,
        device=yolo_device,
        verbose=False,
    )

    boxes = yolo_results[0].boxes if yolo_results else None

    if boxes is None or len(boxes) == 0:
        return annotated, None

    detections = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
        area = max(0.0, float((x2 - x1) * (y2 - y1)))

        if area <= 0:
            continue

        cls_id = int(box.cls.item())
        class_name = yolo_model.names.get(cls_id, str(cls_id))

        if class_name not in TARGET_CLASS_SET:
            continue

        x1i = clamp(int(round(x1)), 0, orig_w - 1)
        x2i = clamp(int(round(x2)), 0, orig_w - 1)
        y1i = clamp(int(round(y1)), 0, orig_h - 1)
        y2i = clamp(int(round(y2)), 0, orig_h - 1)

        if x2i <= x1i or y2i <= y1i:
            continue

        detections.append(
            {
                "box_int": [x1i, y1i, x2i, y2i],
                "conf": float(box.conf.item()),
                "cls": cls_id,
                "name": class_name,
                "area": area,
            }
        )

    if not detections:
        return annotated, None

    # ========================================================
    # 2. Segmentacion SAM
    # ========================================================

    bboxes = [d["box_int"] for d in detections]

    masks = extract_sam_masks(
        frame_bgr=frame_bgr,
        sam_model=sam_model,
        bboxes=bboxes,
        yolo_device=yolo_device,
    )

    if not masks:
        return annotated, None

    n = min(len(detections), len(masks))
    detections = detections[:n]
    masks = masks[:n]

    # ========================================================
    # 3. Profundidad Depth Pro
    # ========================================================

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_input = transform(frame_rgb)

    with torch.inference_mode():
        prediction = depth_model.infer(depth_input, f_px=None)

    depth_map = prediction["depth"].detach().cpu().numpy().squeeze()
    depth_h, depth_w = depth_map.shape[:2]

    focal_px_depthpro = get_depthpro_focal_px(prediction)

    if focal_px_depthpro is None or focal_px_depthpro <= 0:
        focal_px_used = focal_px_lens
        focal_source = "SL-0041 HFOV"
    else:
        focal_px_used = focal_px_depthpro
        focal_source = "Depth Pro"

    # ========================================================
    # 4. Medicion por mascara
    # ========================================================

    detections_with_depth = []

    for detection, mask_orig in zip(detections, masks):
        mask_depth = resize_mask_to_shape(mask_orig, depth_h, depth_w)

        valid_depths = depth_map[
            mask_depth & np.isfinite(depth_map) & (depth_map > 0)
        ]

        if valid_depths.size == 0:
            continue

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
            continue

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

        mask_area_px = int(np.count_nonzero(mask_orig))

        detection["mask_orig"] = mask_orig
        detection["mask_depth"] = mask_depth
        detection["mask_area_px"] = mask_area_px

        detection["distance_p10"] = distance_p10
        detection["distance_p25"] = distance_p25
        detection["distance_median"] = distance_median
        detection["distance_p75"] = distance_p75
        detection["distance_p90"] = distance_p90
        detection["distance_mean"] = distance_mean
        detection["distance_min"] = distance_min
        detection["distance_max"] = distance_max

        detection["points_xyz"] = points_xyz
        detection["colors_rgb"] = colors_rgb

        detection.update(size_info_3d)
        detection.update(size_info_mask)
        detection.update(volume_info)

        passed, reason = passes_detection_filters(detection)
        detection["filter_passed"] = passed
        detection["filter_reason"] = reason

        depth_iqr = max(1e-6, distance_p75 - distance_p25)

        quality_score = (
            detection["conf"]
            * math.log1p(mask_area_px)
            / (1.0 + depth_iqr)
        )

        detection["depth_iqr"] = depth_iqr
        detection["quality_score"] = float(quality_score)

        detections_with_depth.append(detection)

    if not detections_with_depth:
        return annotated, None

    valid_candidates = [
        detection for detection in detections_with_depth
        if detection.get("filter_passed", False)
    ]

    if not valid_candidates:
        return annotated, None

    # ========================================================
    # 5. Tracking / seleccion de objetivo
    # ========================================================

    if ENABLE_TRACKING and tracker is not None:
        tracker.update(
            detections=valid_candidates,
            frame_idx=frame_idx,
        )

        target_track = tracker.get_track_by_id(target_track_id)

        if target_track is not None and target_track.is_active:
            selected_obj = target_track.last_detection
        else:
            selected_obj = min(
                valid_candidates,
                key=lambda detection: detection["distance_p25"],
            )
    else:
        selected_obj = min(
            valid_candidates,
            key=lambda detection: detection["distance_p25"],
        )

    # ========================================================
    # 6. Guardado opcional
    # ========================================================

    mask_path = None
    npy_path = None
    ply_path = None

    if save_artifacts:
        mask_path = OUTPUT_DIR / f"mask_{timestamp}.png"
        npy_path = OUTPUT_DIR / f"pointcloud_{timestamp}.npy"
        ply_path = OUTPUT_DIR / f"pointcloud_{timestamp}.ply"

        selected_mask_u8 = selected_obj["mask_orig"].astype(np.uint8) * 255
        cv2.imwrite(str(mask_path), selected_mask_u8)

        np.save(str(npy_path), selected_obj["points_xyz"])

        save_ply(
            path=ply_path,
            points_xyz=selected_obj["points_xyz"],
            colors_rgb=selected_obj["colors_rgb"],
        )

    # ========================================================
    # 7. Anotacion del frame procesado
    # ========================================================

    for detection in detections_with_depth:
        bx1, by1, bx2, by2 = detection["box_int"]
        is_selected = detection is selected_obj

        if not detection.get("filter_passed", False):
            color = (0, 0, 255)
            thickness = 1
            alpha = 0.10
        else:
            color = (0, 255, 0) if is_selected else (255, 0, 0)
            thickness = 3 if is_selected else 1
            alpha = 0.25 if is_selected else 0.15

        annotated = overlay_mask(
            annotated,
            detection["mask_orig"],
            color_bgr=color,
            alpha=alpha,
        )

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness)

        track_text = ""
        if detection.get("track_id") is not None:
            track_text = f"ID{detection['track_id']} "

        label = (
            f"{track_text}{detection['name']} "
            f"{detection['distance_median']:.2f}m "
            f"L={detection['mask_length_m']:.2f}m "
            f"H={detection['mask_height_m']:.2f}m"
        )

        cv2.putText(
            annotated,
            label,
            (bx1, max(30, by1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    result_info = {
        "mask_path": mask_path,
        "npy_path": npy_path,
        "ply_path": ply_path,
        "closest_obj": selected_obj,
        "track_id": selected_obj.get("track_id"),
        "focal_px_used": focal_px_used,
        "focal_source": focal_source,
        "annotated": annotated,
        "frame_timestamp": timestamp,
    }

    return annotated, result_info


# ============================================================
# Overlay live + estela
# ============================================================

def detection_center(detection: dict) -> tuple[int, int]:
    x1, y1, x2, y2 = detection["box_int"]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_trail(
    image_bgr: np.ndarray,
    trail_points: deque,
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    points = list(trail_points)

    if len(points) < 2:
        return

    for idx in range(1, len(points)):
        p1 = points[idx - 1]
        p2 = points[idx]

        thickness = max(1, int(1 + 5 * idx / len(points)))

        cv2.line(
            image_bgr,
            p1,
            p2,
            color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.circle(image_bgr, points[-1], 6, color, -1)


def draw_live_target_overlay(
    image_bgr: np.ndarray,
    result_info: dict | None,
    trail_points: deque,
) -> np.ndarray:
    display = image_bgr.copy()

    draw_trail(display, trail_points)

    if not result_info or result_info.get("closest_obj") is None:
        cv2.putText(
            display,
            "Sin objetivo seguido - presiona 'p'",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    obj = result_info["closest_obj"]
    x1, y1, x2, y2 = obj["box_int"]

    if "mask_orig" in obj:
        display = overlay_mask(
            display,
            obj["mask_orig"],
            color_bgr=(0, 255, 0),
            alpha=0.18,
        )

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    lines = [
        f"Siguiendo ID: {obj.get('track_id', 'N/A')}",
        f"Dist: {obj.get('distance_median', 0.0):.2f} m",
        f"L/H: {obj.get('mask_length_m', 0.0):.2f} x {obj.get('mask_height_m', 0.0):.2f} m",
        f"Vol: {obj.get('fish_volume_liters', 0.0):.2f} L",
    ]

    draw_multiline_text(
        display,
        lines,
        origin=(20, 70),
        color=(0, 255, 0),
        font_scale=0.75,
        thickness=2,
        line_height=28,
    )

    return display


# ============================================================
# Worker en segundo plano
# ============================================================

class LiveProcessingWorker:
    def __init__(
        self,
        yolo_model: YOLO,
        sam_model: SAM,
        depth_model,
        transform,
        yolo_device,
        frame_shape: tuple[int, int, int],
    ) -> None:
        self.yolo_model = yolo_model
        self.sam_model = sam_model
        self.depth_model = depth_model
        self.transform = transform
        self.yolo_device = yolo_device

        frame_h, frame_w = frame_shape[:2]

        self.tracker = FishTracker(
            frame_w=frame_w,
            frame_h=frame_h,
        )

        self.target_track_id = None
        self.frame_idx = 0

        self.input_queue: queue.Queue = queue.Queue(maxsize=1)
        self.output_queue: queue.Queue = queue.Queue(maxsize=1)

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def request_new_target(self) -> None:
        self.target_track_id = None
        self.tracker.reset()
        print("Se reinicio el tracker. El proximo resultado valido sera el nuevo objetivo.")

    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        item = {
            "frame": frame_bgr.copy(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        }

        try:
            if self.input_queue.full():
                _ = self.input_queue.get_nowait()

            self.input_queue.put_nowait(item)
        except queue.Full:
            pass

    def get_latest_result(self) -> dict | None:
        latest = None

        while not self.output_queue.empty():
            latest = self.output_queue.get_nowait()

        return latest

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame = item["frame"]
            timestamp = item["timestamp"]

            annotated, result_info = process_frame(
                frame_bgr=frame,
                yolo_model=self.yolo_model,
                sam_model=self.sam_model,
                depth_model=self.depth_model,
                transform=self.transform,
                yolo_device=self.yolo_device,
                timestamp=timestamp,
                save_artifacts=False,
                tracker=self.tracker,
                frame_idx=self.frame_idx,
                target_track_id=self.target_track_id,
            )

            self.frame_idx += 1

            if result_info is not None:
                if self.target_track_id is None:
                    self.target_track_id = result_info.get("track_id")
                    print(f"Nuevo objetivo seleccionado: ID {self.target_track_id}")

                result_info["annotated"] = annotated
                result_info["target_track_id"] = self.target_track_id

                try:
                    if self.output_queue.full():
                        _ = self.output_queue.get_nowait()

                    self.output_queue.put_nowait(result_info)
                except queue.Full:
                    pass


def update_3d_viewer(viewer: PointCloudViewer, result_info: dict | None) -> None:
    if not result_info:
        return

    obj = result_info.get("closest_obj")

    if obj is None:
        return

    points_xyz = obj.get("points_xyz")

    if points_xyz is None or points_xyz.size == 0:
        return

    colors_rgb = depth_values_to_rgb(points_xyz[:, 2])

    viewer.update(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        title="Nube de puntos 3D - objeto seguido",
    )


# ============================================================
# Programa principal
# ============================================================

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo_device = 0 if device.type == "cuda" else "cpu"
    precision = torch.half if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")

    yolo_model, sam_model, depth_model, transform = load_models(
        device=device,
        precision=precision,
    )

    viewer = PointCloudViewer(enabled=ENABLE_3D_VIEWER)

    # ========================================================
    # Modo imagen local
    # ========================================================

    if not USE_RTSP:
        frame = cv2.imread(TEST_IMAGE_PATH)

        if frame is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {TEST_IMAGE_PATH}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        annotated, result_info = process_frame(
            frame_bgr=frame,
            yolo_model=yolo_model,
            sam_model=sam_model,
            depth_model=depth_model,
            transform=transform,
            yolo_device=yolo_device,
            timestamp=timestamp,
            save_artifacts=True,
            tracker=None,
            frame_idx=0,
            target_track_id=None,
        )

        result_path = OUTPUT_DIR / f"resultado_test_{timestamp}.jpg"
        cv2.imwrite(str(result_path), annotated)

        print(f"Resultado guardado en: {result_path}")
        update_3d_viewer(viewer, result_info)

        if SHOW_WINDOWS:
            cv2.imshow("Resultado YOLO-World + SAM + Depth Pro", annotated)
            print("Ventanas abiertas. Presiona 'q' o ESC para cerrar.")

            while True:
                viewer.poll()
                key = cv2.waitKey(30) & 0xFF

                if key in (ord("q"), 27):
                    break

        viewer.close()
        cv2.destroyAllWindows()
        return

    # ========================================================
    # Modo RTSP live tracking
    # ========================================================

    print(f"Conectando a: {RTSP_URL}")

    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el stream RTSP")

    ret, first_frame = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError("No se pudo leer el primer frame del stream RTSP")

    worker = LiveProcessingWorker(
        yolo_model=yolo_model,
        sam_model=sam_model,
        depth_model=depth_model,
        transform=transform,
        yolo_device=yolo_device,
        frame_shape=first_frame.shape,
    )

    worker.start()

    latest_result_info = None
    trail_points = deque(maxlen=TRAIL_MAX_POINTS)

    frame_counter = 0
    paused_processing = False

    print()
    print("Controles:")
    print("  p = seleccionar/reenfocar objetivo")
    print("  r = resetear estela")
    print("  s = pausar/reanudar procesamiento")
    print("  q = salir")
    print()
    print("Configuracion live:")
    print(f"  LIVE_PROCESS_EVERY_N_FRAMES = {LIVE_PROCESS_EVERY_N_FRAMES}")
    print(f"  TRAIL_MAX_POINTS = {TRAIL_MAX_POINTS}")
    print()

    try:
        while True:
            viewer.poll()

            ret, frame = cap.read()

            if not ret:
                print("No se pudo leer frame del stream")
                break

            frame_counter += 1

            if (
                BACKGROUND_PROCESSING
                and not paused_processing
                and frame_counter % LIVE_PROCESS_EVERY_N_FRAMES == 0
            ):
                worker.submit_frame(frame)

            new_result_info = worker.get_latest_result()

            if new_result_info is not None:
                latest_result_info = new_result_info

                obj = latest_result_info.get("closest_obj")

                if obj is not None:
                    trail_points.append(detection_center(obj))

                update_3d_viewer(viewer, latest_result_info)

            display_frame = draw_live_target_overlay(
                image_bgr=frame,
                result_info=latest_result_info,
                trail_points=trail_points,
            )

            cv2.putText(
                display_frame,
                "p: objetivo | r: reset estela | s: pausa proc | q: salir",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if paused_processing:
                cv2.putText(
                    display_frame,
                    "PROCESAMIENTO PAUSADO",
                    (20, display_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Stream RTSP - tracking en vivo", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                trail_points.clear()
                print("Estela reiniciada.")

            if key == ord("s"):
                paused_processing = not paused_processing
                print(f"Procesamiento pausado: {paused_processing}")

            if key == ord("p"):
                print("Seleccionando nuevo objetivo...")
                worker.request_new_target()
                trail_points.clear()
                latest_result_info = None
                worker.submit_frame(frame)

    finally:
        worker.stop()
        cap.release()
        viewer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()