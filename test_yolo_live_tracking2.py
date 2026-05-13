"""
Versión enfocada en:
- Leer un stream RTSP o cámara local.
- Detectar objetos con YOLO / YOLO-World.
- Crear tracks únicamente para objetos detectados en la zona de entrada izquierda.
- Mantener múltiples tracks activos mientras los objetos avanzan hacia la derecha.
- Dibujar una caja y una estela por cada objeto seguido.

No incluye:
- Click sobre detecciones.
- Segmentación, profundidad, nubes de puntos o guardado de artefactos.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import os
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# Configuración stream
# ============================================================

USE_RTSP = os.getenv("USE_RTSP", "1") == "1"

RTSP_USER = os.getenv("RTSP_USER", "admin")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", "itg24chile")
RTSP_IP = os.getenv("RTSP_IP", "10.22.100.22")
RTSP_PORT = int(os.getenv("RTSP_PORT", "554"))
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "102")

RTSP_URL = (
    f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}"
    f"@{RTSP_IP}:{RTSP_PORT}/Streaming/Channels/{RTSP_CHANNEL}"
)

VIDEO_SOURCE = RTSP_URL if USE_RTSP else int(os.getenv("CAMERA_INDEX", "0"))


# ============================================================
# Configuración YOLO
# ============================================================

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s-worldv2.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
TARGET_CONFIDENCE_THRESHOLD = float(os.getenv("TARGET_CONFIDENCE_THRESHOLD", "0.45"))

TARGET_CLASSES = [
    "fish",
    "salmon fish",
    "trout fish",
]

TARGET_CLASS_SET = set(TARGET_CLASSES)


# ============================================================
# Zona de entrada
# ============================================================

# Solo se crean tracks nuevos si la detección aparece dentro de esta zona.
# ENTRY_SIDE puede ser "left" o "right".
# ENTRY_SIDE = os.getenv("ENTRY_SIDE", "left").strip().lower()
ENTRY_SIDE = os.getenv("ENTRY_SIDE", "right").strip().lower()

ENTRY_MAX_X_RATIO = float(os.getenv("ENTRY_MAX_X_RATIO", "0.45"))
ENTRY_MIN_Y_RATIO = float(os.getenv("ENTRY_MIN_Y_RATIO", "0.10"))
ENTRY_MAX_Y_RATIO = float(os.getenv("ENTRY_MAX_Y_RATIO", "0.90"))


# ============================================================
# Configuración tracking
# ============================================================

PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "5"))
DISPLAY_FPS = float(os.getenv("DISPLAY_FPS", "15"))

TRACKER_IOU_WEIGHT = float(os.getenv("TRACKER_IOU_WEIGHT", "0.70"))
TRACKER_CENTER_WEIGHT = float(os.getenv("TRACKER_CENTER_WEIGHT", "0.30"))
TRACKER_SIZE_WEIGHT = float(os.getenv("TRACKER_SIZE_WEIGHT", "0.20"))
TRACKER_DIRECTION_WEIGHT = float(os.getenv("TRACKER_DIRECTION_WEIGHT", "0.35"))
TRACKER_MIN_SCORE = float(os.getenv("TRACKER_MIN_SCORE", "0.15"))
TRACKER_MAX_MISSING = int(os.getenv("TRACKER_MAX_MISSING", "10"))
TRAIL_MAX_POINTS = int(os.getenv("TRAIL_MAX_POINTS", "80"))
TRACKER_MAX_CENTER_DIST_RATIO = float(os.getenv("TRACKER_MAX_CENTER_DIST_RATIO", "0.18"))
TRACKER_MIN_SIZE_SIMILARITY = float(os.getenv("TRACKER_MIN_SIZE_SIMILARITY", "0.35"))
TRACKER_MAX_BACKTRACK_X_RATIO = float(os.getenv("TRACKER_MAX_BACKTRACK_X_RATIO", "0.04"))

SHOW_WINDOWS = os.getenv("SHOW_WINDOWS", "1") == "1"
WINDOW_NAME = "YOLO - multi tracking desde zona de entrada"


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
    trail: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=TRAIL_MAX_POINTS))

    def __post_init__(self) -> None:
        self.detection.track_id = self.track_id
        self.trail.append(bbox_center(self.detection.box))

    def update(self, detection: Detection) -> None:
        previous_center = bbox_center(self.detection.box)
        new_center = bbox_center(detection.box)
        delta_x = float(new_center[0] - previous_center[0])
        delta_y = float(new_center[1] - previous_center[1])

        # Suaviza la velocidad para no reaccionar demasiado a una sola detección ruidosa.
        self.velocity = (
            0.65 * delta_x + 0.35 * self.velocity[0],
            0.65 * delta_y + 0.35 * self.velocity[1],
        )
        detection.track_id = self.track_id
        self.detection = detection
        self.missing = 0
        self.hits += 1
        self.trail.append(bbox_center(detection.box))

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


# ============================================================
# Utilidades
# ============================================================

def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


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


def center_distance(
    box_a: list[int],
    box_b: list[int],
) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def bbox_size_similarity(box_a: list[int], box_b: list[int]) -> float:
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    return float(min(area_a, area_b) / max(area_a, area_b))


def direction_similarity(
    previous_box: list[int],
    current_box: list[int],
    frame_w: int,
) -> float:
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


def is_in_entry_zone(
    detection: Detection,
    frame_w: int,
    frame_h: int,
) -> bool:
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


# ============================================================
# YOLO
# ============================================================

def load_yolo_model(device: torch.device) -> tuple[YOLO, int | str]:
    yolo_device = 0 if device.type == "cuda" else "cpu"

    print(f"Cargando YOLO: {YOLO_WEIGHTS}")
    model = YOLO(YOLO_WEIGHTS)

    if TARGET_CLASSES and "world" in YOLO_WEIGHTS.lower():
        model.set_classes(TARGET_CLASSES)
        print(f"Clases YOLO-World configuradas: {TARGET_CLASSES}")

    return model, yolo_device


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


# ============================================================
# Multi tracker desde zona de entrada
# ============================================================

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

        # Asociar detecciones nuevas a tracks existentes.
        for score, track_idx, det_idx in candidates:
            if track_idx not in unmatched_tracks:
                continue

            if det_idx not in unmatched_detections:
                continue

            self.tracks[track_idx].update(detections[det_idx])
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)

        # Marcar tracks que no encontraron detección en este ciclo.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missing()

        # Crear tracks nuevos solo desde la zona de entrada configurada.
        for det_idx in unmatched_detections:
            detection = detections[det_idx]

            if detection.confidence < TARGET_CONFIDENCE_THRESHOLD:
                continue

            if not is_in_entry_zone(detection, frame_w, frame_h):
                continue

            track = Track(
                track_id=self.next_track_id,
                detection=detection,
            )
            self.next_track_id += 1
            self.tracks.append(track)

            print(
                f"Nuevo track desde zona {ENTRY_SIDE}: "
                f"ID {track.track_id} | "
                f"{detection.class_name} | "
                f"conf={detection.confidence:.2f}"
            )

        self.tracks = [track for track in self.tracks if track.is_alive]
        return self.tracks


# ============================================================
# Visualización
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

    cv2.rectangle(
        frame_bgr,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

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


def draw_track(frame_bgr: np.ndarray, track: Track) -> None:
    color = track_color(track.track_id)
    detection = track.detection
    x1, y1, x2, y2 = detection.box

    draw_trail(frame_bgr, track.trail, color)

    thickness = 3 if track.is_active else 1
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

    label = (
        f"ID {track.track_id} | "
        f"{detection.class_name} | "
        f"conf={detection.confidence:.2f} | "
        f"miss={track.missing}"
    )

    cv2.putText(
        frame_bgr,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_overlay(frame_bgr: np.ndarray, tracks: list[Track]) -> None:
    draw_entry_zone(frame_bgr)

    for track in tracks:
        draw_track(frame_bgr, track)

    cv2.putText(
        frame_bgr,
        f"tracks activos: {sum(1 for track in tracks if track.is_active)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_bgr,
        "q: salir | r: reset tracks",
        (20, frame_bgr.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# Programa principal
# ============================================================

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")

    yolo_model, yolo_device = load_yolo_model(device)

    print(f"Abriendo fuente de video: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de video")

    tracker = EntryZoneMultiTracker()
    tracks: list[Track] = []
    frame_counter = 0
    display_delay_ms = int(1000 / max(DISPLAY_FPS, 1.0))

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("No se pudo leer frame del stream")
                break

            frame_counter += 1
            frame_h, frame_w = frame.shape[:2]

            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                detections = detect_objects(frame, yolo_model, yolo_device)
                tracks = tracker.update(detections, frame_w, frame_h)

            display = frame.copy()
            draw_overlay(display, tracks)

            if SHOW_WINDOWS:
                cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(display_delay_ms) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                tracker.reset()
                tracks = []
                print("Tracks reiniciados manualmente.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
