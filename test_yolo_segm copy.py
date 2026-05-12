from datetime import datetime
from pathlib import Path
import math
import os

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

# Si USE_RTSP=False, procesa una imagen local y termina.
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

# Puedes probar:
#   SAM_WEIGHTS = "mobile_sam.pt"
#   SAM_WEIGHTS = "sam_b.pt"
#   SAM_WEIGHTS = "sam2.1_b.pt"
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
# Configuracion multi-frame
# ============================================================

SEQUENCE_NUM_FRAMES = int(os.getenv("SEQUENCE_NUM_FRAMES", "120"))

# Captura 1 de cada N frames. Si es 1, usa frames consecutivos.
# Si es 3, captura un frame, salta 2, captura otro.
SEQUENCE_FRAME_STRIDE = int(os.getenv("SEQUENCE_FRAME_STRIDE", "2"))

# Si True, guarda mascara/PLY de cada frame procesado.
# Si False, solo guarda artefactos del mejor frame/final.
SAVE_SEQUENCE_FRAME_ARTIFACTS = os.getenv("SAVE_SEQUENCE_FRAME_ARTIFACTS", "0") == "1"

# Si True, al presionar p se procesa una secuencia.
# Si False, se procesa solo el frame actual.
USE_MULTIFRAME_ON_KEYPRESS = os.getenv("USE_MULTIFRAME_ON_KEYPRESS", "1") == "1"


# ============================================================
# Filtros de calidad para detecciones / mascaras
# ============================================================

MIN_MASK_AREA_PX = int(os.getenv("MIN_MASK_AREA_PX", "1500"))
MIN_DISTANCE_M = float(os.getenv("MIN_DISTANCE_M", "0.20"))
MAX_DISTANCE_M = float(os.getenv("MAX_DISTANCE_M", "4.00"))

MIN_FISH_LENGTH_M = float(os.getenv("MIN_FISH_LENGTH_M", "0.05"))
MAX_FISH_LENGTH_M = float(os.getenv("MAX_FISH_LENGTH_M", "1.50"))

# Modelo empirico simple para volumen de pez:
# thickness ≈ FISH_THICKNESS_RATIO * height
# volume ≈ FISH_VOLUME_SHAPE_FACTOR * length * height * thickness
FISH_THICKNESS_RATIO = float(os.getenv("FISH_THICKNESS_RATIO", "0.45"))
FISH_VOLUME_SHAPE_FACTOR = float(os.getenv("FISH_VOLUME_SHAPE_FACTOR", "0.55"))


# ============================================================
# Funciones auxiliares
# ============================================================

def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def focal_px_from_hfov(image_width_px: int, hfov_deg: float) -> float:
    """Calcula focal en pixeles desde HFOV para una camara pinhole ideal."""
    hfov_rad = math.radians(hfov_deg)
    return image_width_px / (2.0 * math.tan(hfov_rad / 2.0))


def get_depthpro_focal_px(prediction: dict) -> float | None:
    """Extrae la longitud focal estimada por Depth Pro."""
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
    """Superpone una mascara sobre una imagen BGR."""
    overlay = image_bgr.copy()
    overlay[mask_bool] = color_bgr
    return cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)


def resize_mask_to_shape(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Redimensiona mascara booleana usando nearest-neighbor."""
    mask_u8 = mask_bool.astype(np.uint8) * 255
    resized = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def estimate_size_from_point_cloud(points_xyz: np.ndarray) -> dict:
    """
    Estima dimensiones visibles y volumen aproximado desde la nube de puntos.

    x_size: extension horizontal visible
    y_size: extension vertical visible
    z_size: variacion de profundidad visible

    Ojo:
    La camara solo ve una superficie del objeto. El volumen no es un volumen real
    cerrado; es una aproximacion geometrica desde la nube visible.
    """

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
    """
    Estima largo y alto visibles usando PCA 2D sobre la mascara.

    Ventaja:
    - Si el pez esta inclinado en la imagen, el largo no depende solo del eje X.
    - Calcula dimensiones orientadas segun la forma del pez.
    """

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
    """
    Estima volumen aproximado del pez usando un modelo empirico simple.

    thickness_m ≈ thickness_ratio * height_m
    volume_m3 ≈ shape_factor * length_m * height_m * thickness_m

    Esto NO es volumen real medido; es una aproximacion inicial.
    """

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
    """
    Aplica filtros basicos para evitar falsos positivos o mediciones poco razonables.
    """

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
    """
    Convierte pixeles de una mascara + mapa de profundidad en nube de puntos 3D.

    Modelo pinhole:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    Se asume fx = fy = focal_px y centro optico en el centro de la imagen.
    """

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
    """Convierte profundidades en colores RGB tipo turbo, cercano=calido, lejano=frio."""

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
    """
    Convierte todo el mapa de profundidad en una nube de puntos 3D.

    Si se entrega una mascara, los puntos de esa mascara se resaltan en verde.
    """

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
    fx = focal_px
    fy = focal_px

    x = ((xs.astype(np.float32) - cx) * z) / fx
    y = ((ys.astype(np.float32) - cy) * z) / fy

    points_xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    colors_rgb = depth_values_to_rgb(z)

    if highlight_mask_depth is not None:
        mask_hit = highlight_mask_depth[ys, xs]
        colors_rgb[mask_hit] = np.array([0, 255, 40], dtype=np.uint8)

    return points_xyz, colors_rgb


def save_ply(
    path: Path,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
) -> None:
    """
    Guarda una nube de puntos en formato PLY ASCII.
    Compatible con CloudCompare, MeshLab, Open3D, etc.
    """

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


class PointCloudViewer:
    """Visor Open3D no bloqueante para la profundidad del objeto mas cercano."""

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
        title: str = "Nube de puntos 3D - objeto mas cercano",
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


def extract_sam_masks(
    frame_bgr: np.ndarray,
    sam_model: SAM,
    bboxes: list[list[int]],
    yolo_device,
) -> list[np.ndarray]:
    """
    Ejecuta SAM/MobileSAM usando bounding boxes como prompts.

    Retorna una lista de mascaras booleanas en coordenadas de la imagen original.
    """

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
# Procesamiento de un frame
# ============================================================

def process_frame(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    sam_model: SAM,
    depth_model,
    transform,
    yolo_device,
    timestamp: str,
    save_artifacts: bool = True,
) -> tuple[np.ndarray, dict | None]:
    """
    Procesa un frame:
    - Detecta peces con YOLO-World.
    - Segmenta cada deteccion usando SAM/MobileSAM.
    - Calcula profundidad con Depth Pro.
    - Calcula dimensiones visibles por mascara y nube 3D.
    - Filtra candidatos poco razonables.
    - Selecciona el objeto segmentado mas cercano.
    - Genera nube de puntos para el objeto mas cercano.
    """

    orig_h, orig_w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    focal_px_lens = focal_px_from_hfov(orig_w, LENS_HFOV_DEG)

    # ========================================================
    # 1. Deteccion con YOLO-World
    # ========================================================

    yolo_results = yolo_model.predict(
        source=frame_bgr,
        conf=CONFIDENCE_THRESHOLD,
        device=yolo_device,
        verbose=False,
    )

    boxes = yolo_results[0].boxes if yolo_results else None

    if boxes is None or len(boxes) == 0:
        cv2.putText(
            annotated,
            "Sin detecciones YOLO-World",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print("No se detectaron objetos.")
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
        cv2.putText(
            annotated,
            f"No se detecto {TARGET_LABEL}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print(f"No se detecto ningun objeto de estas clases: {TARGET_CLASSES}")
        return annotated, None

    # ========================================================
    # 2. Segmentacion con SAM usando los boxes detectados
    # ========================================================

    bboxes = [d["box_int"] for d in detections]

    masks = extract_sam_masks(
        frame_bgr=frame_bgr,
        sam_model=sam_model,
        bboxes=bboxes,
        yolo_device=yolo_device,
    )

    if not masks:
        cv2.putText(
            annotated,
            "SAM no genero mascaras",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print("SAM no genero mascaras.")
        return annotated, None

    n = min(len(detections), len(masks))
    detections = detections[:n]
    masks = masks[:n]

    # ========================================================
    # 3. Profundidad con Depth Pro
    # ========================================================

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_input = transform(frame_rgb)

    with torch.inference_mode():
        prediction = depth_model.infer(depth_input, f_px=None)

    depth_map = prediction["depth"].detach().cpu().numpy().squeeze()
    depth_h, depth_w = depth_map.shape[:2]

    focal_px_depthpro = get_depthpro_focal_px(prediction)

    if focal_px_depthpro is None or focal_px_depthpro <= 0:
        print("Depth Pro no entrego focal valida. Se usara focal por HFOV como respaldo.")
        focal_px_used = focal_px_lens
        focal_source = "SL-0041 HFOV"
    else:
        focal_px_used = focal_px_depthpro
        focal_source = "Depth Pro"

    print()
    print("========== Frame procesado ==========")
    print(f"Frame size:              {orig_w}x{orig_h}")
    print(f"Depth map size:          {depth_w}x{depth_h}")
    print(f"Depth min/max:           {np.nanmin(depth_map):.3f} / {np.nanmax(depth_map):.3f} m")
    print(f"HFOV lente SL-0041:      {LENS_HFOV_DEG:.1f} grados")
    print(f"Focal px lente SL-0041:  {focal_px_lens:.2f} px")

    if focal_px_depthpro is not None:
        print(f"Focal px Depth Pro:      {focal_px_depthpro:.2f} px")
    else:
        print("Focal px Depth Pro:      No disponible")

    print(f"Focal px usada:          {focal_px_used:.2f} px ({focal_source})")

    # ========================================================
    # 4. Profundidad, tamano y filtros por mascara
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
        cv2.putText(
            annotated,
            f"{TARGET_LABEL.capitalize()} sin profundidad valida",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print(f"No hay profundidad valida en las mascaras de {TARGET_LABEL}.")
        return annotated, None

    valid_candidates = [
        detection for detection in detections_with_depth
        if detection.get("filter_passed", False)
    ]

    if not valid_candidates:
        cv2.putText(
            annotated,
            "Sin candidatos validos tras filtros",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        print("No hay candidatos validos tras filtros.")
        print("Detecciones descartadas:")

        for detection in detections_with_depth:
            print(
                f"  {detection['name']} "
                f"dist={detection['distance_median']:.2f}m "
                f"len={detection.get('mask_length_m', 0.0):.2f}m "
                f"area={detection.get('mask_area_px', 0)} "
                f"reason={detection.get('filter_reason', '')}"
            )

        return annotated, None

    closest_obj = min(
        valid_candidates,
        key=lambda detection: detection["distance_p25"],
    )

    # ========================================================
    # 5. Guardar mascara y nube del objeto mas cercano
    # ========================================================

    mask_path = OUTPUT_DIR / f"mask_{timestamp}.png"
    npy_path = OUTPUT_DIR / f"pointcloud_{timestamp}.npy"
    ply_path = OUTPUT_DIR / f"pointcloud_{timestamp}.ply"
    scene_ply_path = None

    if save_artifacts:
        closest_mask_u8 = closest_obj["mask_orig"].astype(np.uint8) * 255
        cv2.imwrite(str(mask_path), closest_mask_u8)

        np.save(str(npy_path), closest_obj["points_xyz"])

        save_ply(
            path=ply_path,
            points_xyz=closest_obj["points_xyz"],
            colors_rgb=closest_obj["colors_rgb"],
        )
    else:
        mask_path = None
        npy_path = None
        ply_path = None

    if save_artifacts and SAVE_FULL_DEPTH_POINT_CLOUD:
        scene_points_xyz, scene_colors_rgb = create_point_cloud_from_depth_map(
            depth_map=depth_map,
            frame_bgr=frame_bgr,
            focal_px=focal_px_used,
            highlight_mask_depth=closest_obj["mask_depth"],
            max_points=MAX_DEPTH_VIEWER_POINTS,
        )

        scene_ply_path = OUTPUT_DIR / f"depth_map_pointcloud_{timestamp}.ply"
        save_ply(
            path=scene_ply_path,
            points_xyz=scene_points_xyz,
            colors_rgb=scene_colors_rgb,
        )

    # ========================================================
    # 6. Imprimir resultados
    # ========================================================

    x1, y1, x2, y2 = closest_obj["box_int"]

    print()
    print(f"{TARGET_LABEL.capitalize()} mas cercano segmentado:")
    print(f"Clase:       {closest_obj['name']}")
    print(f"Confianza:   {closest_obj['conf']:.2f}")
    print(f"Box:         {x1}, {y1}, {x2}, {y2}")
    print(f"Filtro:      {closest_obj['filter_reason']}")
    print(f"Score:       {closest_obj['quality_score']:.3f}")

    print()
    print("Distancia estimada por mascara:")
    print(f"P10:         {closest_obj['distance_p10']:.3f} m")
    print(f"P25:         {closest_obj['distance_p25']:.3f} m")
    print(f"Mediana:     {closest_obj['distance_median']:.3f} m")
    print(f"P75:         {closest_obj['distance_p75']:.3f} m")
    print(f"P90:         {closest_obj['distance_p90']:.3f} m")
    print(f"Media:       {closest_obj['distance_mean']:.3f} m")
    print(f"Minima:      {closest_obj['distance_min']:.3f} m")
    print(f"Maxima:      {closest_obj['distance_max']:.3f} m")

    print()
    print("Tamano visible estimado:")
    print(f"Mask PCA largo:      {closest_obj['mask_length_m']:.3f} m")
    print(f"Mask PCA alto:       {closest_obj['mask_height_m']:.3f} m")
    print(f"Mask area:           {closest_obj['mask_area_px']} px")

    print()
    print("Tamano desde nube 3D:")
    print(f"Ancho X:             {closest_obj['x_size_m']:.3f} m")
    print(f"Alto Y:              {closest_obj['y_size_m']:.3f} m")
    print(f"Rango Z visible:     {closest_obj['z_size_m']:.3f} m")
    print(
        "PCA L/W/T 3D:        "
        f"{closest_obj['pca_length_m']:.3f} / "
        f"{closest_obj['pca_width_m']:.3f} / "
        f"{closest_obj['pca_thickness_m']:.3f} m"
    )

    print()
    print("Volumenes aproximados:")
    print(f"BBox Vol 3D:          {closest_obj['bbox_volume_m3']:.6f} m3")
    print(f"PCA BBox Vol 3D:      {closest_obj['pca_bbox_volume_m3']:.6f} m3")
    print(f"Elipsoide 3D visible: {closest_obj['ellipsoid_volume_m3']:.6f} m3")
    print(f"Espesor empirico:     {closest_obj['fish_thickness_m']:.3f} m")
    print(f"Vol pez empirico:     {closest_obj['fish_volume_m3']:.6f} m3")
    print(f"Vol pez litros:       {closest_obj['fish_volume_liters']:.2f} L")
    print(f"Puntos:               {len(closest_obj['points_xyz'])}")

    print()

    if save_artifacts:
        print(f"Mascara guardada en:       {mask_path}")
        print(f"Nube NPY guardada en:      {npy_path}")
        print(f"Nube PLY guardada en:      {ply_path}")

        if scene_ply_path is not None:
            print(f"Depth PLY guardada en:     {scene_ply_path}")
    else:
        print("Artefactos individuales no guardados para este frame.")

    # ========================================================
    # 7. Dibujar resultados
    # ========================================================

    for detection in detections_with_depth:
        bx1, by1, bx2, by2 = detection["box_int"]
        is_closest = detection is closest_obj

        if not detection.get("filter_passed", False):
            color = (0, 0, 255)
            thickness = 1
            alpha = 0.10
        else:
            color = (0, 255, 0) if is_closest else (255, 0, 0)
            thickness = 3 if is_closest else 1
            alpha = 0.25 if is_closest else 0.15

        annotated = overlay_mask(
            annotated,
            detection["mask_orig"],
            color_bgr=color,
            alpha=alpha,
        )

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness)

        if detection.get("filter_passed", False):
            label = (
                f"{detection['name']} "
                f"{detection['distance_median']:.2f}m "
                f"L={detection['mask_length_m']:.2f}m "
                f"H={detection['mask_height_m']:.2f}m"
            )
        else:
            label = (
                f"{detection['name']} desc "
                f"{detection.get('filter_reason', '')[:18]}"
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

    summary_lines = [
        f"{TARGET_LABEL.capitalize()} mas cercano segmentado",
        f"Dist p25/med: {closest_obj['distance_p25']:.2f}/{closest_obj['distance_median']:.2f} m",
        f"L/H mask: {closest_obj['mask_length_m']:.2f} x {closest_obj['mask_height_m']:.2f} m",
        f"Esp est: {closest_obj['fish_thickness_m']:.2f} m",
        f"Vol emp: {closest_obj['fish_volume_liters']:.2f} L",
        f"Pts: {len(closest_obj['points_xyz'])}",
        f"fx: {focal_px_used:.1f}px ({focal_source})",
    ]

    draw_multiline_text(
        annotated,
        summary_lines,
        origin=(20, 40),
        color=(0, 255, 0),
        font_scale=0.75,
        thickness=2,
        line_height=30,
    )

    debug_lines = [
        f"YOLO-World + {SAM_WEIGHTS}",
        f"HFOV SL-0041: {LENS_HFOV_DEG:.1f} deg",
        f"fx lente: {focal_px_lens:.1f}px",
        f"Filtros: area>{MIN_MASK_AREA_PX}, dist {MIN_DISTANCE_M}-{MAX_DISTANCE_M}m",
        f"Modelo vol: T={FISH_THICKNESS_RATIO:.2f}H, k={FISH_VOLUME_SHAPE_FACTOR:.2f}",
    ]

    if focal_px_depthpro is not None:
        debug_lines.append(f"fx DepthPro: {focal_px_depthpro:.1f}px")

    draw_multiline_text(
        annotated,
        debug_lines,
        origin=(20, 270),
        color=(0, 255, 255),
        font_scale=0.60,
        thickness=2,
        line_height=24,
    )

    result_info = {
        "mask_path": mask_path,
        "npy_path": npy_path,
        "ply_path": ply_path,
        "scene_ply_path": scene_ply_path,
        "closest_obj": closest_obj,
        "focal_px_used": focal_px_used,
        "focal_source": focal_source,
    }

    return annotated, result_info


# ============================================================
# Procesamiento multi-frame
# ============================================================

def robust_stats(values: list[float]) -> dict:
    """
    Calcula estadisticas robustas para una lista de valores.
    """

    arr = np.array(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "median": 0.0,
            "mean": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "iqr": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))

    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def collect_sequence_summary(results: list[dict]) -> dict | None:
    """
    Resume los resultados validos de una secuencia multi-frame.
    """

    valid_results = [
        result for result in results
        if result is not None and result.get("closest_obj") is not None
    ]

    if not valid_results:
        return None

    objs = [result["closest_obj"] for result in valid_results]

    summary = {
        "n_valid": len(valid_results),
        "distance_m": robust_stats([obj["distance_median"] for obj in objs]),
        "distance_p25_m": robust_stats([obj["distance_p25"] for obj in objs]),
        "length_m": robust_stats([obj["mask_length_m"] for obj in objs]),
        "height_m": robust_stats([obj["mask_height_m"] for obj in objs]),
        "thickness_m": robust_stats([obj["fish_thickness_m"] for obj in objs]),
        "volume_liters": robust_stats([obj["fish_volume_liters"] for obj in objs]),
        "quality_score": robust_stats([obj.get("quality_score", 0.0) for obj in objs]),
    }

    best_result = max(
        valid_results,
        key=lambda result: result["closest_obj"].get("quality_score", 0.0),
    )

    summary["best_result"] = best_result
    summary["best_obj"] = best_result["closest_obj"]

    return summary


def print_sequence_summary(summary: dict) -> None:
    """
    Imprime resumen agregado multi-frame.
    """

    print()
    print("========== Resumen multi-frame ==========")
    print(f"Frames validos: {summary['n_valid']}")

    print()
    print("Distancia:")
    print(
        f"Mediana: {summary['distance_m']['median']:.3f} m "
        f"IQR: {summary['distance_m']['iqr']:.3f} m "
        f"Rango: {summary['distance_m']['min']:.3f}-{summary['distance_m']['max']:.3f} m"
    )

    print()
    print("Tamano visible por mascara:")
    print(
        f"Largo: {summary['length_m']['median']:.3f} m "
        f"IQR: {summary['length_m']['iqr']:.3f} m"
    )
    print(
        f"Alto:  {summary['height_m']['median']:.3f} m "
        f"IQR: {summary['height_m']['iqr']:.3f} m"
    )

    print()
    print("Volumen empirico:")
    print(
        f"Volumen: {summary['volume_liters']['median']:.2f} L "
        f"IQR: {summary['volume_liters']['iqr']:.2f} L "
        f"Rango: {summary['volume_liters']['min']:.2f}-{summary['volume_liters']['max']:.2f} L"
    )

    print()
    print("Mejor frame:")
    best = summary["best_obj"]
    print(f"Dist: {best['distance_median']:.3f} m")
    print(f"Largo: {best['mask_length_m']:.3f} m")
    print(f"Alto: {best['mask_height_m']:.3f} m")
    print(f"Vol: {best['fish_volume_liters']:.2f} L")
    print(f"Score: {best.get('quality_score', 0.0):.3f}")


def draw_sequence_summary(
    image_bgr: np.ndarray,
    summary: dict,
    origin: tuple[int, int] = (20, 40),
) -> np.ndarray:
    """
    Dibuja resumen multi-frame sobre la imagen del mejor frame.
    """

    annotated = image_bgr.copy()

    lines = [
        "Resumen multi-frame",
        f"Frames validos: {summary['n_valid']}",
        f"Dist: {summary['distance_m']['median']:.2f} m IQR {summary['distance_m']['iqr']:.2f}",
        f"Largo: {summary['length_m']['median']:.2f} m IQR {summary['length_m']['iqr']:.2f}",
        f"Alto: {summary['height_m']['median']:.2f} m IQR {summary['height_m']['iqr']:.2f}",
        f"Esp: {summary['thickness_m']['median']:.2f} m",
        f"Vol: {summary['volume_liters']['median']:.2f} L IQR {summary['volume_liters']['iqr']:.2f}",
    ]

    draw_multiline_text(
        annotated,
        lines,
        origin=origin,
        color=(0, 255, 255),
        font_scale=0.75,
        thickness=2,
        line_height=30,
    )

    return annotated


def save_best_result_artifacts(
    summary: dict,
    timestamp: str,
) -> dict:
    """
    Guarda mascara y nube de puntos del mejor frame de la secuencia.
    """

    best_obj = summary["best_obj"]

    mask_path = OUTPUT_DIR / f"sequence_best_mask_{timestamp}.png"
    npy_path = OUTPUT_DIR / f"sequence_best_pointcloud_{timestamp}.npy"
    ply_path = OUTPUT_DIR / f"sequence_best_pointcloud_{timestamp}.ply"

    mask_u8 = best_obj["mask_orig"].astype(np.uint8) * 255
    cv2.imwrite(str(mask_path), mask_u8)

    np.save(str(npy_path), best_obj["points_xyz"])

    save_ply(
        path=ply_path,
        points_xyz=best_obj["points_xyz"],
        colors_rgb=best_obj["colors_rgb"],
    )

    return {
        "mask_path": mask_path,
        "npy_path": npy_path,
        "ply_path": ply_path,
    }


def save_sequence_summary_csv(
    results: list[dict],
    summary: dict,
    timestamp: str,
) -> Path:
    """
    Guarda un CSV simple con metricas por frame y resumen final.
    """

    csv_path = OUTPUT_DIR / f"sequence_metrics_{timestamp}.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "frame_idx,valid,distance_median,distance_p25,length_m,height_m,"
            "thickness_m,volume_liters,mask_area_px,quality_score\n"
        )

        for idx, result in enumerate(results):
            if result is None or result.get("closest_obj") is None:
                f.write(f"{idx},0,,,,,,,,\n")
                continue

            obj = result["closest_obj"]

            f.write(
                f"{idx},1,"
                f"{obj['distance_median']:.6f},"
                f"{obj['distance_p25']:.6f},"
                f"{obj['mask_length_m']:.6f},"
                f"{obj['mask_height_m']:.6f},"
                f"{obj['fish_thickness_m']:.6f},"
                f"{obj['fish_volume_liters']:.6f},"
                f"{obj['mask_area_px']},"
                f"{obj.get('quality_score', 0.0):.6f}\n"
            )

        f.write("\n")
        f.write("summary_metric,median,mean,p25,p75,iqr,min,max\n")

        for key in ["distance_m", "length_m", "height_m", "thickness_m", "volume_liters"]:
            stats = summary[key]
            f.write(
                f"{key},"
                f"{stats['median']:.6f},"
                f"{stats['mean']:.6f},"
                f"{stats['p25']:.6f},"
                f"{stats['p75']:.6f},"
                f"{stats['iqr']:.6f},"
                f"{stats['min']:.6f},"
                f"{stats['max']:.6f}\n"
            )

    return csv_path


def capture_frame_sequence(
    cap: cv2.VideoCapture,
    num_frames: int,
    frame_stride: int,
) -> list[np.ndarray]:
    """
    Captura una secuencia de frames desde RTSP.

    frame_stride permite espaciar los frames para dar tiempo a que el pez cambie de angulo.
    """

    frames = []
    frame_stride = max(1, frame_stride)

    while len(frames) < num_frames:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer frame durante captura multi-frame.")
            break

        frames.append(frame.copy())

        for _ in range(frame_stride - 1):
            ret_skip, _ = cap.read()

            if not ret_skip:
                break

    return frames


def process_frame_sequence(
    frames: list[np.ndarray],
    yolo_model: YOLO,
    sam_model: SAM,
    depth_model,
    transform,
    yolo_device,
    timestamp: str,
) -> tuple[np.ndarray | None, dict | None]:
    """
    Procesa multiples frames y combina resultados con estadisticas robustas.
    """

    annotated_frames = []
    result_infos = []

    print()
    print("========== Procesando secuencia multi-frame ==========")
    print(f"Frames a procesar: {len(frames)}")

    for idx, frame in enumerate(frames):
        frame_timestamp = f"{timestamp}_f{idx:03d}"

        print()
        print(f"--- Frame {idx + 1}/{len(frames)} ---")

        annotated, result_info = process_frame(
            frame_bgr=frame,
            yolo_model=yolo_model,
            sam_model=sam_model,
            depth_model=depth_model,
            transform=transform,
            yolo_device=yolo_device,
            timestamp=frame_timestamp,
            save_artifacts=SAVE_SEQUENCE_FRAME_ARTIFACTS,
        )

        annotated_frames.append(annotated)
        result_infos.append(result_info)

    summary = collect_sequence_summary(result_infos)

    if summary is None:
        print("No hubo resultados validos en la secuencia.")
        return annotated_frames[-1] if annotated_frames else None, None

    print_sequence_summary(summary)

    saved_paths = save_best_result_artifacts(summary, timestamp)
    csv_path = save_sequence_summary_csv(result_infos, summary, timestamp)

    print()
    print("Artefactos multi-frame guardados:")
    print(f"Mascara mejor frame: {saved_paths['mask_path']}")
    print(f"NPY mejor frame:     {saved_paths['npy_path']}")
    print(f"PLY mejor frame:     {saved_paths['ply_path']}")
    print(f"CSV metricas:        {csv_path}")

    best_result = summary["best_result"]
    best_idx = result_infos.index(best_result)
    best_annotated = annotated_frames[best_idx]

    final_annotated = draw_sequence_summary(
        image_bgr=best_annotated,
        summary=summary,
        origin=(20, 40),
    )

    result_info = {
        "summary": summary,
        "best_result": best_result,
        "closest_obj": summary["best_obj"],
        "mask_path": saved_paths["mask_path"],
        "npy_path": saved_paths["npy_path"],
        "ply_path": saved_paths["ply_path"],
        "csv_path": csv_path,
        "focal_px_used": best_result.get("focal_px_used"),
        "focal_source": best_result.get("focal_source"),
    }

    return final_annotated, result_info


def update_3d_viewer(viewer: PointCloudViewer, result_info: dict | None) -> None:
    if not result_info:
        return

    closest_obj = result_info.get("closest_obj")

    if closest_obj is None:
        return

    points_xyz = closest_obj["points_xyz"]

    if points_xyz.size == 0:
        return

    colors_rgb = depth_values_to_rgb(points_xyz[:, 2])

    viewer.update(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        title="Nube de puntos 3D - objeto mas cercano",
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
    

    # Si no se usa RTSP, procesar una imagen de prueba y mostrar resultados sin ventana de stream
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



    print(f"Conectando a: {RTSP_URL}")  

    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el stream RTSP")

    print()
    print("Controles:")
    print("  p = procesar")
    print("  q = salir")
    print()
    print("Configuracion multi-frame:")
    print(f"  USE_MULTIFRAME_ON_KEYPRESS = {USE_MULTIFRAME_ON_KEYPRESS}")
    print(f"  SEQUENCE_NUM_FRAMES = {SEQUENCE_NUM_FRAMES}")
    print(f"  SEQUENCE_FRAME_STRIDE = {SEQUENCE_FRAME_STRIDE}")
    print()

    try:
        while True:
            viewer.poll()

            ret, frame = cap.read()

            if not ret:
                print("No se pudo leer frame del stream")
                break

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                "Presiona 'p' para procesar | 'q' para salir",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Stream RTSP", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("p"):
                print()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if USE_MULTIFRAME_ON_KEYPRESS:
                    print("Capturando secuencia multi-frame...")
                    print(f"Frames: {SEQUENCE_NUM_FRAMES}")
                    print(f"Stride: {SEQUENCE_FRAME_STRIDE}")

                    frames = [frame.copy()]

                    extra_frames = capture_frame_sequence(
                        cap=cap,
                        num_frames=max(0, SEQUENCE_NUM_FRAMES - 1),
                        frame_stride=SEQUENCE_FRAME_STRIDE,
                    )

                    frames.extend(extra_frames)

                    annotated, result_info = process_frame_sequence(
                        frames=frames,
                        yolo_model=yolo_model,
                        sam_model=sam_model,
                        depth_model=depth_model,
                        transform=transform,
                        yolo_device=yolo_device,
                        timestamp=timestamp,
                    )

                    if annotated is None:
                        print("No se pudo generar resultado multi-frame.")
                        continue

                    raw_path = OUTPUT_DIR / f"sequence_first_frame_{timestamp}.jpg"
                    result_path = OUTPUT_DIR / f"sequence_result_{timestamp}.jpg"

                    cv2.imwrite(str(raw_path), frames[0])
                    cv2.imwrite(str(result_path), annotated)

                    print()
                    print(f"Primer frame guardado en: {raw_path}")
                    print(f"Resultado secuencia en:   {result_path}")

                else:
                    print("Procesando frame actual...")

                    frame_to_process = frame.copy()

                    annotated, result_info = process_frame(
                        frame_bgr=frame_to_process,
                        yolo_model=yolo_model,
                        sam_model=sam_model,
                        depth_model=depth_model,
                        transform=transform,
                        yolo_device=yolo_device,
                        timestamp=timestamp,
                        save_artifacts=True,
                    )

                    raw_path = OUTPUT_DIR / f"frame_{timestamp}.jpg"
                    result_path = OUTPUT_DIR / f"resultado_{timestamp}.jpg"

                    cv2.imwrite(str(raw_path), frame_to_process)
                    cv2.imwrite(str(result_path), annotated)

                    print()
                    print(f"Frame guardado en:     {raw_path}")
                    print(f"Resultado guardado en: {result_path}")

                update_3d_viewer(viewer, result_info)

                if SHOW_WINDOWS:
                    cv2.imshow("Resultado YOLO-World + SAM + Depth Pro", annotated)

    finally:
        cap.release()
        viewer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()