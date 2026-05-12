from datetime import datetime
from pathlib import Path
import math
import os

import cv2
import depth_pro
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# Configuracion RTSP
# ============================================================
# Recomendado: usar variables de entorno para no dejar credenciales en el codigo.
#
# PowerShell ejemplo:
#   $env:RTSP_USER="admin"
#   $env:RTSP_PASSWORD="tu_password"
#   $env:RTSP_IP="172.16.2.184"

USER = os.getenv("RTSP_USER", "admin")
PASSWORD = os.getenv("RTSP_PASSWORD", "123456")
# NVR_IP = os.getenv("RTSP_IP", "10.22.100.22")
NVR_IP = os.getenv("RTSP_IP", "172.16.2.75")

PORT = int(os.getenv("RTSP_PORT", "554"))

RTSP_URL = f"rtsp://{USER}:{PASSWORD}@{NVR_IP}:{PORT}/Streaming/Channels/201"

# Si USE_RTSP=False, procesa una imagen local y termina. Sirve para probar rapido.
USE_RTSP = os.getenv("USE_RTSP", "1") == "1"
TEST_IMAGE_PATH = os.getenv("TEST_IMAGE_PATH", "huelmo_cap2_small.png")


# ============================================================
# Configuracion lente SL-0041
# ============================================================

LENS_HFOV_DEG = 127.3


# ============================================================
# Configuracion YOLO-World / Depth Pro
# ============================================================

YOLO_WEIGHTS = "yolov8s-worldv2.pt"

CONFIDENCE_THRESHOLD = 0.20
TARGET_CLASSES = [
    "fish",
    "salmon fish",
    "trout fish",
]
TARGET_CLASS_SET = set(TARGET_CLASSES)
TARGET_LABEL = "pez"

# Region central usada para estimar profundidad.
# 0.50 = usa el 50% central del bounding box.
CENTRAL_CROP_RATIO = 0.50

OUTPUT_DIR = Path("outputs_stream")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Funciones auxiliares
# ============================================================

def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def focal_px_from_hfov(image_width_px: int, hfov_deg: float) -> float:
    """Calcula focal en pixeles desde HFOV para una camara pinhole ideal."""

    hfov_rad = math.radians(hfov_deg)
    return image_width_px / (2.0 * math.tan(hfov_rad / 2.0))


def central_region(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    """Retorna una region central dentro de un bounding box."""

    ratio = max(0.05, min(ratio, 1.0))
    box_w = x2 - x1
    box_h = y2 - y1

    margin_x = int(box_w * (1.0 - ratio) / 2.0)
    margin_y = int(box_h * (1.0 - ratio) / 2.0)

    rx1 = x1 + margin_x
    ry1 = y1 + margin_y
    rx2 = x2 - margin_x
    ry2 = y2 - margin_y

    if rx2 <= rx1 or ry2 <= ry1:
        return x1, y1, x2, y2

    return rx1, ry1, rx2, ry2


def get_depthpro_focal_px(prediction: dict) -> float | None:
    """Extrae la longitud focal estimada por Depth Pro."""

    focallength_px = prediction.get("focallength_px", None)
    if focallength_px is None:
        return None

    if isinstance(focallength_px, torch.Tensor):
        return float(focallength_px.detach().cpu().item())

    return float(focallength_px)


def estimate_object_size_from_bbox(
    bbox_width_px: int,
    bbox_height_px: int,
    distance_m: float,
    focal_px: float,
) -> tuple[float, float]:
    """
    Estima ancho y alto fisico usando modelo pinhole.

    Esto estima el tamano del bounding box visible, no el tamano real completo.
    """

    if focal_px <= 0:
        return 0.0, 0.0

    width_m = bbox_width_px * distance_m / focal_px
    height_m = bbox_height_px * distance_m / focal_px
    return float(width_m), float(height_m)


def draw_multiline_text(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
    line_height: int = 26,
) -> None:
    """Dibuja varias lineas de texto sobre la imagen."""

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


def load_models(device: torch.device, precision: torch.dtype):
    print("Cargando YOLO-World...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.set_classes(TARGET_CLASSES)

    print(f"Clases configuradas: {TARGET_CLASSES}")
    print(f"Clases del modelo YOLO: {yolo_model.names}")

    print("Cargando Depth Pro...")
    depth_model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=precision,
    )
    depth_model.eval()

    return yolo_model, depth_model, transform


# ============================================================
# Procesamiento del frame capturado
# ============================================================

def process_frame(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    depth_model,
    transform,
    yolo_device,
) -> np.ndarray:
    """
    Procesa un frame:
    - Detecta objetos con YOLO-World.
    - Filtra las clases definidas en TARGET_CLASSES.
    - Calcula profundidad con Depth Pro.
    - Selecciona el objeto detectado mas cercano.
    - Estima tamano usando la focal de Depth Pro o HFOV como respaldo.
    - Retorna imagen anotada.
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
        return annotated

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
        area = max(0.0, float((x2 - x1) * (y2 - y1)))
        if area <= 0:
            continue

        cls_id = int(box.cls.item())
        class_name = yolo_model.names.get(cls_id, str(cls_id))

        # Con YOLO-World y set_classes, normalmente todas las clases vienen de TARGET_CLASSES.
        # Mantenemos el filtro para evitar resultados inesperados si se cambian pesos/clases.
        if class_name not in TARGET_CLASS_SET:
            continue

        detections.append(
            {
                "box": [float(x1), float(y1), float(x2), float(y2)],
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
        return annotated

    # ========================================================
    # 2. Profundidad con Depth Pro
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

    scale_x = depth_w / orig_w
    scale_y = depth_h / orig_h

    # ========================================================
    # 3. Calcular profundidad y tamano de cada deteccion
    # ========================================================

    detections_with_depth = []
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]

        x1 = clamp(x1, 0, orig_w - 1)
        x2 = clamp(x2, 0, orig_w - 1)
        y1 = clamp(y1, 0, orig_h - 1)
        y2 = clamp(y2, 0, orig_h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        dx1 = clamp(int(round(x1 * scale_x)), 0, depth_w - 1)
        dx2 = clamp(int(round(x2 * scale_x)), 0, depth_w - 1)
        dy1 = clamp(int(round(y1 * scale_y)), 0, depth_h - 1)
        dy2 = clamp(int(round(y2 * scale_y)), 0, depth_h - 1)

        if dx2 <= dx1 or dy2 <= dy1:
            continue

        rx1, ry1, rx2, ry2 = central_region(dx1, dy1, dx2, dy2, CENTRAL_CROP_RATIO)
        object_depth_region = depth_map[ry1:ry2, rx1:rx2]
        valid_depths = object_depth_region[
            np.isfinite(object_depth_region) & (object_depth_region > 0)
        ]

        if valid_depths.size == 0:
            continue

        distance_median = float(np.median(valid_depths))
        distance_mean = float(np.mean(valid_depths))
        distance_min = float(np.min(valid_depths))
        distance_max = float(np.max(valid_depths))

        bbox_width_px = x2 - x1
        bbox_height_px = y2 - y1
        width_m, height_m = estimate_object_size_from_bbox(
            bbox_width_px=bbox_width_px,
            bbox_height_px=bbox_height_px,
            distance_m=distance_median,
            focal_px=focal_px_used,
        )

        detection["box_int"] = [x1, y1, x2, y2]
        detection["depth_box"] = [dx1, dy1, dx2, dy2]
        detection["distance_median"] = distance_median
        detection["distance_mean"] = distance_mean
        detection["distance_min"] = distance_min
        detection["distance_max"] = distance_max
        detection["bbox_width_px"] = bbox_width_px
        detection["bbox_height_px"] = bbox_height_px
        detection["width_m"] = width_m
        detection["height_m"] = height_m
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
        print(f"No hay profundidad valida en las detecciones de {TARGET_LABEL}.")
        return annotated

    # ========================================================
    # 4. Elegir el objeto mas cercano
    # ========================================================

    closest_obj = min(detections_with_depth, key=lambda detection: detection["distance_median"])
    x1, y1, x2, y2 = closest_obj["box_int"]

    # ========================================================
    # 5. Imprimir resultados
    # ========================================================

    print()
    print(f"{TARGET_LABEL.capitalize()} mas cercano:")
    print(f"Clase:       {closest_obj['name']}")
    print(f"Confianza:   {closest_obj['conf']:.2f}")
    print(f"Box:         {x1}, {y1}, {x2}, {y2}")
    print(f"Box px:      {closest_obj['bbox_width_px']} x {closest_obj['bbox_height_px']} px")

    print()
    print("Distancia estimada:")
    print(f"Mediana:     {closest_obj['distance_median']:.3f} m")
    print(f"Media:       {closest_obj['distance_mean']:.3f} m")
    print(f"Minima:      {closest_obj['distance_min']:.3f} m")
    print(f"Maxima:      {closest_obj['distance_max']:.3f} m")

    print()
    print("Tamano estimado del bounding box:")
    print(f"Ancho:       {closest_obj['width_m']:.3f} m")
    print(f"Alto:        {closest_obj['height_m']:.3f} m")

    # ========================================================
    # 6. Dibujar resultados
    # ========================================================

    for detection in detections_with_depth:
        bx1, by1, bx2, by2 = detection["box_int"]
        is_closest = detection is closest_obj
        color = (0, 255, 0) if is_closest else (255, 0, 0)
        thickness = 3 if is_closest else 1

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness)

        label = (
            f"{detection['name']} "
            f"{detection['distance_median']:.2f}m "
            f"{detection['width_m']:.2f}x{detection['height_m']:.2f}m"
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
        f"{TARGET_LABEL.capitalize()} mas cercano",
        f"Dist: {closest_obj['distance_median']:.2f} m",
        f"Tam: {closest_obj['width_m']:.2f} x {closest_obj['height_m']:.2f} m",
        f"fx usada: {focal_px_used:.1f}px ({focal_source})",
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
        f"HFOV SL-0041: {LENS_HFOV_DEG:.1f} deg",
        f"fx lente: {focal_px_lens:.1f}px",
    ]
    if focal_px_depthpro is not None:
        debug_lines.append(f"fx DepthPro: {focal_px_depthpro:.1f}px")

    draw_multiline_text(
        annotated,
        debug_lines,
        origin=(20, 170),
        color=(0, 255, 255),
        font_scale=0.65,
        thickness=2,
        line_height=26,
    )

    return annotated


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
    yolo_model, depth_model, transform = load_models(device=device, precision=precision)

    if not USE_RTSP:
        frame = cv2.imread(TEST_IMAGE_PATH)
        if frame is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {TEST_IMAGE_PATH}")

        annotated = process_frame(
            frame_bgr=frame,
            yolo_model=yolo_model,
            depth_model=depth_model,
            transform=transform,
            yolo_device=yolo_device,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = OUTPUT_DIR / f"resultado_test_{timestamp}.jpg"
        cv2.imwrite(str(result_path), annotated)
        print(f"Resultado guardado en: {result_path}")
        return

    print(f"Conectando a: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el stream RTSP")

    print()
    print("Controles:")
    print("  p = procesar frame actual")
    print("  q = salir")
    print()

    while True:
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
            print("Procesando frame actual...")

            frame_to_process = frame.copy()
            annotated = process_frame(
                frame_bgr=frame_to_process,
                yolo_model=yolo_model,
                depth_model=depth_model,
                transform=transform,
                yolo_device=yolo_device,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = OUTPUT_DIR / f"frame_{timestamp}.jpg"
            result_path = OUTPUT_DIR / f"resultado_{timestamp}.jpg"

            cv2.imwrite(str(raw_path), frame_to_process)
            cv2.imwrite(str(result_path), annotated)

            print()
            print(f"Frame guardado en:     {raw_path}")
            print(f"Resultado guardado en: {result_path}")

            cv2.imshow("Resultado YOLO + Depth Pro", annotated)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
