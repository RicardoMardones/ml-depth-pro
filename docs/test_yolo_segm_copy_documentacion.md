# Documentacion   `test_yolo_segm copy.py`

Este documento explica a fondo el script `test_yolo_segm copy.py`. La idea es que puedas entenderlo como un pipeline completo: toma una imagen o un stream RTSP, detecta peces con YOLO-World, refina la silueta con SAM/MobileSAM, estima profundidad con Depth Pro, calcula medidas aproximadas y guarda resultados visuales y nubes de puntos.

## Resumen ejecutivo

El script busca responder una pregunta practica:

> "En una imagen o video de una camara, puedo detectar un pez, segmentarlo, estimar a que distancia esta y aproximar su largo, alto, espesor y volumen?"

Para eso combina tres tipos de modelos:

- `YOLO-World`: detecta objetos usando clases configurables por texto, por ejemplo `fish`, `salmon fish`, `trout fish`.
- `SAM` o `MobileSAM`: recibe las cajas de YOLO y genera mascaras precisas del objeto.
- `Depth Pro`: estima un mapa de profundidad metrica monocular, es decir, profundidad en metros desde una sola imagen.

Luego el script transforma los pixeles segmentados en una nube de puntos 3D usando un modelo de camara pinhole. Con esa nube y con la mascara 2D estima dimensiones visibles y volumen aproximado.

## Flujo general

El flujo principal es:

1. Cargar configuracion desde variables de entorno.
2. Cargar modelos: YOLO-World, SAM/MobileSAM y Depth Pro.
3. Leer una imagen local o abrir un stream RTSP.
4. Cuando se procesa un frame:
   - detectar posibles peces con YOLO-World;
   - segmentarlos con SAM;
   - estimar profundidad con Depth Pro;
   - calcular distancia dentro de cada mascara;
   - convertir la mascara + profundidad a nube de puntos 3D;
   - estimar largo, alto, espesor y volumen;
   - filtrar detecciones poco confiables;
   - elegir el pez valido mas cercano;
   - guardar imagen anotada, mascara y nube de puntos.
5. Opcionalmente, procesar muchos frames y combinar resultados con estadisticas robustas.

## Entradas del script

El script puede trabajar en dos modos.

### Modo RTSP

Es el modo por defecto:

```powershell
python "test_yolo_segm copy.py"
```

Como `USE_RTSP` vale `1` por defecto, el script intenta conectarse a:

```text
rtsp://{RTSP_USER}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/Streaming/Channels/{RTSP_CHANNEL}
```

Las variables por defecto estan en el propio script:

```python
USER = os.getenv("RTSP_USER", "admin")
PASSWORD = os.getenv("RTSP_PASSWORD", "itg24chile")
NVR_IP = os.getenv("RTSP_IP", "10.22.100.22")
PORT = int(os.getenv("RTSP_PORT", "554"))
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "202")
```

Recomendacion importante: estas credenciales no deberian quedar fijas en codigo si el repo se comparte. Es mejor definirlas como variables de entorno.

Ejemplo en PowerShell:

```powershell
$env:RTSP_USER="admin"
$env:RTSP_PASSWORD="tu_password"
$env:RTSP_IP="10.22.100.22"
$env:RTSP_CHANNEL="202"
python "test_yolo_segm copy.py"
```

### Modo imagen local

Para probar sin camara:

```powershell
$env:USE_RTSP="0"
$env:TEST_IMAGE_PATH="huelmo_cap2_small.png"
python "test_yolo_segm copy.py"
```

En este modo procesa una sola imagen, guarda el resultado y termina, salvo que las ventanas esten activas.

## Salidas generadas

Por defecto el script guarda resultados en:

```text
outputs_stream/
```

Segun el modo y las banderas activas, puede generar:

- `resultado_test_YYYYMMDD_HHMMSS.jpg`: imagen anotada al procesar imagen local.
- `frame_YYYYMMDD_HHMMSS.jpg`: frame original capturado desde RTSP.
- `resultado_YYYYMMDD_HHMMSS.jpg`: resultado anotado de un frame individual.
- `sequence_first_frame_YYYYMMDD_HHMMSS.jpg`: primer frame de una secuencia.
- `sequence_result_YYYYMMDD_HHMMSS.jpg`: resultado final de una secuencia multi-frame.
- `mask_YYYYMMDD_HHMMSS.png`: mascara binaria del pez seleccionado.
- `pointcloud_YYYYMMDD_HHMMSS.npy`: nube de puntos en formato NumPy.
- `pointcloud_YYYYMMDD_HHMMSS.ply`: nube de puntos en formato PLY, compatible con Open3D, MeshLab o CloudCompare.
- `depth_map_pointcloud_YYYYMMDD_HHMMSS.ply`: nube de puntos de toda la escena si `SAVE_FULL_DEPTH_POINT_CLOUD=1`.
- `sequence_best_mask_YYYYMMDD_HHMMSS.png`: mascara del mejor frame multi-frame.
- `sequence_best_pointcloud_YYYYMMDD_HHMMSS.npy`: nube NumPy del mejor frame multi-frame.
- `sequence_best_pointcloud_YYYYMMDD_HHMMSS.ply`: nube PLY del mejor frame multi-frame.
- `sequence_metrics_YYYYMMDD_HHMMSS.csv`: metricas por frame y resumen estadistico.

## Configuracion principal

La mayoria de parametros se pueden cambiar con variables de entorno sin editar el script.

| Variable | Default | Para que sirve |
| --- | --- | --- |
| `USE_RTSP` | `1` | Usa stream RTSP si vale `1`; usa imagen local si vale `0`. |
| `TEST_IMAGE_PATH` | `huelmo_cap2_small.png` | Imagen local de prueba. |
| `RTSP_USER` | `admin` | Usuario RTSP. |
| `RTSP_PASSWORD` | `itg24chile` | Password RTSP. |
| `RTSP_IP` | `10.22.100.22` | IP del NVR/camara. |
| `RTSP_PORT` | `554` | Puerto RTSP. |
| `RTSP_CHANNEL` | `202` | Canal del stream. |
| `YOLO_WEIGHTS` | `yolov8s-worldv2.pt` | Pesos del detector YOLO-World. |
| `SAM_WEIGHTS` | `mobile_sam.pt` | Pesos de SAM/MobileSAM. |
| `CONFIDENCE_THRESHOLD` | `0.20` | Confianza minima para detecciones YOLO. |
| `OUTPUT_DIR` | `outputs_stream` | Carpeta donde se guardan resultados. |
| `ENABLE_3D_VIEWER` | `1` | Activa visor 3D con Open3D si esta instalado. |
| `SHOW_WINDOWS` | `1` | Muestra ventanas OpenCV. |
| `SAVE_FULL_DEPTH_POINT_CLOUD` | `0` | Guarda nube 3D de toda la escena si vale `1`. |
| `MAX_POINT_CLOUD_POINTS` | `60000` | Limite de puntos para nube del pez. |
| `MAX_DEPTH_VIEWER_POINTS` | `220000` | Limite de puntos para nube completa de profundidad. |

## Clases objetivo

El script configura YOLO-World con estas clases:

```python
TARGET_CLASSES = [
    "fish",
    "salmon fish",
    "trout fish",
]
```

Esto significa que YOLO no se usa como un detector cerrado de clases COCO, sino como un detector abierto donde se le indican etiquetas textuales. Despues de predecir, el script descarta cualquier deteccion cuyo nombre no este en `TARGET_CLASS_SET`.

El texto mostrado en la interfaz usa:

```python
TARGET_LABEL = "pez"
```

## Concepto clave: camara pinhole y longitud focal

Para convertir pixeles con profundidad en coordenadas 3D, el script usa el modelo pinhole:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
```

Donde:

- `u`, `v`: coordenadas del pixel.
- `cx`, `cy`: centro optico asumido, aqui el centro de la imagen.
- `Z`: profundidad estimada por Depth Pro, en metros.
- `fx`, `fy`: focal en pixeles. El script asume `fx = fy = focal_px`.

Hay dos posibles fuentes para la focal:

1. Depth Pro puede entregar `focallength_px`.
2. Si Depth Pro no entrega focal valida, se calcula usando el HFOV del lente SL-0041.

La formula usada para obtener focal desde HFOV es:

```text
focal_px = image_width_px / (2 * tan(hfov_rad / 2))
```

Con:

```python
LENS_HFOV_DEG = 127.3
```

Importante: esta es una aproximacion de camara ideal. Si hay distorsion fuerte del lente, carcasa bajo agua o refraccion, las medidas pueden desviarse.

## Funciones auxiliares

### `clamp(value, low, high)`

Limita un numero entero dentro de un rango. Se usa para asegurar que las coordenadas de las cajas no salgan de la imagen.

Ejemplo:

```python
x1i = clamp(int(round(x1)), 0, orig_w - 1)
```

### `focal_px_from_hfov(image_width_px, hfov_deg)`

Convierte el campo de vision horizontal del lente a una focal en pixeles. Esto es necesario para convertir tamanos en pixeles a metros.

### `get_depthpro_focal_px(prediction)`

Extrae `focallength_px` desde la salida de Depth Pro. Puede venir como tensor de PyTorch o como numero normal, por eso la funcion contempla ambos casos.

### `draw_multiline_text(...)`

Dibuja varias lineas de texto sobre una imagen usando OpenCV. Se usa para mostrar resumen de distancia, largo, alto, volumen y depuracion.

### `overlay_mask(...)`

Superpone una mascara sobre la imagen con transparencia. Sirve para pintar visualmente la zona segmentada por SAM.

### `resize_mask_to_shape(...)`

Redimensiona una mascara booleana. Es importante porque:

- SAM devuelve mascaras en tamano de imagen original.
- Depth Pro puede devolver un mapa de profundidad con otra resolucion.

Para no crear valores intermedios raros, usa interpolacion `INTER_NEAREST`.

## Estimacion de medidas

El script calcula medidas de dos maneras complementarias.

### Medidas desde nube 3D: `estimate_size_from_point_cloud`

Recibe puntos `XYZ` en metros y calcula:

- extension en X;
- extension en Y;
- variacion visible en Z;
- volumen de una caja alineada a los ejes;
- dimensiones principales mediante PCA 3D;
- volumen aproximado de caja PCA;
- volumen elipsoidal visible.

Usa percentiles 2 y 98 en vez de minimos y maximos absolutos. Esto evita que un punto aislado muy erroneo agrande artificialmente las medidas.

Tambien usa SVD/PCA para orientar la nube segun sus ejes principales. Esto ayuda cuando el pez esta inclinado respecto a los ejes de la imagen.

Limitacion: la camara ve solo la superficie visible del pez, no un volumen cerrado real. Por eso estos volumenes son aproximaciones geometricas de la parte visible.

### Medidas desde mascara 2D: `estimate_oriented_size_from_mask_2d`

Esta es una medicion muy relevante para el largo visible del pez.

Pasos:

1. Obtiene todos los pixeles donde la mascara es verdadera.
2. Aplica PCA 2D sobre esas coordenadas.
3. Proyecta la forma sobre sus ejes principales.
4. Usa percentiles 2 y 98 para calcular largo y alto en pixeles.
5. Convierte pixeles a metros con:

```text
length_m = length_px * depth_m / focal_px
height_m = height_px * depth_m / focal_px
```

Ventaja: si el pez esta diagonal en la imagen, el largo no depende solo del ancho de la caja rectangular.

### Volumen empirico: `estimate_fish_empirical_volume`

El script no mide el espesor real del pez. Lo estima asi:

```text
thickness_m = FISH_THICKNESS_RATIO * height_m
volume_m3 = FISH_VOLUME_SHAPE_FACTOR * length_m * height_m * thickness_m
```

Con defaults:

```python
FISH_THICKNESS_RATIO = 0.45
FISH_VOLUME_SHAPE_FACTOR = 0.55
```

Interpretacion:

- `FISH_THICKNESS_RATIO`: asume que el espesor es 45% del alto visible.
- `FISH_VOLUME_SHAPE_FACTOR`: reduce el volumen de una caja rectangular para acercarse a una forma organica.

Esto debe leerse como una estimacion inicial, no como volumen real certificado.

## Filtros de calidad

La funcion `passes_detection_filters` descarta detecciones que parecen poco confiables.

Condiciones por defecto:

```python
MIN_MASK_AREA_PX = 1500
MIN_DISTANCE_M = 0.20
MAX_DISTANCE_M = 4.00
MIN_FISH_LENGTH_M = 0.05
MAX_FISH_LENGTH_M = 1.50
```

Una deteccion se descarta si:

- la mascara es demasiado pequena;
- la distancia esta fuera del rango esperado;
- el largo estimado esta fuera del rango esperado para un pez.

Esto ayuda a reducir falsos positivos, reflejos, ruido de profundidad o segmentos demasiado pequenos.

## Nubes de puntos

### `create_point_cloud_from_mask`

Convierte solo los pixeles del pez segmentado a puntos 3D.

Entrada principal:

- `depth_map`: profundidad en metros.
- `mask_depth`: mascara del pez redimensionada al tamano del mapa de profundidad.
- `frame_bgr`: imagen original para extraer color.
- `focal_px`: focal usada para reconstruir 3D.

Salida:

- `points_xyz`: arreglo `N x 3` con coordenadas en metros.
- `colors_rgb`: colores `N x 3` si `SAVE_POINT_CLOUD_COLOR=1`.

Si hay demasiados pixeles, toma una muestra aleatoria limitada por `MAX_POINT_CLOUD_POINTS`.

### `create_point_cloud_from_depth_map`

Convierte todo el mapa de profundidad a nube 3D. Si se entrega una mascara, los puntos del objeto se resaltan en verde. Esta salida puede ser pesada, por eso se controla con:

```python
SAVE_FULL_DEPTH_POINT_CLOUD = False
MAX_DEPTH_VIEWER_POINTS = 220000
```

### `save_ply`

Guarda una nube de puntos en formato PLY ASCII. Este formato se puede abrir con herramientas como:

- Open3D;
- CloudCompare;
- MeshLab.

## Visor 3D: `PointCloudViewer`

Esta clase abre una ventana no bloqueante de Open3D para ver la nube de puntos del pez mas cercano.

Detalles:

- Si Open3D no esta instalado, el script no falla; solo omite la ventana 3D.
- Invierte el eje `Y` para que la nube se vea mas natural en la visualizacion.
- Puede mostrar un eje 3D si `SHOW_3D_AXIS=1`.
- `poll()` mantiene viva la ventana mientras OpenCV esta mostrando video.

## Segmentacion con SAM

La funcion `extract_sam_masks` usa las cajas de YOLO como prompts para SAM:

```python
results = sam_model.predict(
    source=frame_bgr,
    bboxes=bboxes,
    device=yolo_device,
    verbose=False,
)
```

YOLO entrega cajas aproximadas. SAM convierte esas cajas en mascaras pixel a pixel, que son mucho mejores para:

- calcular area real de la silueta;
- extraer solo profundidad del pez;
- crear la nube 3D del objeto;
- evitar incluir fondo dentro de la caja.

## Carga de modelos

La funcion `load_models` hace tres cosas:

1. Carga YOLO-World desde `YOLO_WEIGHTS`.
2. Configura sus clases con `yolo_model.set_classes(TARGET_CLASSES)`.
3. Carga SAM/MobileSAM desde `SAM_WEIGHTS`.
4. Carga Depth Pro con `depth_pro.create_model_and_transforms`.

El dispositivo se decide en `main`:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yolo_device = 0 if device.type == "cuda" else "cpu"
precision = torch.half if device.type == "cuda" else torch.float32
```

Si hay GPU CUDA, usa media precision (`torch.half`) para acelerar Depth Pro.

## Procesamiento de un frame

La funcion central es `process_frame`. Es el corazon del script.

### Paso 1: preparar imagen y focal de respaldo

Obtiene tamano original:

```python
orig_h, orig_w = frame_bgr.shape[:2]
```

Calcula focal del lente:

```python
focal_px_lens = focal_px_from_hfov(orig_w, LENS_HFOV_DEG)
```

### Paso 2: detectar con YOLO-World

Ejecuta:

```python
yolo_results = yolo_model.predict(
    source=frame_bgr,
    conf=CONFIDENCE_THRESHOLD,
    device=yolo_device,
    verbose=False,
)
```

Luego filtra:

- cajas sin area;
- clases que no esten en `TARGET_CLASSES`;
- coordenadas invalidas.

Cada deteccion guardada contiene:

- `box_int`;
- `conf`;
- `cls`;
- `name`;
- `area`.

### Paso 3: segmentar con SAM

Convierte cada caja a mascara:

```python
bboxes = [d["box_int"] for d in detections]
masks = extract_sam_masks(...)
```

Si SAM no devuelve mascaras, se termina el procesamiento del frame.

### Paso 4: estimar profundidad

Convierte BGR a RGB:

```python
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

Aplica transform de Depth Pro y ejecuta inferencia:

```python
depth_input = transform(frame_rgb)
prediction = depth_model.infer(depth_input, f_px=None)
```

Luego obtiene:

- `depth_map`: matriz de profundidad en metros.
- `focal_px_depthpro`: focal estimada por Depth Pro, si existe.

Si Depth Pro no entrega focal valida, usa la focal calculada desde HFOV.

### Paso 5: medir cada mascara

Para cada deteccion y mascara:

1. Redimensiona la mascara al tamano del mapa de profundidad.
2. Extrae profundidades validas dentro de la mascara.
3. Calcula estadisticas:
   - percentil 10;
   - percentil 25;
   - mediana;
   - percentil 75;
   - percentil 90;
   - media;
   - minima;
   - maxima.
4. Genera nube de puntos 3D.
5. Calcula medidas desde nube 3D.
6. Calcula largo/alto desde mascara 2D.
7. Estima espesor y volumen empirico.
8. Aplica filtros de calidad.
9. Calcula `quality_score`.

El `quality_score` se calcula asi:

```text
quality_score = confidence * log1p(mask_area_px) / (1 + depth_iqr)
```

Donde `depth_iqr = distance_p75 - distance_p25`.

Lectura intuitiva:

- sube si YOLO esta confiado;
- sube si la mascara tiene buena area;
- baja si la profundidad dentro de la mascara es muy dispersa.

### Paso 6: elegir el pez mas cercano

Entre candidatos validos, elige el de menor `distance_p25`:

```python
closest_obj = min(
    valid_candidates,
    key=lambda detection: detection["distance_p25"],
)
```

Usar P25 en vez de minima evita que un pixel aislado muy cercano determine el objeto. Es una eleccion robusta: toma una distancia cercana, pero no tan sensible a outliers.

### Paso 7: guardar artefactos

Si `save_artifacts=True`, guarda:

- mascara PNG;
- nube NumPy `.npy`;
- nube PLY `.ply`;
- opcionalmente nube PLY de toda la escena.

### Paso 8: imprimir resultados

El script imprime en consola:

- clase;
- confianza;
- bounding box;
- score;
- estadisticas de distancia;
- largo y alto estimados;
- area de mascara;
- dimensiones 3D;
- volumenes aproximados;
- cantidad de puntos;
- rutas guardadas.

### Paso 9: dibujar imagen anotada

Sobre el frame dibuja:

- mascaras coloreadas;
- cajas;
- etiquetas por deteccion;
- resumen del objeto mas cercano;
- informacion de depuracion.

Colores:

- verde: candidato valido mas cercano;
- azul: otros candidatos validos;
- rojo: candidatos descartados.

## Procesamiento multi-frame

El modo multi-frame intenta mejorar estabilidad midiendo varios frames.

Variables relevantes:

| Variable | Default | Significado |
| --- | --- | --- |
| `USE_MULTIFRAME_ON_KEYPRESS` | `1` | Si vale `1`, al presionar `p` procesa una secuencia. |
| `SEQUENCE_NUM_FRAMES` | `120` | Cantidad de frames a procesar. |
| `SEQUENCE_FRAME_STRIDE` | `2` | Captura un frame y salta `stride - 1` frames. |
| `SAVE_SEQUENCE_FRAME_ARTIFACTS` | `0` | Guarda artefactos de cada frame individual si vale `1`. |

### `capture_frame_sequence`

Lee varios frames desde el stream RTSP. Si `frame_stride=2`, guarda un frame y salta uno.

Esto puede ayudar cuando el pez cambia levemente de posicion o angulo entre frames.

### `process_frame_sequence`

Procesa cada frame llamando a `process_frame`.

Luego llama a `collect_sequence_summary`, que conserva solo resultados validos.

### `robust_stats`

Calcula estadisticas robustas:

- mediana;
- media;
- percentil 25;
- percentil 75;
- IQR;
- desviacion estandar;
- minimo;
- maximo.

La mediana y el IQR suelen ser mas utiles que la media cuando hay errores puntuales.

### `collect_sequence_summary`

Agrupa medidas de todos los frames validos:

- distancia;
- largo;
- alto;
- espesor;
- volumen;
- score de calidad.

Tambien escoge el mejor frame segun `quality_score`.

### `save_sequence_summary_csv`

Guarda un CSV con una fila por frame:

```text
frame_idx,valid,distance_median,distance_p25,length_m,height_m,thickness_m,volume_liters,mask_area_px,quality_score
```

Y al final agrega un resumen estadistico por metrica.

## Programa principal

La funcion `main` decide entre imagen local y RTSP.

### Si `USE_RTSP=0`

1. Lee `TEST_IMAGE_PATH`.
2. Procesa la imagen con `process_frame`.
3. Guarda `resultado_test_...jpg`.
4. Actualiza visor 3D si esta activado.
5. Muestra ventana hasta presionar `q` o `ESC`.

### Si `USE_RTSP=1`

1. Abre el stream RTSP.
2. Muestra la ventana `Stream RTSP`.
3. Espera teclas:
   - `p`: procesar;
   - `q`: salir.
4. Si `USE_MULTIFRAME_ON_KEYPRESS=1`, captura y procesa secuencia.
5. Si no, procesa solo el frame actual.
6. Guarda resultados y actualiza visor 3D.

## Como ejecutar ejemplos utiles

### Prueba rapida con imagen local y sin ventanas

Util para servidores o para comprobar que el pipeline corre:

```powershell
$env:USE_RTSP="0"
$env:SHOW_WINDOWS="0"
$env:ENABLE_3D_VIEWER="0"
python "test_yolo_segm copy.py"
```

### Procesar RTSP frame por frame, no secuencia

```powershell
$env:USE_RTSP="1"
$env:USE_MULTIFRAME_ON_KEYPRESS="0"
python "test_yolo_segm copy.py"
```

Presiona `p` para procesar el frame actual.

### Procesar secuencia mas corta

```powershell
$env:USE_RTSP="1"
$env:SEQUENCE_NUM_FRAMES="20"
$env:SEQUENCE_FRAME_STRIDE="2"
python "test_yolo_segm copy.py"
```

### Guardar nube completa de profundidad

```powershell
$env:SAVE_FULL_DEPTH_POINT_CLOUD="1"
python "test_yolo_segm copy.py"
```

Esta opcion puede generar archivos grandes.

## Como interpretar los resultados

### Distancia

La distancia principal mostrada es la mediana de profundidad dentro de la mascara. Tambien se muestra `P25`, que se usa para elegir el objeto mas cercano.

- `P10` y `P25`: partes mas cercanas del objeto.
- `Mediana`: distancia tipica del objeto.
- `P75` y `P90`: partes mas lejanas.
- `IQR`: dispersion entre P25 y P75; si es alto, la profundidad dentro de la mascara es inestable o el objeto tiene mucha variacion.

### Largo y alto

El largo y alto principales vienen de la mascara 2D orientada por PCA, no de la caja YOLO. Esto suele ser mejor para peces inclinados.

### Volumen

Hay varios volumenes:

- `BBox Vol 3D`: volumen de caja alineada a ejes XYZ.
- `PCA BBox Vol 3D`: volumen de caja orientada por PCA 3D.
- `Elipsoide 3D visible`: aproximacion elipsoidal de la nube visible.
- `Vol pez empirico`: modelo simple usando largo, alto y espesor estimado.

Para reportes practicos, el volumen empirico en litros es el mas directo, pero debe calibrarse con mediciones reales si se necesita precision.

## Supuestos importantes

El script asume:

- Depth Pro entrega profundidad metrica razonable para la escena.
- La focal estimada por Depth Pro o por HFOV es representativa.
- El centro optico esta en el centro de la imagen.
- `fx` y `fy` son iguales.
- La mascara de SAM corresponde realmente al pez.
- El pez esta suficientemente visible.
- La refraccion, distorsion de lente y movimiento no arruinan la estimacion.

En escenas bajo agua, estos supuestos pueden ser fragiles. La calibracion empirica con objetos de tamano conocido es muy recomendable.

## Posibles fuentes de error

### 1. Deteccion incorrecta

YOLO-World puede detectar reflejos, partes del fondo o peces parcialmente visibles. Los filtros ayudan, pero no garantizan deteccion perfecta.

### 2. Segmentacion incorrecta

SAM puede incluir agua, jaula, sombra o partes de otro pez. Esto afecta area, profundidad y volumen.

### 3. Profundidad monocular

Depth Pro estima profundidad desde una sola imagen. Aunque es metrica, puede equivocarse si la escena no se parece a sus datos de entrenamiento o si hay condiciones dificiles.

### 4. Refraccion y lente gran angular

El script usa un modelo pinhole ideal. Si hay lente gran angular, carcasa, agua o distorsion, conviene calibrar o corregir la imagen antes.

### 5. Volumen no real

El volumen empirico no mide el cuerpo completo. Estima una forma 3D a partir de la silueta visible y parametros fijos.

## Consejos para calibrar

Para mejorar precision:

1. Grabar o fotografiar un objeto de largo conocido a varias distancias.
2. Comparar el largo estimado por el script con el largo real.
3. Ajustar `LENS_HFOV_DEG` si se usa focal por lente.
4. Ajustar `FISH_THICKNESS_RATIO` y `FISH_VOLUME_SHAPE_FACTOR` con peces medidos realmente.
5. Revisar si la focal de Depth Pro es estable entre frames.
6. Guardar CSV multi-frame y analizar variabilidad.

## Problemas frecuentes

### "No se pudo abrir el stream RTSP"

Revisar:

- IP;
- usuario/password;
- canal RTSP;
- conectividad de red;
- si otra app puede abrir la misma URL.

### "Sin detecciones YOLO-World"

Posibles causas:

- el umbral `CONFIDENCE_THRESHOLD` esta muy alto;
- las clases textuales no describen bien el objeto;
- la imagen esta borrosa u oscura;
- el pez aparece muy pequeno.

Prueba:

```powershell
$env:CONFIDENCE_THRESHOLD="0.10"
```

### "SAM no genero mascaras"

Puede pasar si las cajas son invalidas, muy pequenas o si el modelo SAM no carga bien.

Revisar:

- que `mobile_sam.pt` exista;
- probar otro peso SAM;
- revisar si YOLO genera cajas razonables.

### "Sin candidatos validos tras filtros"

El script detecto algo, pero lo descarto por area, distancia o largo.

Puedes ajustar:

```powershell
$env:MIN_MASK_AREA_PX="500"
$env:MAX_DISTANCE_M="8.0"
$env:MAX_FISH_LENGTH_M="2.0"
```

### Open3D no abre visor

Si `open3d` no esta instalado, el script continua sin visor 3D.

Puedes desactivar visor:

```powershell
$env:ENABLE_3D_VIEWER="0"
```

## Lectura conceptual del pipeline

Una forma sencilla de pensar el script es:

```text
Imagen
  -> YOLO-World encuentra cajas de posibles peces
  -> SAM convierte cajas en siluetas precisas
  -> Depth Pro estima profundidad por pixel
  -> Mascara + profundidad seleccionan solo pixeles del pez
  -> Modelo pinhole convierte pixeles a XYZ
  -> PCA y percentiles estiman medidas robustas
  -> Filtros descartan candidatos raros
  -> Se guarda el mejor resultado
```

## Mejoras recomendadas

Si este script va a evolucionar, estas mejoras tendrian alto impacto:

1. Renombrar el archivo a algo sin espacio, por ejemplo `measure_fish_depth.py`.
2. Mover configuracion a un `.env` o archivo YAML.
3. Eliminar credenciales por defecto del codigo.
4. Separar el script en modulos:
   - `config.py`;
   - `models.py`;
   - `geometry.py`;
   - `processing.py`;
   - `io_outputs.py`.
5. Guardar resultados tambien en JSON para trazabilidad.
6. Agregar calibracion de lente y correccion de distorsion.
7. Agregar seguimiento temporal para medir el mismo pez entre frames.
8. Agregar tests unitarios para formulas geometricas y filtros.
9. Hacer que `process_frame` devuelva estructuras tipadas, por ejemplo `dataclass`.
10. Registrar version de modelos y parametros usados en cada salida.

## Mini glosario

- `BGR`: formato de color usado por OpenCV.
- `RGB`: formato de color comun usado por modelos de vision.
- `Bounding box`: rectangulo que encierra un objeto detectado.
- `Mascara`: imagen binaria que marca pixeles pertenecientes al objeto.
- `Depth map`: matriz donde cada pixel contiene una distancia estimada.
- `Focal en pixeles`: parametro de camara necesario para convertir pixeles a escala metrica.
- `PCA`: tecnica que encuentra ejes principales de una forma o nube de puntos.
- `IQR`: rango intercuartil, `P75 - P25`; mide dispersion robusta.
- `PLY`: formato de archivo para nubes de puntos 3D.

## En una frase

`test_yolo_segm copy.py` es un prototipo de medicion visual: detecta peces, segmenta su silueta, estima profundidad monocular, reconstruye puntos 3D del objeto y calcula medidas aproximadas para seleccionar y reportar el pez valido mas cercano.
