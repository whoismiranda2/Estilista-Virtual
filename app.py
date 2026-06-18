from flask import Flask, render_template, request, jsonify
import os, subprocess, json, shutil, time
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

CATEGORIAS = ["top", "pantalon", "vestido", "falda", "short", "otro"]

# ─────────────────────────────────────────────────────────────────────────────
# Ruta al modelo MediaPipe Pose Landmarker (incluido en el proyecto)
# ─────────────────────────────────────────────────────────────────────────────
POSE_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
_exe_name        = "procesar.exe" if os.name == "nt" else "procesar"
PROCESAR_EXE     = os.path.join(os.path.dirname(__file__), _exe_name)

# Índices de landmarks de MediaPipe Pose:
#   11 = hombro izq, 12 = hombro der
#   23 = cadera izq, 24 = cadera der
#   25 = rodilla izq, 26 = rodilla der
_LM_HOMBRO_IZQ  = 11
_LM_HOMBRO_DER  = 12
_LM_CADERA_IZQ  = 23
_LM_CADERA_DER  = 24
_LM_RODILLA_IZQ = 25
_LM_RODILLA_DER = 26


def _clasificar_tipo(ratio_ch: float, ratio_cah: float) -> str:
    """
    Clasifica el tipo de cuerpo a partir de dos ratios:
      ratio_ch  = cintura / hombros   (< 1 → cintura más estrecha que hombros)
      ratio_cah = cadera  / hombros   (< 1 → cadera más estrecha, > 1 → más ancha)
      ratio_cc  = cintura / cadera    (derivado; < 1 → cintura más estrecha que cadera)

    Las medidas vienen de la silueta fotográfica, donde "hombros" incluye el ancho
    de los brazos al costado → los ratios son más altos que en medidas con cinta.
    Los umbrales se ajustan a esa realidad.

    Tipos:
      triangulo_invertido  hombros claramente más anchos que cadera   ratio_cah < 0.86
      triangulo            cadera  claramente más ancha  que hombros  ratio_cah > 1.14
      reloj_de_arena       cintura estrecha vs hombros Y vs cadera    ratio_ch < 0.93
                                                                     y ratio_cc < 0.90
      ovalo                cintura amplia, poca definición de talle   ratio_ch > 0.93
      rectangulo           proporciones uniformes (caso por defecto)
    """
    # cintura / cadera — clave para distinguir reloj de arena de rectángulo
    ratio_cc = ratio_ch / ratio_cah if ratio_cah > 0 else ratio_ch

    # ── Diferencias marcadas entre hombros y cadera ───────────────────────────
    if ratio_cah < 0.86:
        return "triangulo_invertido"
    if ratio_cah > 1.14:
        return "triangulo"

    # ── Hombros y cadera similares (0.86 ≤ ratio_cah ≤ 1.14) ─────────────────
    # Reloj de arena: cintura estrecha respecto a AMBAS referencias.
    # Condición dual: evita que figuras con cintura moderada queden aquí.
    if ratio_ch < 0.93 and ratio_cc < 0.90:
        return "reloj_de_arena"

    # Óvalo: cintura casi tan ancha como los hombros
    if ratio_ch > 0.93:
        return "ovalo"

    # Rectángulo: proporciones uniformes
    return "rectangulo"


def _clasificar_por_medidas(busto: float, cintura: float, cadera_alta: float, cadera: float) -> str:
    """
    Clasifica el tipo de cuerpo a partir de las 4 medidas estándar de moda:
    busto, cintura, cadera alta y cadera (todas en cm).
    """
    ref = max(busto, cadera)
    dif = busto - cadera  # positivo → busto > cadera

    if dif > ref * 0.07:
        return "triangulo_invertido"
    if -dif > ref * 0.07:
        return "triangulo"
    # Busto y cadera similares (dentro del 7%)
    if cintura >= busto * 0.87:
        return "ovalo"
    if cintura <= busto * 0.76 and abs(dif) <= ref * 0.04:
        return "reloj_de_arena"
    return "rectangulo"


def _silueta_en_banda(pil_img: "Image.Image", W: int, H: int,
                      y_desde: int, y_hasta: int,
                      cx_ref: int, margen_cx: int) -> list[tuple[int, int, int]]:
    """
    Devuelve una lista de (y_absoluta, x_izq, x_der) con el ancho de silueta
    por fila dentro del rango [y_desde, y_hasta].

    Estrategia:
      - Gaussian blur suave para eliminar textura de ropa.
      - Sobel-X para encontrar bordes laterales del cuerpo.
      - Solo se consideran bordes dentro del rango horizontal
        [cx_ref - margen_cx, cx_ref + margen_cx] para ignorar brazos
        separados o ruido de fondo.
      - Por fila se toman los bordes MÁS externos dentro de esa ventana
        (no los más internos), porque queremos el ancho real del cuerpo.
    """
    from PIL import ImageFilter

    y_desde = max(0, y_desde)
    y_hasta = min(H - 1, y_hasta)
    if y_hasta <= y_desde:
        return []

    # Recortar banda horizontal centrada en el cuerpo
    x_izq_crop = max(0, cx_ref - margen_cx)
    x_der_crop  = min(W, cx_ref + margen_cx)
    banda = pil_img.crop((x_izq_crop, y_desde, x_der_crop, y_hasta + 1))
    Wb, Hb = banda.size

    blurred = banda.filter(ImageFilter.GaussianBlur(radius=4))
    gray    = blurred.convert("L")
    edge    = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=(-1, 0, 1, -2, 0, 2, -1, 0, 1),
        scale=4, offset=128
    ))
    epx = list(edge.getdata())

    mags = sorted(abs(v - 128) for v in epx)
    th   = max(6, mags[int(len(mags) * 0.78)])

    resultados = []
    for dy in range(Hb):
        base = dy * Wb
        bs = [x for x in range(Wb) if abs(epx[base + x] - 128) >= th]
        if len(bs) < 2:
            continue
        span = bs[-1] - bs[0]
        if span < 4:
            continue
        # Coordenadas absolutas en la imagen escalada
        y_abs  = y_desde + dy
        x_izq  = x_izq_crop + bs[0]
        x_der  = x_izq_crop + bs[-1]
        resultados.append((y_abs, x_izq, x_der))

    return resultados


def _medir_zona(filas_silueta: list[tuple[int, int, int]],
                y_desde: int, y_hasta: int,
                buscar_minimo: bool = False) -> tuple[int, int, int, int]:
    """
    Dado un perfil de silueta, encuentra en el rango [y_desde, y_hasta] la fila
    con el ancho máximo (hombros/cadera) o mínimo (cintura).

    Devuelve (fila, x_izq, x_der, ancho). Si no hay datos, devuelve (-1,0,0,0).
    """
    candidatas = [(y, xi, xd, xd - xi) for y, xi, xd in filas_silueta
                  if y_desde <= y <= y_hasta and xd - xi > 0]
    if not candidatas:
        return -1, 0, 0, 0

    if buscar_minimo:
        mejor = min(candidatas, key=lambda t: t[3])
    else:
        mejor = max(candidatas, key=lambda t: t[3])

    return mejor[0], mejor[1], mejor[2], mejor[3]


def analizar_cuerpo_con_pose(ruta_imagen: str) -> dict | None:
    """
    Motor PRIMARIO de análisis de tipo de cuerpo.

    Usa MediaPipe Pose Landmarker para obtener la posición anatómica exacta
    de hombros y cadera. A partir de esas dos anclas, divide el torso en
    zonas garantizando el orden correcto: hombros → cintura → cadera.

    Para cada zona mide el ancho REAL de la silueta (no solo la distancia
    entre landmarks articulares) usando un perfil Sobel-X acotado lateralmente
    al cuerpo (excluye brazos separados y ruido de fondo).

    Garantías estructurales:
      • fila_hombros  < fila_cintura  < fila_cadera  (siempre)
      • ancho_cintura ≤ min(ancho_hombros, ancho_cadera) (siempre)
      • Las coordenadas izq/der de cada zona coinciden con la silueta real.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import numpy as np
    except ImportError:
        print("[Pose] MediaPipe no instalado — usando fallback Sobel")
        return None

    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[Pose] Modelo no encontrado en {POSE_MODEL_PATH}")
        return None

    # ── 1. Cargar y escalar imagen ────────────────────────────────────────────
    pil_img = Image.open(ruta_imagen).convert("RGB")
    W0, H0  = pil_img.size

    MAX_DIM = 800
    scale   = min(1.0, MAX_DIM / max(W0, H0))
    if scale < 1.0:
        W = int(W0 * scale); H = int(H0 * scale)
        pil_img = pil_img.resize((W, H), Image.LANCZOS)
    else:
        W, H = W0, H0

    img_np = np.array(pil_img, dtype=np.uint8)

    # ── 2. Inferencia MediaPipe Pose ──────────────────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    detector  = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result    = detector.detect(mp_image)
    detector.close()

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        print("[Pose] No se detectó pose en la imagen")
        return None

    lm = result.pose_landmarks[0]

    # ── 3. Extraer landmarks y validar visibilidad ────────────────────────────
    puntos_clave = [_LM_HOMBRO_IZQ, _LM_HOMBRO_DER, _LM_CADERA_IZQ, _LM_CADERA_DER]
    for idx in puntos_clave:
        vis = getattr(lm[idx], "visibility", 1.0)
        if vis < 0.35:
            print(f"[Pose] Landmark {idx} con visibilidad baja ({vis:.2f})")
            return None

    def px(idx):
        return int(lm[idx].x * W), int(lm[idx].y * H)

    h_izq_x, h_izq_y  = px(_LM_HOMBRO_IZQ)
    h_der_x, h_der_y  = px(_LM_HOMBRO_DER)
    c_izq_x, c_izq_y  = px(_LM_CADERA_IZQ)
    c_der_x, c_der_y  = px(_LM_CADERA_DER)

    # ── 4. Filas ancla (posición vertical de cada zona) ───────────────────────
    # Estas filas son la referencia anatómica; el orden hombros < cintura < cadera
    # se garantiza más abajo al definir las bandas de búsqueda.
    fila_lm_hombros = (h_izq_y + h_der_y) // 2
    fila_lm_cadera  = (c_izq_y + c_der_y) // 2

    # Sanity: la cadera DEBE estar debajo de los hombros
    if fila_lm_cadera <= fila_lm_hombros + 10:
        print(f"[Pose] Landmarks inconsistentes: hombros y={fila_lm_hombros}, "
              f"cadera y={fila_lm_cadera}")
        return None

    segmento = fila_lm_cadera - fila_lm_hombros   # altura del torso

    # ── 5. Centro horizontal del cuerpo ──────────────────────────────────────
    # Se usa el centroide de los 4 landmarks para centrar las ventanas de Sobel.
    cx_cuerpo = (h_izq_x + h_der_x + c_izq_x + c_der_x) // 4
    # Margen lateral: 60 % del ancho estimado entre landmarks más un margen extra
    # para capturar la silueta exterior del cuerpo (que sobresale del landmark).
    ancho_lm_max = max(abs(h_der_x - h_izq_x), abs(c_der_x - c_izq_x))
    margen_cx    = int(ancho_lm_max * 0.75)   # 75 % a cada lado del centro

    # ── 6. Perfil de silueta completo del torso ───────────────────────────────
    # Calculamos el perfil UNA SOLA VEZ sobre toda la zona del torso con margen.
    y_torso_ini = max(0,   fila_lm_hombros - int(segmento * 0.10))
    y_torso_fin = min(H-1, fila_lm_cadera  + int(segmento * 0.05))

    perfil = _silueta_en_banda(pil_img, W, H,
                               y_torso_ini, y_torso_fin,
                               cx_cuerpo, margen_cx)

    if len(perfil) < 10:
        print(f"[Pose] Perfil de silueta insuficiente ({len(perfil)} filas)")
        return None

    # ── 7. Zonas de búsqueda (con orden garantizado) ─────────────────────────
    #
    #  Hombros: zona superior del torso [lm_hombros ± 8 %]
    #           → buscamos el MÁXIMO ancho (hombro más ancho de la silueta)
    #
    #  Cintura: zona media [35 % a 65 % del segmento, desde fila_hombros]
    #           → buscamos el MÍNIMO ancho
    #           → la banda NO llega nunca a fila_cadera, garantizando orden
    #
    #  Cadera:  zona inferior [70 % a 100 % del segmento, desde fila_hombros]
    #           → buscamos el MÁXIMO ancho
    #           → comienza siempre después del límite superior de la cintura

    margen_h = int(segmento * 0.06)
    y_h_ini  = fila_lm_hombros - margen_h
    y_h_fin  = fila_lm_hombros + int(segmento * 0.22)

    # Cintura: empieza en 42 % del torso (evita el pecho, que ocupa 0-38 %)
    y_c_ini  = fila_lm_hombros + int(segmento * 0.42)
    y_c_fin  = fila_lm_hombros + int(segmento * 0.70)

    y_ca_ini = fila_lm_hombros + int(segmento * 0.72)
    y_ca_fin = fila_lm_cadera  + int(segmento * 0.08)

    fila_h,  izq_h,  der_h,  ancho_h  = _medir_zona(perfil, y_h_ini,  y_h_fin,  buscar_minimo=False)
    fila_c,  izq_c,  der_c,  ancho_c  = _medir_zona(perfil, y_c_ini,  y_c_fin,  buscar_minimo=True)
    fila_ca, izq_ca, der_ca, ancho_ca = _medir_zona(perfil, y_ca_ini, y_ca_fin, buscar_minimo=False)

    # ── 8. Fallbacks si alguna zona no tuvo silueta ───────────────────────────
    if fila_h < 0:
        # Usar landmarks directamente como último recurso
        fila_h  = fila_lm_hombros
        izq_h   = min(h_izq_x, h_der_x)
        der_h   = max(h_izq_x, h_der_x)
        ancho_h = max(1, der_h - izq_h)

    if fila_ca < 0:
        fila_ca  = fila_lm_cadera
        izq_ca   = min(c_izq_x, c_der_x)
        der_ca   = max(c_izq_x, c_der_x)
        ancho_ca = max(1, der_ca - izq_ca)

    if fila_c < 0 or ancho_c <= 0:
        # Cintura estimada como interpolación entre hombros y cadera
        t        = 0.50   # 50 % del camino entre hombros y cadera
        fila_c   = fila_h + int((fila_ca - fila_h) * t)
        ancho_c  = int((ancho_h + ancho_ca) / 2 * 0.82)
        cx_m     = (izq_h + der_h + izq_ca + der_ca) // 4
        izq_c    = cx_m - ancho_c // 2
        der_c    = izq_c + ancho_c

    # ── 9. Aplicar restricción de orden fila_h < fila_c < fila_ca ────────────
    # Si alguna medición de silueta movió una fila fuera de su rango anatómico,
    # la corregimos interpolando.
    if fila_c <= fila_h:
        fila_c = fila_h + max(1, (fila_ca - fila_h) // 2)
    if fila_c >= fila_ca:
        fila_c = fila_ca - max(1, (fila_ca - fila_h) // 4)

    # ── 10. Restricción de ancho: cintura ≤ min(hombros, cadera) ─────────────
    tope_cintura = min(ancho_h, ancho_ca)
    if ancho_c > tope_cintura:
        ancho_c = int(tope_cintura * 0.90)
        cx_c    = (izq_c + der_c) // 2
        izq_c   = cx_c - ancho_c // 2
        der_c   = izq_c + ancho_c

    ancho_h  = max(1, ancho_h)
    ancho_ca = max(1, ancho_ca)
    ancho_c  = max(1, ancho_c)

    # ── 11. Ratios y clasificación ────────────────────────────────────────────
    ratio_ch  = round(ancho_c  / ancho_h, 3)
    ratio_cah = round(ancho_ca / ancho_h, 3)
    tipo      = _clasificar_tipo(ratio_ch, ratio_cah)

    print(f"[Pose] {tipo} | h={ancho_h} c={ancho_c} ca={ancho_ca} "
          f"| ch={ratio_ch} cah={ratio_cah} "
          f"| filas h={fila_h} c={fila_c} ca={fila_ca}")

    # ── 12. Escalar de vuelta a la imagen original ────────────────────────────
    inv = (1.0 / scale) if scale < 1.0 else 1.0
    sc  = lambda v: int(v * inv)

    return {
        "tipo":      tipo,
        "hombros":   sc(ancho_h),
        "cintura":   sc(ancho_c),
        "cadera":    sc(ancho_ca),
        "ratio_ch":  ratio_ch,
        "ratio_cah": ratio_cah,
        "fila_h":  sc(fila_h),  "izq_h":  sc(izq_h),  "der_h":  sc(der_h),
        "fila_c":  sc(fila_c),  "izq_c":  sc(izq_c),  "der_c":  sc(der_c),
        "fila_ca": sc(fila_ca), "izq_ca": sc(izq_ca), "der_ca": sc(der_ca),
        "metodo":  "mediapipe_pose",
    }


def analizar_cuerpo_por_pixeles(ruta_imagen: str) -> dict | None:
    """
    Motor SECUNDARIO (fallback) de análisis de tipo de cuerpo.

    Detecta la silueta del cuerpo usando un kernel Sobel-X sobre una versión
    muy blureada de la imagen (el blur elimina texturas de ropa; Sobel detecta
    dónde cambia la intensidad horizontalmente = bordes laterales del cuerpo).

    Se activa automáticamente cuando MediaPipe no está disponible o no detecta
    pose (fondo complejo, imagen parcial, baja resolución, etc.).

    Por fila:
      • Hombros / cadera → posiciones de borde más EXTERNAS (span total).
      • Cintura          → si hay 4+ bordes, el par más INTERNO descarta
                           los codos/brazos externos y mide solo el torso.
    """
    from PIL import ImageFilter

    img = Image.open(ruta_imagen).convert("RGB")
    W0, H0 = img.size

    MAX = 600
    scale = min(1.0, MAX / max(W0, H0))
    if scale < 1.0:
        W, H = int(W0 * scale), int(H0 * scale)
        img = img.resize((W, H), Image.LANCZOS)
    else:
        W, H = W0, H0

    # ── 1. Blur fuerte — elimina texturas de ropa, preserva silueta ──────────
    blurred = img.filter(ImageFilter.GaussianBlur(radius=6))
    gray    = blurred.convert("L")

    # ── 2. Kernel Sobel-X 3×3 ────────────────────────────────────────────────
    edge_img = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=(-1, 0, 1,
                -2, 0, 2,
                -1, 0, 1),
        scale=4,
        offset=128
    ))
    epx = list(edge_img.getdata())

    # ── 3. Umbral adaptativo: percentil 82 de las magnitudes de borde ─────────
    mags    = sorted(abs(v - 128) for v in epx)
    edge_th = max(10, mags[int(len(mags) * 0.82)])

    # ── 4. Por fila: encontrar posiciones de borde y medir silueta ────────────
    def bordes_fila(y):
        base = y * W
        return [x for x in range(W) if abs(epx[base + x] - 128) >= edge_th]

    def medir_fila(y, cintura=False):
        bs = bordes_fila(y)
        if len(bs) < 2:
            return None, None
        izq, der = bs[0], bs[-1]
        if der - izq < W * 0.04:
            return None, None
        if cintura and len(bs) >= 4:
            izq_int = bs[1]
            der_int = bs[-2]
            if (der_int > izq_int
                    and (der_int - izq_int) < (der - izq) * 0.88):
                izq, der = izq_int, der_int
        return izq, der

    # ── 5. Bounding box del cuerpo ────────────────────────────────────────────
    filas = {}
    for y in range(H):
        izq, der = medir_fila(y)
        if izq is not None:
            filas[y] = (izq, der)

    if len(filas) < 30:
        return None

    ys    = sorted(filas)
    top_y = ys[0]; bot_y = ys[-1]; body_h = bot_y - top_y
    if body_h < 80:
        return None

    def z(p): return top_y + int(body_h * p)

    def zona(ya, yb, cintura=False):
        best = -1 if not cintura else W + 1
        ry   = (ya + yb) // 2
        bi   = bj = W // 2
        for y in range(ya, min(H, yb + 1)):
            izq, der = medir_fila(y, cintura=cintura)
            if izq is None:
                continue
            w = der - izq
            if not cintura and w > best:
                best = w; ry = y; bi = izq; bj = der
            elif cintura and 0 < w < best:
                best = w; ry = y; bi = izq; bj = der
        return ry, bi, bj

    # Hombros: mayor span en la parte superior (saltando ~13 % de cabeza)
    rh, ih, dh = zona(z(0.13), z(0.42))

    # Cadera: mayor span en la parte inferior
    rca, ica, dca = zona(z(0.52), z(0.82))

    # Cintura: menor span ENTRE las filas reales de hombros y cadera (adaptivo).
    # Usar 30-78 % del espacio entre ellos evita que caiga en el pecho o la cadera.
    if rca > rh + 10:
        wa_ini = rh  + max(8, int((rca - rh) * 0.30))
        wa_fin = rca - max(8, int((rca - rh) * 0.22))
    else:
        wa_ini, wa_fin = z(0.42), z(0.65)
    rc, ic, dc = zona(wa_ini, wa_fin, cintura=True)

    ah  = max(1, dh  - ih)
    aca = max(1, dca - ica)
    ac  = min(max(1, dc - ic), int(ah * 0.95), int(aca * 0.95))

    if ah < 10 or aca < 10:
        return None

    ratio_ch  = round(ac  / ah,  3)
    ratio_cah = round(aca / ah, 3)
    tipo      = _clasificar_tipo(ratio_ch, ratio_cah)

    inv = 1.0 / scale if scale < 1.0 else 1.0
    sc  = lambda v: int(v * inv)

    print(f"[Sobel] {tipo} | h={ah} c={ac} ca={aca} "
          f"| ch={ratio_ch} cah={ratio_cah}")

    return {
        "tipo":      tipo,
        "hombros":   sc(ah),  "cintura":   sc(ac),  "cadera":   sc(aca),
        "ratio_ch":  ratio_ch, "ratio_cah": ratio_cah,
        "fila_h":  sc(rh),  "izq_h":  sc(ih),  "der_h":  sc(dh),
        "fila_c":  sc(rc),  "izq_c":  sc(ic),  "der_c":  sc(dc),
        "fila_ca": sc(rca), "izq_ca": sc(ica), "der_ca": sc(dca),
        "metodo":  "sobel_perfil",
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    archivo = request.files["imagen"]
    ruta_temp = os.path.join("static", secure_filename(archivo.filename) or "entrada_temp.bmp")
    archivo.save(ruta_temp)
    imagen = Image.open(ruta_temp).convert("RGB")
    ruta_bmp = os.path.join("static", "entrada.bmp")
    imagen.save(ruta_bmp)
    return {"original": "/static/entrada.bmp"}

@app.route("/procesar", methods=["POST"])
def procesar():
    op = request.form.get("op")
    if not op:
        return {"error": "Operación no especificada"}
    ruta = os.path.join("static", "entrada.bmp")

    if op.startswith("hist"):
        resultado = subprocess.run([PROCESAR_EXE, ruta, op], capture_output=True, text=True)
        datos = json.loads(resultado.stdout)
        return jsonify(datos)

    elif op == "color":
        r = request.form.get("r")
        g = request.form.get("g")
        b = request.form.get("b")
        subprocess.run([PROCESAR_EXE, ruta, "color", r, g, b])
        return {"procesada": "/static/procesada.bmp"}

    elif op == "grises":
        subprocess.run([PROCESAR_EXE, ruta, "grises"])
        return {"procesada": "/static/procesada.bmp"}

    elif op == "guardar_original":
        # Guardar directamente desde entrada.bmp sin usar procesada
        origen = "static/entrada.bmp"
        if not os.path.exists(origen):
            return {"error": "No hay imagen cargada"}

        resultado = subprocess.run(
            [PROCESAR_EXE, origen, "clasificar"],
            capture_output=True, text=True
        )
        categoria = resultado.stdout.strip()
        if categoria not in CATEGORIAS:
            categoria = "otro"

        subprocess.run([PROCESAR_EXE, origen, "segmentar"])

        carpeta = f"static/prendas/{categoria}"
        os.makedirs(carpeta, exist_ok=True)
        nombre = f"{categoria}_{int(time.time()*1000)}_{str(uuid.uuid4())[:6]}.bmp"
        shutil.copy("static/procesada.bmp", os.path.join(carpeta, nombre))

        return {"ok": True, "categoria": categoria}

    elif op == "guardar":
        # Usar procesada si existe, si no la original
        origen = "static/procesada.bmp"
        if not os.path.exists(origen):
            origen = "static/entrada.bmp"

        # Clasificar con la imagen antes de segmentar
        resultado = subprocess.run(
            [PROCESAR_EXE, origen, "clasificar"],
            capture_output=True, text=True
        )
        categoria = resultado.stdout.strip()
        if categoria not in CATEGORIAS:
            categoria = "otro"

        # Segmentar: eliminar fondo, recortar, guardar en procesada.bmp
        subprocess.run([PROCESAR_EXE, origen, "segmentar"])
        origen_seg = "static/procesada.bmp"

        # Guardar la imagen segmentada en subcarpeta de la categoria
        carpeta = f"static/prendas/{categoria}"
        os.makedirs(carpeta, exist_ok=True)
        nombre = f"{categoria}_{int(time.time()*1000)}_{str(uuid.uuid4())[:6]}.bmp"
        shutil.copy(origen_seg, os.path.join(carpeta, nombre))

        return {"ok": True, "categoria": categoria}

    return {"error": "Operación inválida"}

@app.route("/clasificar_preview", methods=["POST"])
def clasificar_preview():
    """Clasifica la imagen actual y devuelve la categoría sugerida sin guardar."""
    origen = "static/procesada.bmp"
    if not os.path.exists(origen):
        origen = "static/entrada.bmp"
    if not os.path.exists(origen):
        return jsonify({"error": "No hay imagen cargada"})
    resultado = subprocess.run(
        [PROCESAR_EXE, origen, "clasificar"],
        capture_output=True, text=True
    )
    categoria = resultado.stdout.strip()
    if categoria not in CATEGORIAS:
        categoria = "otro"
    return jsonify({"categoria": categoria})


@app.route("/guardar_con_meta", methods=["POST"])
def guardar_con_meta():
    """Guarda la prenda con sus metadatos en prendas.json."""
    categoria    = request.form.get("categoria", "otro")
    color        = request.form.get("color", "")
    color_nombre = request.form.get("color_nombre", "")
    estacion     = request.form.get("estacion", "")
    ocasion      = request.form.get("ocasion", "")
    material     = request.form.get("material", "")
    valoracion   = request.form.get("valoracion", "0")
    usar_original = request.form.get("usar_original", "0") == "1"

    if categoria not in CATEGORIAS:
        categoria = "otro"

    origen = "static/entrada.bmp" if usar_original else "static/procesada.bmp"
    if not os.path.exists(origen):
        origen = "static/entrada.bmp"
    if not os.path.exists(origen):
        return jsonify({"error": "No hay imagen cargada"})

    subprocess.run([PROCESAR_EXE, origen, "segmentar"])
    origen_seg = "static/procesada.bmp"

    carpeta = f"static/prendas/{categoria}"
    os.makedirs(carpeta, exist_ok=True)
    nombre  = f"{categoria}_{int(time.time())}.bmp"
    dest    = os.path.join(carpeta, nombre)
    shutil.copy(origen_seg, dest)

    meta_path = "static/prendas/prendas.json"
    try:
        with open(meta_path) as f2:
            meta = json.load(f2)
    except:
        meta = {}

    ruta_key = f"/static/prendas/{categoria}/{nombre}"
    meta[ruta_key] = {
        "categoria":    categoria,
        "color":        color,
        "color_nombre": color_nombre,
        "estacion":     estacion,
        "ocasion":      ocasion,
        "material":     material,
        "valoracion":   int(valoracion) if valoracion.isdigit() else 0,
        "fecha":        int(time.time())
    }
    with open(meta_path, "w") as f2:
        json.dump(meta, f2, ensure_ascii=False, indent=2)

    return jsonify({"ok": True, "categoria": categoria, "ruta": ruta_key})


@app.route("/meta_prendas")
def meta_prendas():
    """Devuelve los metadatos de todas las prendas."""
    try:
        with open("static/prendas/prendas.json") as f2:
            return jsonify(json.load(f2))
    except:
        return jsonify({})


@app.route("/armario")
def armario():
    return render_template("armario.html")

@app.route("/cuerpo")
def cuerpo():
    return render_template("cuerpo.html")

@app.route("/prendas")
def prendas():
    resultado = {}
    for cat in CATEGORIAS:
        carpeta = f"static/prendas/{cat}"
        if os.path.exists(carpeta):
            archivos = [f for f in os.listdir(carpeta) if f.endswith(".bmp")]
            resultado[cat] = [f"/static/prendas/{cat}/{a}" for a in archivos]
        else:
            resultado[cat] = []
    return jsonify(resultado)

@app.route("/recomendar/<tipo_cuerpo>")
def recomendar(tipo_cuerpo):
    reglas = {
        "reloj_de_arena":      {"top": 2, "vestido": 2, "falda": 2, "pantalon": 2, "short": 2, "otro": 1},
        "triangulo":           {"top": 2, "vestido": 1, "falda": 0, "pantalon": 2, "short": 1, "otro": 1},
        "triangulo_invertido": {"top": 0, "vestido": 1, "falda": 2, "pantalon": 2, "short": 1, "otro": 1},
        "ovalo":               {"top": 1, "vestido": 2, "falda": 0, "pantalon": 2, "short": 0, "otro": 1},
        "rectangulo":          {"top": 1, "vestido": 2, "falda": 2, "pantalon": 1, "short": 1, "otro": 1}
    }
    scores = reglas.get(tipo_cuerpo, {cat: 1 for cat in CATEGORIAS})
    resultado = []
    for cat in CATEGORIAS:
        carpeta = f"static/prendas/{cat}"
        score = scores.get(cat, 0)
        if score == 0 or not os.path.exists(carpeta):
            continue
        archivos = [f for f in os.listdir(carpeta) if f.endswith(".bmp")]
        for a in archivos:
            resultado.append({"ruta": f"/static/prendas/{cat}/{a}", "categoria": cat, "score": score})
    resultado.sort(key=lambda x: -x["score"])
    return jsonify(resultado)

@app.route("/seleccionar", methods=["POST"])
def seleccionar():
    ruta = request.form.get("ruta")
    origen = ruta.replace("/static/", "static/")
    shutil.copy(origen, "static/entrada.bmp")
    return {"ok": True}

@app.route("/eliminar", methods=["POST"])
def eliminar():
    ruta = request.form.get("ruta")
    archivo = ruta.replace("/static/", "static/")
    if os.path.exists(archivo):
        os.remove(archivo)
    return {"ok": True}

@app.route("/imagen_actual")
def imagen_actual():
    ruta = "static/entrada.bmp"
    if os.path.exists(ruta):
        return {"existe": True, "ruta": "/static/entrada.bmp"}
    return {"existe": False}

@app.route("/analizar_cuerpo", methods=["POST"])
def analizar_cuerpo():
    archivo = request.files.get("imagen")
    if not archivo:
        return jsonify({"error": "No se recibió imagen"})

    ruta_temp = "static/cuerpo_temp.bmp"
    Image.open(archivo).convert("RGB").save(ruta_temp)

    # ── 1. Motor primario: MediaPipe Pose Landmarker ──────────────────────────
    datos = analizar_cuerpo_con_pose(ruta_temp)
    metodo_dibujo = "python"

    # ── 2. Fallback: análisis por perfil de silueta Sobel ────────────────────
    if datos is None:
        print("[Cuerpo] MediaPipe falló — usando Sobel")
        datos = analizar_cuerpo_por_pixeles(ruta_temp)

    # ── 3. Segundo fallback: motor C++ ────────────────────────────────────────
    if datos is None:
        print("[Cuerpo] Sobel falló — usando C++")
        metodo_dibujo = "cpp"
        res = subprocess.run(
            [PROCESAR_EXE, ruta_temp, "cuerpo"],
            capture_output=True, text=True
        )
        try:
            datos = json.loads(res.stdout.strip())
            datos["metodo"] = "cpp"
        except:
            return jsonify({
                "error": "No se pudo analizar la imagen. "
                         "Usa una foto de cuerpo completo de frente, "
                         "con buena iluminación y fondo claro.",
                "detalle": res.stderr[:200] if res.stderr else ""
            })

    # ── 4. Dibujar líneas de medición sobre la imagen ─────────────────────────
    # El motor C++ genera cuerpo_marcado.bmp directamente.
    # Para MediaPipe y Sobel dibujamos con PIL, re-escaneando píxeles en la
    # imagen original para refinar las coordenadas x (izq/der) de cada zona.
    if metodo_dibujo == "python":
        try:
            from PIL import ImageFilter
            img_orig = Image.open(ruta_temp).convert("RGB")
            W_img, H_img = img_orig.size

            # ── Blur leve para suavizar ruido sin borrar bordes del cuerpo ──
            img_blur = img_orig.filter(ImageFilter.GaussianBlur(radius=3))
            lp = img_blur.load()   # PixelAccess: lp[x, y] = (R, G, B)

            # ── Estimar color de fondo desde las 4 esquinas ─────────────────
            S = max(5, min(12, W_img // 14, H_img // 14))
            bR = bG = bB = n = 0
            for y in range(S):
                for x in range(S):
                    for cy, cx in [(y, x), (y, W_img-1-x),
                                   (H_img-1-y, x), (H_img-1-y, W_img-1-x)]:
                        r, g, b = lp[cx, cy]
                        bR += r; bG += g; bB += b; n += 1
            bR //= n; bG //= n; bB //= n

            var = 0
            for y in range(S):
                for x in range(S):
                    for cy, cx in [(y, x), (y, W_img-1-x),
                                   (H_img-1-y, x), (H_img-1-y, W_img-1-x)]:
                        r, g, b = lp[cx, cy]
                        var += (abs(r-bR) + abs(g-bG) + abs(b-bB)) ** 2
            umbral_px = max(28, (var / n) ** 0.5 * 2.0 + 14)

            # ── Centro y radio de búsqueda ────────────────────────────────────
            fila_h  = datos.get("fila_h")
            fila_c  = datos.get("fila_c")
            fila_ca = datos.get("fila_ca")

            cx_ref = ((datos.get("izq_h", 0) + datos.get("der_h", W_img)) // 2
                      if fila_h is not None else W_img // 2)
            # Radio amplio: cubre la silueta aunque el análisis haya subestimado
            radio = max(
                int(max(datos.get("hombros", 0), datos.get("cadera", 0)) * 1.20),
                int(W_img * 0.38)
            )
            radio = min(radio, W_img // 2 - 1)

            # ── Utilidad común ────────────────────────────────────────────────
            def es_cuerpo(x, y):
                r, g, b = lp[x, y]
                return abs(r-bR) + abs(g-bG) + abs(b-bB) > umbral_px

            # ── Scan desde afuera hacia adentro (hombros / cadera) ────────────
            # Requiere al menos 2 píxeles de cuerpo contiguos para confirmar
            # el borde (descarta píxeles de ruido aislados).
            def escanear_fila(fila):
                if fila is None or not (0 <= fila < H_img):
                    return None
                x0 = max(0, cx_ref - radio)
                x1 = min(W_img - 1, cx_ref + radio)

                izq = None
                for x in range(x0, x1):
                    if es_cuerpo(x, fila) and es_cuerpo(x + 1, fila):
                        izq = x; break
                if izq is None:  # sin confirmación — aceptar primer píxel
                    for x in range(x0, x1 + 1):
                        if es_cuerpo(x, fila):
                            izq = x; break

                der = None
                for x in range(x1, x0, -1):
                    if es_cuerpo(x, fila) and es_cuerpo(x - 1, fila):
                        der = x; break
                if der is None:
                    for x in range(x1, x0 - 1, -1):
                        if es_cuerpo(x, fila):
                            der = x; break

                if izq is None or der is None or der <= izq:
                    return None
                return izq, der

            # ── Scan desde el centro hacia afuera (cintura) ───────────────────
            # Detecta el borde verdadero del torso: cuando los píxeles
            # vuelven a parecerse al fondo, hemos salido del cuerpo.
            # Esto descarta brazos que sobresalgan en esa fila.
            def escanear_cintura(fila):
                if fila is None or not (0 <= fila < H_img):
                    return None
                x0 = max(0, cx_ref - radio)
                x1 = min(W_img - 1, cx_ref + radio)

                # Si el centro de referencia no es cuerpo, usar scan normal
                if not es_cuerpo(cx_ref, fila):
                    return escanear_fila(fila)

                # Izquierda: avanzar desde el centro mientras sea cuerpo
                izq = x0
                for x in range(cx_ref, x0 - 1, -1):
                    if not es_cuerpo(x, fila):
                        izq = x + 1; break  # borde: último px cuerpo = x+1
                else:
                    izq = x0

                # Derecha: ídem
                der = x1
                for x in range(cx_ref, x1 + 1):
                    if not es_cuerpo(x, fila):
                        der = x - 1; break
                else:
                    der = x1

                if der <= izq or (der - izq) < 4:
                    return escanear_fila(fila)  # fallback
                return izq, der

            # ── Detectar bordes ───────────────────────────────────────────────
            res_h  = escanear_fila(fila_h)
            res_ca = escanear_fila(fila_ca)
            res_c  = escanear_cintura(fila_c)

            # Fallback a valores del análisis si el scan falla
            izq_h,  der_h  = res_h  or (datos.get("izq_h",  0), datos.get("der_h",  W_img))
            izq_ca, der_ca = res_ca or (datos.get("izq_ca", 0), datos.get("der_ca", W_img))
            izq_c,  der_c  = res_c  or (datos.get("izq_c",  0), datos.get("der_c",  W_img))

            # ── Reclasificar con las medidas reales de la silueta ────────────
            # Las medidas del análisis (MediaPipe/Sobel) suelen subestimar los
            # hombros y sobreestimar la cintura. Las del escaneo pixel son más
            # fieles a la silueta visible, por lo que se usan para la clasificación final.
            ah_px  = max(1, int(der_h)  - int(izq_h))
            aca_px = max(1, int(der_ca) - int(izq_ca))
            ac_px  = min(max(1, int(der_c) - int(izq_c)),
                         int(ah_px  * 0.98),
                         int(aca_px * 0.98))

            if ah_px > 10 and aca_px > 10:
                ratio_ch_px  = round(ac_px  / ah_px,  3)
                ratio_cah_px = round(aca_px / ah_px,  3)
                tipo_px = _clasificar_tipo(ratio_ch_px, ratio_cah_px)
                print(f"[PxScan] {tipo_px} | h={ah_px} c={ac_px} ca={aca_px} "
                      f"| ch={ratio_ch_px} cah={ratio_cah_px}")
                datos.update({
                    "tipo":      tipo_px,
                    "hombros":   ah_px,
                    "cintura":   ac_px,
                    "cadera":    aca_px,
                    "ratio_ch":  ratio_ch_px,
                    "ratio_cah": ratio_cah_px,
                    "izq_h":  int(izq_h),  "der_h":  int(der_h),
                    "izq_c":  int(izq_c),  "der_c":  int(der_c),
                    "izq_ca": int(izq_ca), "der_ca": int(der_ca),
                })

            # ── Dibujar líneas ────────────────────────────────────────────────
            draw   = ImageDraw.Draw(img_orig)
            grosor = max(2, W_img // 220)

            def linea(fila, izq, der, color):
                if fila is None or not (0 <= fila < H_img):
                    return
                izq = max(0, min(int(izq), W_img - 1))
                der = max(0, min(int(der), W_img - 1))
                if der <= izq:
                    return
                draw.rectangle([izq, fila - grosor // 2,
                                 der, fila + grosor // 2], fill=color)
                tick = max(5, grosor * 2)
                for dx in range(max(2, grosor)):
                    draw.rectangle([izq+dx, fila-tick, izq+dx+1, fila+tick], fill=color)
                    draw.rectangle([der-dx, fila-tick, der-dx+1, fila+tick], fill=color)

            linea(fila_h,  izq_h,  der_h,  (93,  202, 165))
            linea(fila_c,  izq_c,  der_c,  (167, 139, 192))
            linea(fila_ca, izq_ca, der_ca, (232, 168, 124))

            img_orig.save("static/cuerpo_marcado.bmp")
        except Exception as e:
            print(f"[Cuerpo] Error dibujando líneas: {e}")

    datos["marcada"] = f"/static/cuerpo_marcado.bmp?t={int(time.time())}"
    return jsonify(datos)

@app.route("/analizar_colorimetria", methods=["POST"])
def analizar_colorimetria():
    archivo = request.files.get("imagen")
    if not archivo:
        return jsonify({"error": "No se recibió imagen"})
    ruta_temp = "static/colorimetria_temp.bmp"
    Image.open(archivo).convert("RGB").save(ruta_temp)
    resultado = subprocess.run(
        [PROCESAR_EXE, ruta_temp, "colorimetria"],
        capture_output=True, text=True
    )
    try:
        return jsonify(json.loads(resultado.stdout.strip()))
    except:
        return jsonify({"error": "No se pudo analizar", "stdout": resultado.stdout})

@app.route("/clasificar_medidas", methods=["POST"])
def clasificar_medidas():
    try:
        busto   = float(request.form.get("busto",       0))
        cintura = float(request.form.get("cintura",     0))
        c_alta  = float(request.form.get("cadera_alta", 0))
        cadera  = float(request.form.get("cadera",      0))
        unidad  = request.form.get("unidad", "cm")
    except ValueError:
        return jsonify({"error": "Medidas inválidas"})

    if unidad == "in":
        busto   *= 2.54
        cintura *= 2.54
        c_alta  *= 2.54
        cadera  *= 2.54

    if busto <= 0 or cintura <= 0 or cadera <= 0:
        return jsonify({"error": "Por favor ingresa busto, cintura y cadera"})

    tipo = _clasificar_por_medidas(busto, cintura, c_alta, cadera)
    return jsonify({
        "tipo":    tipo,
        "busto":   round(busto, 1),
        "cintura": round(cintura, 1),
        "c_alta":  round(c_alta, 1) if c_alta else None,
        "cadera":  round(cadera, 1),
        "metodo":  "medidas_manuales"
    })


@app.route("/detectar_prendas_foto", methods=["POST"])
def detectar_prendas_foto():
    archivo = request.files.get("imagen")
    if not archivo:
        return jsonify({"error": "No se recibió imagen"})
    ruta_temp = "static/foto_temp.bmp"
    Image.open(archivo).convert("RGB").save(ruta_temp)
    resultado = subprocess.run(
        [PROCESAR_EXE, ruta_temp, "detectar_prendas"],
        capture_output=True, text=True
    )
    try:
        datos = json.loads(resultado.stdout.strip())
        return jsonify(datos)
    except:
        return jsonify({"error": "No se pudo procesar",
                        "stdout": resultado.stdout,
                        "stderr": resultado.stderr})

@app.route("/guardar_prenda_detectada", methods=["POST"])
def guardar_prenda_detectada():
    ruta_origen = request.form.get("ruta", "")
    categoria   = request.form.get("categoria", "")

    # Limpiar la ruta
    ruta_origen = ruta_origen.split("?")[0]  # quitar ?t=timestamp
    ruta_origen = ruta_origen.lstrip("/")    # quitar slash inicial
    ruta_origen = os.path.normpath(ruta_origen)

    if not ruta_origen.startswith("static"):
        return jsonify({"error": "Ruta inválida"})
    if categoria not in CATEGORIAS:
        categoria = "otro"
    if not os.path.exists(ruta_origen):
        return jsonify({"error": f"Archivo no encontrado: {ruta_origen}"})

    carpeta = f"static/prendas/{categoria}"
    os.makedirs(carpeta, exist_ok=True)
    nombre = f"{categoria}_{int(time.time()*1000)}_{str(uuid.uuid4())[:6]}.bmp"
    shutil.copy(ruta_origen, os.path.join(carpeta, nombre))
    return jsonify({"ok": True, "categoria": categoria,
                    "ruta": f"/static/prendas/{categoria}/{nombre}"})

@app.route("/colorimetria")
def colorimetria():
    return render_template("colorimetria.html")


@app.route("/actualizar_meta", methods=["POST"])
def actualizar_meta():
    ruta         = request.form.get("ruta", "")
    estacion     = request.form.get("estacion", "")
    ocasion      = request.form.get("ocasion", "")
    material     = request.form.get("material", "")
    valoracion   = request.form.get("valoracion", "0")
    color        = request.form.get("color", "")
    color_nombre = request.form.get("color_nombre", "")

    meta_path = "static/prendas/prendas.json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    if ruta not in meta:
        return jsonify({"error": "Prenda no encontrada"})

    meta[ruta].update({
        "estacion":     estacion,
        "ocasion":      ocasion,
        "material":     material,
        "valoracion":   int(valoracion) if str(valoracion).isdigit() else 0,
        "color":        color,
        "color_nombre": color_nombre,
    })

    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True})


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(port=5000, debug=True)