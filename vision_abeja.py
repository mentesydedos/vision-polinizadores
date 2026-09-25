"""
Visión de polinizadores con la webcam
=====================================
Muestra lado a lado la imagen original de la webcam ("Humano") y una
simulación de cómo vería el mundo un animal polinizador. Cada animal tiene
su propio perfil de biología ocular:

  Abeja        Ojos compuestos (~5.500 omatidios). Tricrómata UV-azul-verde,
               no ve el rojo. Percibe ~200 imágenes/s.
  Abejorro     Como la abeja, pero con ojos más grandes (más facetas, algo más
               nítido) y activo con menos luz.
  Mariposa     Ojos compuestos con ~12.000 omatidios. Algunas especies (Papilio)
               tienen 5-6 tipos de fotorreceptores: ven UV Y también el rojo.
  Sírfido      Mosca de las flores. Ojos enormes, UV-azul-verde, y la visión de
               movimiento más rápida de la lista (~300 imágenes/s).
  Polilla      Polilla esfinge nocturna. Ojos de superposición: juntan la luz de
               muchas facetas, ven en color de noche pero con poca nitidez.
  Colibrí      Ojo tipo cámara (como el nuestro, sin facetas) y muy nítido.
               Tetracrómata: rojo, verde, azul y UV; ve colores "UV+verde"
               que nosotros no podemos imaginar.
  Murciélago   Murciélago nectarívoro. Visión nocturna casi monocromática,
               sensible al UV, borrosa. (Para orientarse usa sobre todo
               ecolocalización y olfato.)

Filtros (se configuran solos al elegir animal, pero se pueden cambiar):
  1. Color      - Falso color según los fotorreceptores del animal. La webcam
                  no capta UV: se estima a partir de los colores.
  2. Ojo de pez - Campo visual muy amplio: centro ampliado, bordes comprimidos.
  3. Omatidios  - Cuadrícula hexagonal de facetas (solo en ojos compuestos).
  4. Movimiento - Estelas que muestran la velocidad de su visión.
  5. Viñeta     - El borde del campo visual se oscurece.

Al elegir un animal se narra en voz alta cómo ve (voz en español de Windows).
Mientras dura la narración, la visión del animal ocupa casi toda la ventana y
la imagen real queda en un recuadro; al terminar vuelve a la vista 50% / 50%.

Botones en la parte inferior de la ventana, o teclas:
  F1-F7 / a  cambiar de animal (a = siguiente)
  1-5        activar/desactivar cada filtro
  + / -      facetas más grandes / más pequeñas
  b          bordes de las facetas
  i          mostrar/ocultar la ficha de biología
  n          activar/desactivar la narración en voz alta
  s          guardar una captura
  q / ESC    salir
"""

import base64
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # sin Pillow se usa la fuente de OpenCV (sin acentos)
    Image = None

CAM_INDEX = 0
WIDTH, HEIGHT = 640, 480


# ---------------------------------------------------------------- perfiles

# La webcam solo capta azul, verde y rojo. El UV se estima: las zonas azuladas,
# blancas y el cielo reflejan mucho UV; las rojas y oscuras, poco.
#   uv ≈ 1.315·B + 0.09·G − 0.255·R
UV = np.array([1.315, 0.09, -0.255], np.float32)
B = np.array([1.0, 0.0, 0.0], np.float32)
G = np.array([0.0, 1.0, 0.0], np.float32)
R = np.array([0.0, 0.0, 1.0], np.float32)


def color_matrix(out_b, out_g, out_r):
    """Matriz 3x3 para cv2.transform: cada fila es un canal de salida (B, G, R)
    como combinación de los canales de entrada (B, G, R)."""
    return np.array([out_b, out_g, out_r], np.float32)


@dataclass
class Animal:
    name: str
    info: tuple            # dos líneas de ficha de biología
    narration: str         # texto que se lee en voz alta al elegir el animal
    matrix: np.ndarray     # falso color
    compound: bool         # ojo compuesto (omatidios) o tipo cámara
    hex_size: int          # tamaño de faceta en píxeles (menos = más omatidios)
    fisheye: float         # 0 = sin distorsión
    blur: float            # pérdida de nitidez (agudeza visual)
    motion_decay: float    # duración de las estelas (más = visión más rápida)
    motion_thresh: int     # sensibilidad al movimiento (menos = más sensible)
    night_gain: float = 1.0  # >1 = ojo adaptado a poca luz (más brillo y ruido)
    motion_on: bool = True


ANIMALS = [
    Animal(
        "Abeja",
        ("Ve UV, azul y verde; el rojo le parece negro.",
         "~5.500 omatidios por ojo · ~200 imágenes por segundo."),
        (
            "Abeja melífera. Sus ojos compuestos tienen unas cinco mil quinientas facetas, "
            "llamadas omatidios, y cada una capta un solo punto de color, por eso ve el mundo "
            "como un mosaico. Ve el ultravioleta, el azul y el verde, pero no el rojo: para "
            "ella, el rojo es casi negro. Muchas flores tienen dibujos ultravioleta, invisibles "
            "para nosotros, que le señalan dónde está el néctar. Además percibe unas doscientas "
            "imágenes por segundo, así que nota el movimiento mucho más rápido que nosotros."
        ),
        color_matrix(UV, 1.1 * B, 0.9 * G + 0.1 * R),
        compound=True, hex_size=9, fisheye=0.45, blur=2.5,
        motion_decay=0.85, motion_thresh=18,
    ),
    Animal(
        "Abejorro",
        ("Ve UV, azul y verde, como la abeja; ojos más grandes.",
         "Algo más nítido y activo con menos luz (días nublados)."),
        (
            "Abejorro. Ve los colores como la abeja: ultravioleta, azul y verde, sin el rojo. "
            "Sus ojos son más grandes y tienen más facetas, así que ve con algo más de detalle. "
            "Es de los pocos polinizadores que salen con frío, niebla o cielo nublado, y su "
            "vista funciona bien con poca luz. Puede aprender a reconocer las flores por su "
            "forma y por patrones que nosotros no vemos."
        ),
        color_matrix(0.9 * UV + 0.1 * B, 1.05 * B + 0.05 * G, 0.95 * G + 0.1 * R),
        compound=True, hex_size=7, fisheye=0.4, blur=2.0,
        motion_decay=0.83, motion_thresh=18, night_gain=1.3,
    ),
    Animal(
        "Mariposa",
        ("Hasta 6 tipos de fotorreceptores: ve UV y también el rojo.",
         "~12.000 omatidios · colores muy saturados, flores con 'guías' UV."),
        (
            "Mariposa. Tiene alrededor de doce mil omatidios en cada ojo. Algunas especies, "
            "como la mariposa cola de golondrina, tienen seis tipos de fotorreceptores, el "
            "doble que nosotros: ven el ultravioleta y también el rojo. Por eso su mundo tiene "
            "colores muy intensos, y distingue matices que para nosotros son idénticos. Muchas "
            "mariposas tienen dibujos ultravioleta en las alas, que usan para reconocer a su pareja."
        ),
        color_matrix(0.5 * B + 0.55 * UV, 1.15 * G - 0.1 * R + 0.05 * B, 1.25 * R - 0.2 * G),
        compound=True, hex_size=6, fisheye=0.5, blur=1.8,
        motion_decay=0.8, motion_thresh=20,
    ),
    Animal(
        "Sírfido",
        ("Mosca de las flores: ve UV, azul y verde.",
         "Visión de movimiento ultrarrápida (~300 imágenes por segundo)."),
        (
            "Sírfido, la mosca de las flores. Imita a las abejas con sus rayas amarillas y "
            "negras, pero es una mosca. Ve ultravioleta, azul y verde, y sus enormes ojos "
            "cubren casi todo lo que la rodea. Su visión del movimiento es de las más rápidas "
            "del reino animal: capta unas trescientas imágenes por segundo. Eso le permite "
            "quedarse suspendida en el aire y perseguir a otros insectos con gran precisión."
        ),
        color_matrix(UV, 1.05 * B + 0.05 * G, 0.85 * G + 0.15 * R),
        compound=True, hex_size=7, fisheye=0.55, blur=2.2,
        motion_decay=0.92, motion_thresh=10,
    ),
    Animal(
        "Polilla",
        ("Polilla esfinge: ve en color de noche (UV, azul y verde).",
         "Ojos de superposición: capta mucha luz pero ve borroso."),
        (
            "Polilla esfinge. Vuela de noche y, a diferencia de nosotros, ve en color incluso "
            "a la luz de las estrellas. Sus ojos de superposición reúnen la luz de muchas "
            "facetas en cada punto de la imagen: así ve muy brillante, pero borroso y con "
            "grano. Ve ultravioleta, azul y verde, y busca sobre todo flores blancas o "
            "pálidas, que destacan en la oscuridad."
        ),
        color_matrix(UV, 1.1 * B, 0.9 * G + 0.1 * R),
        compound=True, hex_size=12, fisheye=0.45, blur=4.5,
        motion_decay=0.9, motion_thresh=14, night_gain=2.4,
    ),
    Animal(
        "Colibrí",
        ("Ojo tipo cámara, muy nítido. Ve rojo, verde, azul y UV:",
         "distingue colores 'UV+verde' que nosotros no podemos ver."),
        (
            "Colibrí. Sus ojos son de tipo cámara, como los nuestros, sin facetas, y muy "
            "nítidos. Pero tiene cuatro tipos de conos: rojo, verde, azul y ultravioleta. "
            "Gracias a eso percibe colores que para nosotros no existen, como el ultravioleta "
            "mezclado con verde o con rojo. Le atraen especialmente las flores rojas y "
            "tubulares, y su vista le ayuda a calcular distancias mientras flota frente a ellas."
        ),
        color_matrix(B + 0.35 * UV, G + 0.1 * UV, R + 0.25 * UV),
        compound=False, hex_size=9, fisheye=0.15, blur=0.0,
        motion_decay=0.6, motion_thresh=25, motion_on=False,
    ),
    Animal(
        "Murciélago",
        ("Murciélago nectarívoro: visión nocturna casi en blanco y negro,",
         "sensible al UV. Se guía sobre todo por ecolocalización y olfato."),
        (
            "Murciélago nectarívoro. Es un polinizador nocturno muy importante, por ejemplo "
            "del agave. Su vista está adaptada a la oscuridad: ve casi en blanco y negro y "
            "con poca nitidez, pero es sensible al ultravioleta, lo que le ayuda a encontrar "
            "flores que lo reflejan. Para orientarse usa sobre todo la ecolocalización y su "
            "excelente olfato."
        ),
        # gris ≈ 0.4·UV + 0.3·B + 0.3·G, con un tinte azulado
        color_matrix(*(k * (0.4 * UV + 0.3 * B + 0.3 * G) for k in (1.1, 1.0, 0.8))),
        compound=False, hex_size=9, fisheye=0.1, blur=3.0,
        motion_decay=0.7, motion_thresh=22, night_gain=2.8, motion_on=False,
    ),
]


# ---------------------------------------------------------------- mapas fijos

def build_fisheye_maps(w, h, strength=0.45):
    """Mapas de remapeo para una distorsión tipo ojo de pez (centro ampliado)."""
    ys, xs = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    nx, ny = (xs - cx) / cx, (ys - cy) / cy
    r = np.sqrt(nx ** 2 + ny ** 2)
    # En el centro se toma una zona más pequeña de la imagen original (se amplía);
    # hacia el borde el factor tiende a 1.
    factor = (1 - strength) + strength * np.clip(r, 0, 1.5) ** 2
    map_x = (nx * factor * cx + cx).astype(np.float32)
    map_y = (ny * factor * cy + cy).astype(np.float32)
    return map_x, map_y


def build_hex_maps(w, h, size):
    """Para cada píxel calcula el centro de su hexágono (omatidio) y si
    está en el borde de la faceta. Hexágonos 'pointy-top', coordenadas cúbicas."""
    ys, xs = np.indices((h, w), dtype=np.float64)
    sqrt3 = np.sqrt(3.0)
    q = (sqrt3 / 3 * xs - 1.0 / 3 * ys) / size
    r = (2.0 / 3 * ys) / size
    s = -q - r

    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    rs = -rq - rr

    cx = size * sqrt3 * (rq + rr / 2)
    cy = size * 1.5 * rr
    map_x = np.clip(cx, 0, w - 1).astype(np.float32)
    map_y = np.clip(cy, 0, h - 1).astype(np.float32)

    # Distancia hexagonal al centro (0 en el centro, 1 en el borde)
    d = np.maximum.reduce([np.abs(q - rq), np.abs(r - rr), np.abs(s - rs)]) * 2
    # Sombreado suave: cada faceta es un poco más brillante en el centro (efecto lente).
    # Se guarda como uint8 escalado x128 para multiplicar rápido con cv2.multiply.
    shade = np.clip(1.08 - 0.25 * d ** 2, 0, 1.99)
    border = d > (1 - 1.6 / size)
    shade_border = np.where(border, shade * 0.25, shade)
    to_u8 = lambda m: cv2.merge([(m * 128).astype(np.uint8)] * 3)
    return map_x, map_y, to_u8(shade), to_u8(shade_border)


def build_vignette(w, h):
    ys, xs = np.indices((h, w), dtype=np.float32)
    nx, ny = (xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)
    r = np.sqrt(nx ** 2 + ny ** 2)
    v = np.clip(1.25 - 0.65 * r ** 2, 0, 1)
    return cv2.merge([(v * 128).astype(np.uint8)] * 3)


# ---------------------------------------------------------------- filtros

def animal_color(frame, animal):
    return cv2.transform(frame, animal.matrix)


def night_vision(frame, gain):
    """Ojo adaptado a poca luz: sube el brillo de las sombras y añade el
    'grano' propio de captar pocos fotones."""
    lut = np.clip(255 * (np.arange(256) / 255) ** (1 / gain) * 1.05, 0, 255).astype(np.uint8)
    out = cv2.LUT(frame, lut)
    noise = np.empty(frame.shape[:2], np.int16)
    cv2.randn(noise, 0, 4 * gain)
    return cv2.add(out, cv2.merge([noise] * 3), dtype=cv2.CV_8U)


def ommatidia(frame, maps, show_borders, blur):
    map_x, map_y, shade, shade_border = maps
    blurred = cv2.GaussianBlur(frame, (0, 0), max(blur, 0.5))
    out = cv2.remap(blurred, map_x, map_y, cv2.INTER_NEAREST)
    return cv2.multiply(out, shade_border if show_borders else shade, scale=1 / 128)


class MotionDetector:
    """Resalta el movimiento con estelas amarillas que se desvanecen."""

    def __init__(self):
        self.prev = None
        self.trail = None

    def apply(self, frame_raw, frame_out, decay, thresh):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        if self.prev is None:
            self.prev = gray
            self.trail = np.zeros(gray.shape, np.float32)
            return frame_out
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2).astype(np.float32) / 255.0
        self.trail = np.maximum(self.trail * decay, mask)
        alpha = cv2.merge([self.trail * 0.6] * 3)
        glow = np.array([40, 230, 255], np.float32)  # amarillo brillante (BGR)
        out = frame_out.astype(np.float32)
        out += (glow - out) * alpha
        return out.astype(np.uint8)


# ---------------------------------------------------------------- narración

class Narrator:
    """Lee textos en voz alta con la síntesis de voz de Windows (voz en español
    si está instalada). Cada narración corre en un proceso aparte, así el video
    no se congela, y se corta si se pide otra antes de terminar."""

    def __init__(self):
        self.proc = None
        self.enabled = sys.platform == "win32"

    def speak(self, text):
        self.stop()
        if not self.enabled:
            return
        # El texto va en base64 para que los acentos lleguen intactos.
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = (
            f"$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64}'));"
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$v=$s.GetInstalledVoices()|?{$_.VoiceInfo.Culture.Name -like 'es*'}|select -First 1;"
            "if($v){$s.SelectVoice($v.VoiceInfo.Name)};"
            "$s.Speak($t)"
        )
        try:
            self.proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            self.enabled = False

    def speaking(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.speaking():
            self.proc.kill()
        self.proc = None


# ---------------------------------------------------------------- texto

FONT_FILES = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
              "DejaVuSans.ttf", "Arial.ttf"]
FONT_BOLD_FILES = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf",
                   "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]


@lru_cache(maxsize=None)
def _pil_font(size, bold):
    for path in (FONT_BOLD_FILES if bold else FONT_FILES):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=512)
def _text_patch(text, size, color, bold, outline):
    """Renderiza un texto (con acentos) una sola vez y lo guarda en caché
    como imagen BGR + alfa, para pegarlo rápido en cada cuadro."""
    font = _pil_font(size, bold)
    pad = outline + 2
    x0, y0, x1, y1 = font.getbbox(text)
    img = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font,
                             fill=(color[2], color[1], color[0], 255),
                             stroke_width=outline, stroke_fill=(0, 0, 0, 255))
    arr = np.asarray(img)
    bgr = arr[..., 2::-1].astype(np.float32)
    alpha = arr[..., 3:].astype(np.float32) / 255.0
    return bgr, alpha


def text_size(text, size=16, bold=False):
    if Image is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 32, 1)
        return tw, th
    bgr, _ = _text_patch(text, size, (255, 255, 255), bold, 0)
    return bgr.shape[1] - 4, bgr.shape[0] - 4


def put_text(img, text, org, size=16, color=(255, 255, 255), bold=False, outline=2):
    """Escribe texto con la esquina superior izquierda en `org`."""
    x, y = org
    if Image is None:  # respaldo sin acentos
        th = text_size(text, size)[1]
        f, sc = cv2.FONT_HERSHEY_SIMPLEX, size / 32
        if outline:
            cv2.putText(img, text, (x, y + th), f, sc, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y + th), f, sc, color, 1, cv2.LINE_AA)
        return
    bgr, alpha = _text_patch(text, size, tuple(color), bold, outline)
    ph, pw = bgr.shape[:2]
    x, y = x - outline - 2, y - outline - 2
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + pw, img.shape[1]), min(y + ph, img.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    roi = img[y1:y2, x1:x2].astype(np.float32)
    b = bgr[y1 - y:y2 - y, x1 - x:x2 - x]
    a = alpha[y1 - y:y2 - y, x1 - x:x2 - x]
    img[y1:y2, x1:x2] = (roi * (1 - a) + b * a).astype(np.uint8)


# ---------------------------------------------------------------- interfaz

ROW_H = 46
BAR_H = ROW_H * 2 + 8   # dos filas de botones
FILTER_NAMES = ["Color", "Ojo de pez", "Omatidios", "Movimiento", "Viñeta"]

# Fila 1: animales. Fila 2: filtros y acciones.
ROW_ANIMALS = [(f"animal{i}", a.name) for i, a in enumerate(ANIMALS)]
ROW_TOOLS = [(f"filtro{i}", name) for i, name in enumerate(FILTER_NAMES)] + [
    ("bordes", "Bordes"),
    ("menos", "Faceta −"),
    ("mas", "Faceta +"),
    ("info", "Ficha"),
    ("voz", "Narrar"),
    ("captura", "Captura"),
    ("salir", "Salir"),
]


def layout_buttons(width, top):
    """Lista de (accion, texto, (x1, y1, x2, y2)) repartidos en dos filas."""
    gap = 8
    buttons = []
    for row, items in enumerate((ROW_ANIMALS, ROW_TOOLS)):
        bw = (width - gap * (len(items) + 1)) / len(items)
        y1 = top + 8 + row * ROW_H
        for i, (action, text) in enumerate(items):
            x1 = int(gap + i * (bw + gap))
            buttons.append((action, text, (x1, y1, int(x1 + bw), y1 + ROW_H - 8)))
    return buttons


def draw_toolbar(width, state, buttons, hover):
    top = buttons[0][2][1] - 8
    bar = np.full((BAR_H, width, 3), (32, 32, 32), np.uint8)
    for action, text, (x1, y1, x2, y2) in buttons:
        y1, y2 = y1 - top, y2 - top
        if action.startswith("animal"):
            active = int(action[6:]) == state["animal"]
            on_fill = (60, 175, 90)      # verde = animal elegido
        elif action.startswith("filtro"):
            active = state["flags"][int(action[6:])]
            on_fill = (40, 150, 220)     # ámbar = filtro activo
        elif action in ("bordes", "info", "voz"):
            active = state[{"bordes": "borders", "info": "info", "voz": "voice"}[action]]
            on_fill = (40, 150, 220)
        else:
            active = None                # botón de acción, no interruptor

        if action == "salir":
            fill = (60, 60, 170)
        elif active:
            fill = on_fill
        elif active is None:
            fill = (90, 90, 90)
        else:
            fill = (55, 55, 55)
        if hover == action:
            fill = tuple(min(255, c + 35) for c in fill)

        cv2.rectangle(bar, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
        cv2.rectangle(bar, (x1, y1), (x2, y2),
                      (210, 210, 210) if hover == action else (20, 20, 20), 1)
        color = (20, 20, 20) if active else (235, 235, 235)
        bold = action.startswith("animal")
        tw, th = text_size(text, 15, bold)
        put_text(bar, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 - th) // 2 - 1),
                 15, color, bold, outline=0)
    return bar


def draw_info(img, animal):
    """Ficha de biología en la parte inferior de la imagen del animal."""
    h, w = img.shape[:2]
    box_h = 62
    overlay = img[h - box_h:h].astype(np.float32) * 0.35
    img[h - box_h:h] = overlay.astype(np.uint8)
    for i, line in enumerate(animal.info):
        put_text(img, line, (12, h - box_h + 8 + i * 24), 15, (235, 235, 235), outline=0)


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ok, frame = cap.read()
    if not ok:
        print("No se pudo leer de la webcam.")
        return
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    h, w = frame.shape[:2]

    narrator = Narrator()
    # Los mapas dependen del tamaño de la imagen (normal o modo narración),
    # así que se guardan en caché por tamaño.
    cache, motions = {}, {}

    def cached(kind, *args):
        if (kind, *args) not in cache:
            builder = {"fish": build_fisheye_maps, "hex": build_hex_maps,
                       "vig": build_vignette}[kind]
            cache[(kind, *args)] = builder(*args)
        return cache[(kind, *args)]

    def render_animal(src, animal):
        """Aplica los filtros del animal a `src` (de cualquier tamaño)."""
        sh, sw = src.shape[:2]
        flags = state["flags"]
        raw = src
        if flags[1] and animal.fisheye > 0:
            fx, fy = cached("fish", sw, sh, animal.fisheye)
            raw = cv2.remap(raw, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out = raw
        if flags[0]:
            out = animal_color(out, animal)
            if animal.night_gain > 1:
                out = night_vision(out, animal.night_gain)
        if flags[3]:
            motion = motions.setdefault((sw, sh), MotionDetector())
            out = motion.apply(raw, out, animal.motion_decay, animal.motion_thresh)
        if flags[2]:
            out = ommatidia(out, cached("hex", sw, sh, state["hex_size"]), state["borders"],
                            animal.blur)
        elif animal.blur > 0:
            out = cv2.GaussianBlur(out, (0, 0), animal.blur)
        if flags[4]:
            out = cv2.multiply(out, cached("vig", sw, sh), scale=1 / 128)
        return out.copy()

    state = {
        "animal": 0,
        "flags": [True] * 5,
        "borders": True,
        "info": True,
        "voice": narrator.enabled,
        "hex_size": 9,
        "running": True,
        "snapshot": False,
        "hover": None,
        "message": "",
        "message_until": 0.0,
        "big": False,         # modo narración (animal a pantalla casi completa)
        "last_video": None,
        "fade_from": None,
        "fade_start": 0.0,
    }

    def select_animal(i):
        a = ANIMALS[i]
        state["animal"] = i
        state["flags"] = [True, a.fisheye > 0, a.compound, a.motion_on, True]
        state["hex_size"] = a.hex_size
        if state["voice"]:
            narrator.speak(a.narration)

    select_animal(0)
    buttons = layout_buttons(w * 2, h)

    def do(action):
        if action.startswith("animal"):
            select_animal(int(action[6:]))
        elif action == "siguiente":
            select_animal((state["animal"] + 1) % len(ANIMALS))
        elif action.startswith("filtro"):
            i = int(action[6:])
            state["flags"][i] = not state["flags"][i]
        elif action == "bordes":
            state["borders"] = not state["borders"]
        elif action == "info":
            state["info"] = not state["info"]
        elif action == "voz":
            state["voice"] = not state["voice"]
            if state["voice"]:
                narrator.enabled = True
                narrator.speak(ANIMALS[state["animal"]].narration)
            else:
                narrator.stop()
        elif action in ("mas", "menos"):
            step = 1 if action == "mas" else -1
            state["hex_size"] = int(np.clip(state["hex_size"] + step, 3, 40))
        elif action == "captura":
            state["snapshot"] = True
        elif action == "salir":
            state["running"] = False

    def button_at(x, y):
        for action, _, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return action
        return None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            state["hover"] = button_at(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            action = button_at(x, y)
            if action:
                do(action)

    keymap = {ord(str(i + 1)): f"filtro{i}" for i in range(5)}
    keymap.update({ord("b"): "bordes", ord("+"): "mas", ord("="): "mas", ord("-"): "menos",
                   ord("i"): "info", ord("n"): "voz", ord("a"): "siguiente", ord("s"): "captura",
                   ord("q"): "salir", 27: "salir"})
    # Teclas F1-F7 (códigos de waitKeyEx en Windows)
    fkeys = {0x700000 + i: f"animal{i}" for i in range(len(ANIMALS))}

    fps, last = 0.0, time.time()
    window = "Vision de polinizadores"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, w * 2, h + BAR_H)
    cv2.setMouseCallback(window, on_mouse)

    while state["running"]:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(cv2.resize(frame, (w, h)), 1)  # efecto espejo
        animal = ANIMALS[state["animal"]]
        flags = state["flags"]
        big = narrator.speaking()  # durante la narración, el animal ocupa casi todo

        if big:
            # Franja central de la cámara con la proporción de toda el área de video
            # (1280x480). Se procesa a 640x240 y se amplía x2: las facetas se ven
            # del mismo tamaño que en el modo normal y va igual de fluido.
            band = w * h // (2 * w)
            y0 = (h - band) // 2
            view = render_animal(frame[y0:y0 + band], animal)
            view = cv2.resize(view, (w * 2, h), interpolation=cv2.INTER_LINEAR)
        else:
            view = render_animal(frame, animal)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        vw = view.shape[1]

        put_text(view, animal.name, (12, 10), 22, bold=True)
        info = f"faceta {state['hex_size']} px · {fps:.0f} fps" if flags[2] else f"{fps:.0f} fps"
        tw, _ = text_size(info, 14)
        if big:
            put_text(view, "Narrando…", (12, 40), 14, (120, 220, 255))
            put_text(view, info, (12, 62), 14, (220, 220, 220))
        else:
            put_text(view, info, (vw - tw - 12, 14), 14, (220, 220, 220))
        if state["info"]:
            draw_info(view, animal)

        if big:
            # Imagen real en un recuadro, arriba a la derecha
            iw, ih = 240, 180
            x0, y0 = vw - iw - 14, 14
            inset = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
            put_text(inset, "Humano", (8, 6), 16, bold=True)
            cv2.rectangle(view, (x0 - 3, y0 - 3), (x0 + iw + 2, y0 + ih + 2), (235, 235, 235), -1)
            view[y0:y0 + ih, x0:x0 + iw] = inset
            video = view
        else:
            left = frame.copy()
            put_text(left, "Humano", (12, 10), 22, bold=True)
            video = np.hstack([left, view])

        # Transición suave al cambiar de modo
        if big != state["big"]:
            state["big"], state["fade_from"], state["fade_start"] = big, state["last_video"], now
        if state["fade_from"] is not None:
            t = (now - state["fade_start"]) / 0.35
            if t < 1:
                video = cv2.addWeighted(state["fade_from"], 1 - t, video, t, 0)
            else:
                state["fade_from"] = None
        state["last_video"] = video.copy()

        if state["snapshot"]:
            state["snapshot"] = False
            name = time.strftime(f"{animal.name.lower()}_%Y%m%d_%H%M%S.png")
            cv2.imencode(".png", video)[1].tofile(name)  # admite nombres con acentos
            print("Captura guardada:", name)
            state["message"], state["message_until"] = f"Guardada: {name}", now + 2.5
        if now < state["message_until"]:
            put_text(video, state["message"], (12 if state["big"] else w + 12, 84 if state["big"] else 42),
                     14, (120, 255, 120))

        cv2.imshow(window, np.vstack([video, draw_toolbar(w * 2, state, buttons, state["hover"])]))

        key = cv2.waitKeyEx(1)
        if key in fkeys:
            do(fkeys[key])
        elif key != -1 and (key & 0xFF) in keymap:
            do(keymap[key & 0xFF])
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    narrator.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
