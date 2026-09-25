"""
Genera los audios de narración de la versión web (web/audio/*.mp3) con una voz
de Windows, para que la narración suene igual en todos los dispositivos.

Los textos se leen directamente de web/index.html (campo `narration` de cada
animal), así que después de cambiar un texto basta con volver a ejecutar:

    python tools/generar_audios.py            (voz Sabina)
    python tools/generar_audios.py Raul       (otra voz de Windows)

Necesita Windows y ffmpeg (en el PATH o en la variable de entorno FFMPEG).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "web" / "index.html"
OUT = ROOT / "web" / "audio"
FFMPEG_FALLBACK = r"C:\Program Files\BioStar 2(x64)\ve\dist\lib\bsve\dll\ffmpeg.exe"


def slug(name):
    """'Murciélago' -> 'murcielago' (nombre de archivo sin acentos)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else "Sabina"
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or FFMPEG_FALLBACK
    if not Path(ffmpeg).exists():
        sys.exit("No se encontró ffmpeg. Instálalo o indica su ruta en la variable FFMPEG.")

    html = HTML.read_text(encoding="utf-8")
    pairs = re.findall(r'name: "([^"]+)".*?narration: "([^"]+)"', html, re.S)
    if not pairs:
        sys.exit("No se encontraron narraciones en web/index.html")
    texts = {slug(name): text for name, text in pairs}

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "textos.json"
        json_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(Path(__file__).with_name("tts_winrt.ps1")),
                        "-Voice", voice, "-InFile", str(json_path), "-OutDir", tmp], check=True)
        for key in texts:
            mp3 = OUT / f"{key}.mp3"
            # MP3 mono a 48 kbps: suena bien para voz y pesa ~6 KB por segundo.
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(Path(tmp) / f"{key}.wav"),
                            "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "48k", str(mp3)], check=True)
            print(f"{mp3.relative_to(ROOT)}  {mp3.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
