# Visión de polinizadores

Simula con la webcam cómo ven el mundo siete polinizadores: abeja, abejorro,
mariposa, sírfido, polilla esfinge, colibrí y murciélago nectarívoro. Cada uno
tiene su propia biología ocular (colores que percibe, omatidios, campo visual,
velocidad de visión y visión nocturna) y una narración en voz alta.

## Sitio

- `index.html` y `boceto.html`: página del proyecto *Guías de polinización.
  Fotobordado y softwares en vivo* (Dannia Aguilar), según el diseño original.
  Imágenes y estilos en `assets/`.
- `web/`: el software en vivo.

## Versión web (cualquier computador)

`web/index.html`: se abre directamente en Chrome, Edge o Firefox, sin instalar
nada. Permite usar la cámara, un video o foto, o una escena de demostración.

Las narraciones están grabadas en `web/audio/` (voz Sabina de Windows) para que
suenen igual en todos los dispositivos. Si cambias un texto de narración en
`web/index.html`, vuelve a generar los audios (Windows + ffmpeg):

```
python tools/generar_audios.py
```

## Versión Python (Windows)

`vision_abeja.py`: requiere Python con `opencv-python`, `numpy` y `Pillow`.

```
python vision_abeja.py
```
