"""Punto de entrada para Vercel, que busca una variable ``app`` en este archivo.

En el equipo local se usa ``postura web``; este archivo no hace falta.
"""

import os
import sys
from pathlib import Path

# En Vercel solo /tmp admite escritura; matplotlib guarda ahí su caché.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# Usa el código del repositorio (src/), con su config/default.yaml al lado.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from postura.web import crear_app  # noqa: E402

app = crear_app()
