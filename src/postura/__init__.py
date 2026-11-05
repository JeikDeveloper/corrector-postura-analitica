"""Analítica de registros de un dispositivo corrector de postura.

Flujo: texto crudo → eventos (parser) → filtro de archivos y hallazgos (validacion)
→ sesiones y mediciones (sesiones, ingesta) → esquema (modelo) → calidad → figuras.

Advertencia de dominio: los ángulos son relativos a la calibración de su propia
sesión. Dos sesiones no son comparables en términos absolutos.
"""

import logging

__version__ = "0.1.0"

# Biblioteca: sin manejador propio, los avisos de ingesta no se imprimen solos en la
# consola (p. ej. del servidor web). La CLI añade un FileHandler para ingesta.log.
logging.getLogger("postura.ingesta").addHandler(logging.NullHandler())
