"""Paleta, tipografía y tamaños. Todos los valores salen de la sección `estilo` de la configuración."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # sin pantalla: las figuras se exportan a archivo

import matplotlib.pyplot as plt
from matplotlib import colormaps


def parametros(config: dict) -> dict:
    e = config["estilo"]
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [e["fuente"], "DejaVu Sans", "Arial"],
        "font.size": e["tamano_base"],
        "axes.labelsize": e["tamano_rotulo"],
        "xtick.labelsize": e["tamano_base"],
        "ytick.labelsize": e["tamano_base"],
        "figure.facecolor": e["color_fondo"],
        "axes.facecolor": e["color_fondo"],
        "savefig.facecolor": e["color_fondo"],
        "axes.edgecolor": e["color_malla"],
        "axes.labelcolor": e["color_texto"],
        "text.color": e["color_texto"],
        "xtick.color": e["color_texto_secundario"],
        "ytick.color": e["color_texto_secundario"],
        "axes.grid": True,
        "grid.color": e["color_malla"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",  # texto editable en el SVG
    }


@contextmanager
def aplicar(config: dict):
    with plt.rc_context(parametros(config)):
        yield


def paleta(config: dict):
    """Mapa de color secuencial, perceptualmente uniforme (viridis por defecto)."""
    return colormaps[config["figura"]["paleta"]]
