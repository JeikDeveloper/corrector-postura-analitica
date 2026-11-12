"""Mapa de densidad en 3D: la misma densidad del mapa 2D (sección 4.1) como superficie.

- ``malla_3d``: datos de la superficie para la vista interactiva del navegador (Plotly).
- ``figura_densidad_3d``: imagen fija (matplotlib) para descargar en PNG, SVG, PDF o JPG.

Ejes: lateral (X), frontal (Y) y tiempo de permanencia en muestras por grado² (Z).
Se usan las mismas muestras, el mismo ancho de banda y los mismos límites de vista
que el mapa 2D, así que las alturas son comparables con sus colores.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.figure import Figure

from . import estilo
from .densidad import _alcance, _bordes_enteros, _como_lista, _kde, _limites, seleccion
from .estilo import plt


@dataclass
class Superficie:
    x: np.ndarray            # centros en lateral (°)
    y: np.ndarray            # centros en frontal (°)
    z: np.ndarray            # z[fila=y, columna=x]
    metodo: str              # kde | histograma2d | vacio
    unidad: str
    n: int
    n_sesiones: int
    n_usuarios: int
    n_pre: int
    n_sc: int
    titulo: str
    pie: str


def _superficie(mediciones, config, usuario, sesion) -> Superficie:
    c3 = config["figura"]["tres_d"]
    usuarios, sesiones = _como_lista(usuario), _como_lista(sesion)
    datos, n_pre, n_sc = seleccion(mediciones, config, usuarios, sesiones)
    lat = datos["lateral"].to_numpy(dtype=float)
    fro = datos["frontal"].to_numpy(dtype=float)
    n = int(lat.size)
    n_ses, n_usr = int(datos["sesion_id"].nunique()), int(datos["usuario_id"].nunique())
    xlim, ylim = _limites(config, lat, fro)
    titulo = f"Espacio postural en 3D · {_alcance(usuarios, sesiones)} · n = {n} muestra{'' if n == 1 else 's'}"

    if n == 0:
        x = np.linspace(*xlim, 2)
        y = np.linspace(*ylim, 2)
        z = np.zeros((2, 2))
        metodo, unidad = "vacio", "muestras / grado²"
        nota = "No hay muestras analizables con este filtro; no se rellena ni se simula nada."
    else:
        cfg_malla = {**config, "figura": {**config["figura"],
                                          "kde": {**config["figura"]["kde"],
                                                  "resolucion_malla": int(c3["resolucion_malla"])}}}
        res = _kde(lat, fro, cfg_malla, xlim, ylim)
        if res is None:
            bx, by = _bordes_enteros(xlim, 1.0), _bordes_enteros(ylim, 1.0)
            H, _, _ = np.histogram2d(lat, fro, bins=[bx, by])
            x, y, z = (bx[:-1] + bx[1:]) / 2, (by[:-1] + by[1:]) / 2, H.T
            metodo, unidad = "histograma2d", "muestras por celda de 1°×1°"
            nota = (f"Densidad kernel no estimable con estos datos (n = {n}, o varianza nula en un eje): la altura "
                    "es el conteo por celda de 1°×1° sin suavizar.")
        else:
            X, Y, Z = res
            x, y, z = X[0, :], Y[:, 0], Z
            metodo, unidad = "kde", "muestras / grado²"
            nota = (f"Altura: densidad kernel gaussiana (ancho de banda {str(config['figura']['kde']['ancho_banda']).capitalize()}) "
                    "escalada a muestras por grado², la misma del mapa 2D. Las muestras aisladas que el mapa 2D "
                    "dibuja como puntos aquí quedan como relieve casi plano.")

    u = config["umbrales"]
    partes = [f"{nota} Rectángulo en el piso: umbrales de alerta (lateral {u['LATERAL_MIN']}° a "
              f"{u['LATERAL_MAX']}°, frontal {u['FRONTAL_MIN']}° a {u['FRONTAL_MAX']}°). Solo muestras posteriores "
              f"al CALIBRADO; excluidas: {n_pre} previas a la calibración" + (f" y {n_sc} de sesiones sin calibrar" if n_sc else "") + "."]
    if n_ses > 1 or n_usr > 1:
        partes.append(f"AGREGACIÓN: combina {n_ses} sesiones de {n_usr} persona{'s' if n_usr != 1 else ''}; cada "
                      "sesión tiene su propio cero por la recalibración, así que los ángulos no son comparables en "
                      "términos absolutos entre sesiones.")
    return Superficie(x, y, z, metodo, unidad, n, n_ses, n_usr, n_pre, n_sc, titulo, "\n".join(partes))


def malla_3d(mediciones: pd.DataFrame, config: dict, usuario: str | None = None,
             sesion: str | None = None) -> dict:
    """Superficie en un dict serializable a JSON, para la vista interactiva."""
    s = _superficie(mediciones, config, usuario, sesion)
    dec = int(config["figura"]["tres_d"]["decimales_z"])
    u = config["umbrales"]
    return {
        "x": np.round(s.x, 2).tolist(), "y": np.round(s.y, 2).tolist(), "z": np.round(s.z, dec).tolist(),
        "metodo": s.metodo, "unidad": s.unidad, "n": s.n, "n_sesiones": s.n_sesiones,
        "titulo": s.titulo, "pie": s.pie,
        "umbrales": {"lateral": [u["LATERAL_MIN"], u["LATERAL_MAX"]], "frontal": [u["FRONTAL_MIN"], u["FRONTAL_MAX"]]},
        "paleta": config["figura"]["paleta"], "color_umbral": config["estilo"]["color_umbral"],
        "proporcion_altura": float(config["figura"]["tres_d"]["proporcion_altura"]),
        "plotly_cdn": config["figura"]["tres_d"]["plotly_cdn"],
    }


def figura_densidad_3d(mediciones: pd.DataFrame, config: dict, usuario: str | None = None,
                       sesion: str | None = None) -> tuple[Figure, Superficie]:
    """Imagen fija de la superficie, con la cámara de ``figura.tres_d``."""
    s = _superficie(mediciones, config, usuario, sesion)
    c3, ce, cf = config["figura"]["tres_d"], config["estilo"], config["figura"]
    u = config["umbrales"]
    with estilo.aplicar(config):
        fig = plt.figure(figsize=tuple(c3["tamano_pulgadas"]))
        ax = fig.add_axes([0.02, 0.2, 0.84, 0.72], projection="3d")
        X, Y = np.meshgrid(s.x, s.y)
        cmap = colormaps[cf["paleta"]]
        if s.metodo == "vacio":
            ax.text2D(0.5, 0.5, "Sin muestras analizables para este filtro", transform=ax.transAxes,
                      ha="center", color=ce["color_texto_secundario"])
        else:
            sup = ax.plot_surface(X, Y, s.z, cmap=cmap, linewidth=0, antialiased=True,
                                  rstride=1, cstride=1, vmin=0, vmax=float(s.z.max()) or 1.0)
            sup.set_rasterized(True)   # SVG/PDF livianos; los textos siguen siendo vectoriales
            cax = fig.add_axes([0.87, 0.3, 0.025, 0.5])
            fig.colorbar(sup, cax=cax).set_label(f"Tiempo de permanencia ({s.unidad})")
        x0, x1 = u["LATERAL_MIN"], u["LATERAL_MAX"]
        y0, y1 = u["FRONTAL_MIN"], u["FRONTAL_MAX"]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [0, 0, 0, 0, 0], color=ce["color_umbral"],
                linewidth=float(cf["umbral_rectangulo"]["grosor_linea"]),
                linestyle=(0, tuple(cf["umbral_rectangulo"]["patron"])), zorder=10)
        ax.set_xlim(s.x.min(), s.x.max())
        ax.set_ylim(s.y.min(), s.y.max())
        ax.set_zlim(bottom=0)
        ax.set_xlabel("Lateral (°)  ← izq. · der. →", labelpad=8)
        ax.set_ylabel("Frontal (°)  ← atrás · adelante →", labelpad=8)
        ax.set_zlabel(s.unidad, labelpad=6)
        ax.view_init(elev=float(c3["elevacion_grados"]), azim=float(c3["azimut_grados"]))
        for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
            eje.pane.set_facecolor(ce["color_fondo"])
            eje.pane.set_edgecolor(ce["color_malla"])
        fig.suptitle(s.titulo, x=0.06, ha="left", fontsize=ce["tamano_titulo"], fontweight="bold")
        ancho = int(cf["ancho_pie_caracteres"])
        pie = (s.pie + f"\nCámara: elevación {c3['elevacion_grados']}°, azimut {c3['azimut_grados']}°. En la página la "
               "superficie se puede girar.")
        fig.text(0.06, 0.15, "\n".join(textwrap.fill(p, width=ancho) for p in pie.split("\n")), ha="left",
                 va="top", fontsize=ce["tamano_pie"], color=ce["color_texto_secundario"])
    return fig, s
