"""Mapa de densidad conjunta del espacio postural con histogramas marginales.

Eje X: ángulo lateral. Eje Y: ángulo frontal. El color representa tiempo de
permanencia (cantidad de muestras por región). Solo se grafican muestras
posteriores al CALIBRADO de su sesión.

Advertencia que la figura repite en su pie cuando agrega más de una sesión:
cada sesión tiene su propio cero por la recalibración, así que sus ángulos no son
comparables en términos absolutos. Nada aquí rellena huecos ni genera muestras.
"""

from __future__ import annotations

import io
import re
import textwrap
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

from ..modelo import seleccion_analizable
from . import estilo
from .estilo import plt  # pyplot ya configurado con el backend sin pantalla

TIPOS_MIME = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "jpg": "image/jpeg",
}


@dataclass
class ResultadoFigura:
    figura: Figure
    titulo: str
    pie: str
    n_muestras: int
    n_sesiones: int
    n_usuarios: int
    metodo: str                  # kde | hex | histograma2d (respaldo) | vacio
    rutas: list[Path] = field(default_factory=list)


def _como_lista(x) -> list[str] | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def _alcance(usuarios: list[str] | None, sesiones: list[str] | None) -> str:
    partes = []
    if usuarios:
        partes.append(("persona " if len(usuarios) == 1 else "personas ") + ", ".join(usuarios))
    if sesiones:
        partes.append(("sesión " if len(sesiones) == 1 else "sesiones ") + ", ".join(sesiones))
    return " · ".join(partes) if partes else "conjunto completo"


def _limites(config: dict, lat: np.ndarray, fro: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Vista cuadrada (misma escala en grados en ambos ejes) que contiene umbrales y datos."""
    u = config["umbrales"]
    margen = float(config["figura"]["margen_vista_grados"])
    x0, x1 = u["LATERAL_MIN"], u["LATERAL_MAX"]
    y0, y1 = u["FRONTAL_MIN"], u["FRONTAL_MAX"]
    if lat.size:
        x0, x1 = min(x0, lat.min()), max(x1, lat.max())
        y0, y1 = min(y0, fro.min()), max(y1, fro.max())
    lado = max(x1 - x0, y1 - y0) + 2 * margen
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - lado / 2, cx + lado / 2), (cy - lado / 2, cy + lado / 2)


def _kde(lat: np.ndarray, fro: np.ndarray, config: dict, xlim, ylim):
    """Devuelve (X, Y, Z en muestras/grado²) o None si la densidad no es estimable."""
    ck = config["figura"]["kde"]
    if lat.size < int(ck["min_muestras"]):
        return None
    try:
        kde = gaussian_kde(np.vstack([lat, fro]), bw_method=ck["ancho_banda"])  # "scott", "silverman" o número
    except (np.linalg.LinAlgError, ValueError):
        return None  # covarianza singular: p. ej. todos los valores iguales en un eje
    r = int(ck["resolucion_malla"])
    X, Y = np.meshgrid(np.linspace(*xlim, r), np.linspace(*ylim, r))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape) * lat.size
    if not np.isfinite(Z).all() or Z.max() <= 0:
        return None
    return X, Y, Z


def _valor_en_malla(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Valor de la malla Z en la celda más cercana a cada punto (x, y)."""
    xs, ys = X[0, :], Y[:, 0]
    ix = np.clip(np.searchsorted(xs, x), 0, xs.size - 1)
    iy = np.clip(np.searchsorted(ys, y), 0, ys.size - 1)
    return Z[iy, ix]


def _bordes_enteros(lim: tuple[float, float], ancho: float) -> np.ndarray:
    """Bordes de bin centrados en grados enteros (la resolución del sensor es 1°)."""
    a = np.floor(lim[0]) - ancho / 2
    b = np.ceil(lim[1]) + ancho / 2
    return np.arange(a, b + ancho, ancho)


def _slug(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos.lower()).strip("_")


def figura_densidad(
    mediciones: pd.DataFrame,
    config: dict,
    usuario: str | list[str] | None = None,
    sesion: str | list[str] | None = None,
    metodo: str | None = None,
    salida: str | Path | None = None,
    nombre: str | None = None,
) -> ResultadoFigura:
    """Genera el mapa de densidad conjunta.

    Filtros: ``usuario`` y ``sesion`` (uno o varios). Sin filtros, el conjunto completo.
    ``metodo``: kde | hex | auto (por defecto, el de configuración).
    Si se pasa ``salida``, exporta PNG y SVG (según ``figura.formatos``) y el pie en .txt.
    """
    cf = config["figura"]
    ce = config["estilo"]
    u = config["umbrales"]
    usuarios = _como_lista(usuario)
    ids_sesion = _como_lista(sesion)
    incluir_sc = bool(config["sesiones"]["incluir_sin_calibrar"])

    filtro = pd.Series(True, index=mediciones.index)
    if usuarios:
        filtro &= mediciones["usuario_id"].isin(usuarios)
    if ids_sesion:
        filtro &= mediciones["sesion_id"].isin(ids_sesion)
    en_filtro = mediciones.loc[filtro]
    analizable = seleccion_analizable(en_filtro, incluir_sc)
    datos = en_filtro.loc[analizable]
    n_pre = int((~en_filtro["post_calibracion"] & en_filtro["t_rel_calibracion"].notna()).sum())
    n_sc = int(en_filtro["t_rel_calibracion"].isna().sum()) if not incluir_sc else 0

    lat = datos["lateral"].to_numpy(dtype=float)
    fro = datos["frontal"].to_numpy(dtype=float)
    n = int(lat.size)
    n_ses = int(datos["sesion_id"].nunique())
    n_usr = int(datos["usuario_id"].nunique())

    metodo = metodo or cf["metodo"]
    if metodo == "auto":
        metodo = "hex" if n >= int(cf["auto_umbral_muestras"]) else "kde"
    if metodo not in ("kde", "hex"):
        raise ValueError(f"Método de figura desconocido: {metodo!r} (use kde, hex o auto)")

    alcance = _alcance(usuarios, ids_sesion)
    titulo = f"Espacio postural · {alcance} · n = {n} muestra{'' if n == 1 else 's'}"
    xlim, ylim = _limites(config, lat, fro)
    cmap = estilo.paleta(config)

    with estilo.aplicar(config):
        fig = plt.figure(figsize=tuple(cf["tamano_pulgadas"]))
        prop = float(cf["proporcion_marginal"])
        gs = fig.add_gridspec(
            2, 3, width_ratios=[1, prop, 0.045], height_ratios=[prop, 1],
            wspace=0.05, hspace=0.05, left=0.10, right=0.88, top=0.92, bottom=0.22,
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_sup = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_der = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_cb = fig.add_subplot(gs[1, 2])

        metodo_usado = metodo
        notas_metodo = ""
        if n == 0:
            metodo_usado = "vacio"
            ax.text(0.5, 0.5, "Sin muestras analizables para este filtro", transform=ax.transAxes,
                    ha="center", va="center", color=ce["color_texto_secundario"])
            ax_cb.set_axis_off()
        elif metodo == "hex":
            ch = cf["hex"]
            hb = ax.hexbin(lat, fro, gridsize=int(ch["tamano_malla"]), mincnt=int(ch["min_conteo"]),
                           cmap=cmap, extent=(*xlim, *ylim), linewidths=0.2, edgecolors=ce["color_fondo"])
            fig.colorbar(hb, cax=ax_cb).set_label("Tiempo de permanencia (muestras por hexágono)")
            notas_metodo = (f"Color: muestras por hexágono (malla de {ch['tamano_malla']} hexágonos de ancho). ")
        else:
            res = _kde(lat, fro, config, xlim, ylim)
            if res is None:
                metodo_usado = "histograma2d"
                bx = _bordes_enteros(xlim, 1.0)
                by = _bordes_enteros(ylim, 1.0)
                H, _, _ = np.histogram2d(lat, fro, bins=[bx, by])
                malla = ax.pcolormesh(bx, by, np.ma.masked_equal(H.T, 0), cmap=cmap,
                                      vmin=0, vmax=max(H.max(), 1))
                fig.colorbar(malla, cax=ax_cb).set_label("Tiempo de permanencia (muestras por celda de 1°×1°)")
                notas_metodo = (f"Densidad kernel no estimable con estos datos (n = {n}, mínimo "
                                f"{cf['kde']['min_muestras']}, o varianza nula en un eje); se muestra el "
                                "conteo por celda de 1°×1° sin suavizar. ")
            else:
                X, Y, Z = res
                ck = cf["kde"]
                piso = float(ck["fraccion_minima_nivel"]) * Z.max()
                niveles = np.linspace(piso, Z.max(), int(ck["niveles"]) + 1)
                cs = ax.contourf(X, Y, Z, levels=niveles, cmap=cmap)
                ax.contour(X, Y, Z, levels=niveles, colors=ce["color_fondo"], linewidths=0.3, alpha=0.6)
                cb = fig.colorbar(cs, cax=ax_cb, format="%.1f")
                cb.set_label("Tiempo de permanencia (muestras / grado²)")
                # Las muestras que caen donde no hay relleno se dibujan una a una:
                # el suavizado no debe esconder excursiones aisladas.
                z_muestra = _valor_en_malla(X, Y, Z, lat, fro)
                aisladas = z_muestra < piso
                n_aisladas = int(aisladas.sum())
                if n_aisladas:
                    ax.scatter(lat[aisladas], fro[aisladas], s=float(cf["kde"]["tamano_punto_aislado"]),
                               color=ce["color_texto"], linewidths=0, zorder=4,
                               label=f"Muestras en zona de baja densidad ({n_aisladas})")
                nombre_bw = str(ck["ancho_banda"]).capitalize()
                notas_metodo = (f"Color: densidad kernel gaussiana (ancho de banda {nombre_bw}) "
                                "escalada a muestras por grado²; las zonas por debajo del "
                                f"{float(ck['fraccion_minima_nivel']):.0%} del máximo no se rellenan y "
                                f"sus {n_aisladas} muestras se dibujan como puntos. ")

        # Rectángulo de umbrales de alerta
        ax.add_patch(Rectangle(
            (u["LATERAL_MIN"], u["FRONTAL_MIN"]),
            u["LATERAL_MAX"] - u["LATERAL_MIN"], u["FRONTAL_MAX"] - u["FRONTAL_MIN"],
            fill=False, edgecolor=ce["color_umbral"], linewidth=float(cf["umbral_rectangulo"]["grosor_linea"]),
            linestyle=(0, tuple(cf["umbral_rectangulo"]["patron"])), zorder=5,
            label=(f"Umbrales de alerta (lateral {u['LATERAL_MIN']}° a {u['LATERAL_MAX']}°, "
                   f"frontal {u['FRONTAL_MIN']}° a {u['FRONTAL_MAX']}°)"),
        ))
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor=ce["color_malla"],
                  fontsize=ce["tamano_pie"])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Ángulo lateral (°)   ← izquierda · derecha →")
        ax.set_ylabel("Ángulo frontal (°)   ← atrás · adelante →")
        ax.set_axisbelow(True)

        # Histogramas marginales (bins de 1°: la resolución del sensor)
        ancho = float(cf["histograma"]["ancho_bin_grados"])
        hist = dict(color=ce["color_histograma"], alpha=float(ce["alfa_histograma"]),
                    edgecolor=ce["color_fondo"], linewidth=0.4)
        if n:
            ax_sup.hist(lat, bins=_bordes_enteros(xlim, ancho), **hist)
            ax_der.hist(fro, bins=_bordes_enteros(ylim, ancho), orientation="horizontal", **hist)
        linea_umbral = dict(color=ce["color_umbral"], linewidth=0.9,
                            linestyle=(0, tuple(cf["umbral_rectangulo"]["patron"])))
        for v in (u["LATERAL_MIN"], u["LATERAL_MAX"]):
            ax_sup.axvline(v, **linea_umbral)
        for v in (u["FRONTAL_MIN"], u["FRONTAL_MAX"]):
            ax_der.axhline(v, **linea_umbral)
        ax_sup.set_ylabel("Muestras")
        ax_der.set_xlabel("Muestras")
        ax_sup.tick_params(labelbottom=False)
        ax_der.tick_params(labelleft=False)
        ax_sup.grid(axis="x", visible=False)
        ax_der.grid(axis="y", visible=False)

        fig.suptitle(titulo, x=0.10, ha="left", fontsize=ce["tamano_titulo"], fontweight="bold")

        pie = _pie(config, n, n_pre, n_sc, n_ses, n_usr, notas_metodo, datos)
        envuelto = "\n".join(textwrap.fill(p, width=int(cf["ancho_pie_caracteres"])) for p in pie.split("\n"))
        fig.text(0.10, 0.14, envuelto, ha="left", va="top", fontsize=ce["tamano_pie"],
                 color=ce["color_texto_secundario"])

    resultado = ResultadoFigura(fig, titulo, pie, n, n_ses, n_usr, metodo_usado)
    if salida is not None:
        base = Path(salida)
        base.mkdir(parents=True, exist_ok=True)
        stem = nombre or f"densidad_{_slug(alcance)}_{metodo}"
        for fmt in cf["formatos"]:
            ruta = base / f"{stem}.{fmt}"
            ruta.write_bytes(exportar(fig, fmt, config))
            resultado.rutas.append(ruta)
        ruta_pie = base / f"{stem}.pie.txt"
        ruta_pie.write_text(titulo + "\n\n" + pie + "\n", encoding="utf-8")
        resultado.rutas.append(ruta_pie)
    return resultado


def exportar(figura: Figure, formato: str, config: dict) -> bytes:
    """Serializa la figura. Formatos: png, svg, pdf, jpg (ráster a figura.dpi).

    Se guarda dentro del estilo porque parte de él (p. ej. svg.fonttype, que deja el
    texto del SVG editable) se lee al guardar, no al dibujar.
    """
    if formato not in TIPOS_MIME:
        raise ValueError(f"Formato no soportado: {formato!r} (use {', '.join(TIPOS_MIME)})")
    buf = io.BytesIO()
    raster = formato in ("png", "jpg")
    with estilo.aplicar(config):
        figura.savefig(buf, format="jpeg" if formato == "jpg" else formato,
                       dpi=int(config["figura"]["dpi"]) if raster else None)
    return buf.getvalue()


def _pie(config, n, n_pre, n_sc, n_ses, n_usr, notas_metodo, datos) -> str:
    nominal = config["calidad"]["cadencia_nominal_s"]
    partes = [
        f"Muestras graficadas: {n}, todas posteriores al CALIBRADO de su sesión"
        + (" (o de sesiones sin calibrar incluidas por configuración)" if config["sesiones"]["incluir_sin_calibrar"] else "")
        + f". Excluidas: {n_pre} previas "
        f"a la calibración" + (f" y {n_sc} de sesiones sin calibrar" if n_sc else "") + ". "
        + notas_metodo
        + f"Con cadencia nominal de {nominal} s por muestra, una muestra equivale a ~{nominal} s de permanencia."
    ]
    if n_ses > 1 or n_usr > 1:
        partes.append(
            f"AGREGACIÓN: esta figura combina {n_ses} sesiones de {n_usr} "
            f"persona{'s' if n_usr != 1 else ''}. ADVERTENCIA: cada sesión tiene su propio cero por la "
            "recalibración; los ángulos son relativos a esa calibración y no son comparables en términos "
            "absolutos entre sesiones."
        )
    elif n_ses == 1:
        partes.append(f"Una sola sesión ({datos['sesion_id'].iloc[0]}); ángulos relativos a su calibración.")
    if n == 0:
        partes.append("No hay datos que graficar con este filtro; no se rellena ni se simula nada.")
    return "\n".join(partes)
