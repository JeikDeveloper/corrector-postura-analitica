"""Figuras del análisis ergonómico (ISO 11226).

- ``zonas``: densidad sobre las zonas óptima / aceptable / extrema, con |σ| y |R|.
- ``momentos``: un panel por momento del día (mañana, tarde, noche).
- ``sujetos``: un panel por persona (o por sesión si no hay personas identificadas).
- ``histogramas``: frecuencia de cada ángulo, coloreada por zona.

En ``momentos`` y ``sujetos`` el color es el porcentaje del tiempo de cada panel por
grado², con la misma escala en todos los paneles, para comparar formas aunque
tengan distinta cantidad de muestras.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Patch, Rectangle

from .. import ergonomia as erg
from . import estilo
from .densidad import _bordes_enteros, _kde
from .estilo import plt

FIGURAS = ("zonas", "momentos", "sujetos", "histogramas")


@dataclass
class ResultadoErgonomia:
    figura: Figure
    titulo: str
    pie: str
    n_muestras: int


# ---------------------------------------------------------------- piezas comunes

def _vista(config: dict, *series) -> tuple[float, float]:
    """Rango cuadrado: el de configuración, ampliado si hay datos fuera."""
    a, b = (float(x) for x in config["ergonomia"]["rango_grados"])
    for s in series:
        if len(s):
            a, b = min(a, float(np.min(s)) - 2), max(b, float(np.max(s)) + 2)
    return a, b


def _pintar_zonas(ax, config: dict, lim: tuple[float, float], alfa: float | None = None) -> None:
    ce = config["ergonomia"]
    alfa = float(ce["alfa_zonas"]) if alfa is None else alfa
    colores = ce["colores_zona"]
    a, b = lim
    ax.add_patch(Rectangle((a, a), b - a, b - a, color=colores["extrema"], alpha=alfa, lw=0, zorder=0))
    # En orden inverso: la primera zona de la lista (la que gana) queda encima.
    for z in reversed(ce["zonas"]):
        (f0, f1), (l0, l1) = z["frontal"], z["lateral"]
        l0, l1 = max(l0, a), min(l1, b)
        f0, f1 = max(f0, a), min(f1, b)
        if l1 > l0 and f1 > f0:
            ax.add_patch(Rectangle((l0, f0), l1 - l0, f1 - f0, color=colores[z["nivel"]],
                                   alpha=1.0 if alfa >= 1 else min(1.0, alfa + 0.15), lw=0, zorder=0.5))


def _silueta(ax, config: dict, alfa: float = 0.75) -> None:
    """Persona sentada vista desde arriba, de frente al escritorio (adelante = arriba).

    Es solo una referencia de orientación: no está a escala de los ángulos.
    """
    ce = config["ergonomia"]
    if not ce["silueta"]:
        return
    c = dict(fill=False, edgecolor=ce["color_silueta"], linewidth=1.0, alpha=alfa, zorder=3)
    ax.add_patch(Rectangle((-27, 21), 54, 8, **c))                 # escritorio
    ax.add_patch(Rectangle((-12, 25.5), 24, 2.5, **c))             # pantalla
    ax.add_patch(Rectangle((-9, 21.8), 18, 2.6, **c))              # teclado
    ax.add_patch(Rectangle((12, 22), 3, 3.6, **c))                 # mouse
    ax.add_patch(Ellipse((0, -4), 34, 11, **c))                    # hombros y tronco
    ax.add_patch(Circle((0, 1.5), 5.2, **c))                       # cabeza
    for lado in (-1, 1):                                           # brazos
        ax.add_line(Line2D([lado * 15.5, lado * 14, lado * 7], [-3, 10, 22],
                           color=ce["color_silueta"], linewidth=1.0, alpha=alfa, zorder=3))
    ax.add_patch(Rectangle((-13, -16), 26, 6, **c))                # respaldo de la silla


def _ejes_angulares(ax, lim) -> None:
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#ffffff", linewidth=0.8, alpha=0.9, zorder=1)
    ax.axvline(0, color="#ffffff", linewidth=0.8, alpha=0.9, zorder=1)


def _densidad_pct(lat: np.ndarray, fro: np.ndarray, config: dict, lim):
    """Malla de densidad en % del tiempo por grado², o None si no es estimable."""
    res = _kde(lat, fro, config, lim, lim)
    if res is None:
        return None
    X, Y, Z = res
    return X, Y, Z / lat.size * 100.0


def _pie_texto(fig, config: dict, texto: str, y: float) -> None:
    ce = config["estilo"]
    ancho = int(config["figura"]["ancho_pie_caracteres"])
    envuelto = "\n".join(textwrap.fill(p, width=ancho) for p in texto.split("\n"))
    fig.text(0.06, y, envuelto, ha="left", va="top", fontsize=ce["tamano_pie"],
             color=ce["color_texto_secundario"])


def _leyenda_zonas(config: dict, pct: dict | None = None, colores: str = "colores_zona") -> list[Patch]:
    ce = config["ergonomia"]
    return [Patch(facecolor=ce[colores][z], edgecolor="none",
                  label=ce["nombres_zona"][z] + (f" · {pct[z]:.1f} %" if pct and pct.get(z) is not None else ""))
            for z in erg.NIVELES]


def _nota_calibracion(n_ses: int) -> str:
    texto = ("Los ángulos son relativos al CALIBRADO de cada sesión (postura erguida = 0°); las zonas se "
             "aplican sobre ese cero.")
    if n_ses > 1:
        texto += (f" AGREGACIÓN: combina {n_ses} sesiones con ceros distintos; no son comparables en "
                  "términos absolutos.")
    return texto


# ---------------------------------------------------------------- figura: zonas

def _figura_zonas(datos, global_, config, alcance) -> ResultadoErgonomia:
    ce, cs = config["ergonomia"], config["estilo"]
    lat = datos["lateral"].to_numpy(dtype=float)
    fro = datos["frontal"].to_numpy(dtype=float)
    n = lat.size
    m = erg.metricas(datos, config)
    g = erg.metricas(global_, config)
    lim = _vista(config, lat, fro)
    titulo = f"Clasificación postural ISO 11226 · {alcance} · n = {n}"

    fig = plt.figure(figsize=tuple(ce["tamanos_pulgadas"]["zonas"]))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.2, 0.04], height_ratios=[0.2, 1],
                          wspace=0.05, hspace=0.05, left=0.09, right=0.88, top=0.92, bottom=0.25)
    ax = fig.add_subplot(gs[1, 0])
    ax_sup = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_der = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_cb = fig.add_subplot(gs[1, 2])
    _pintar_zonas(ax, config, lim)
    _ejes_angulares(ax, lim)
    ax.grid(False)
    _silueta(ax, config)

    nota_metodo = ""
    if n == 0:
        ax.text(0.5, 0.5, "Sin muestras analizables para este filtro", transform=ax.transAxes,
                ha="center", va="center", color=cs["color_texto_secundario"])
        ax_cb.set_axis_off()
    else:
        res = _kde(lat, fro, config, lim, lim)
        cmap = colormaps[ce["paleta"]]
        if res is None:
            bordes = _bordes_enteros(lim, 1.0)
            H, _, _ = np.histogram2d(lat, fro, bins=[bordes, bordes])
            malla = ax.pcolormesh(bordes, bordes, np.ma.masked_equal(H.T, 0), cmap=cmap,
                                  alpha=float(ce["alfa_densidad"]), zorder=2)
            fig.colorbar(malla, cax=ax_cb).set_label("Muestras por celda de 1°×1°")
            nota_metodo = "Densidad kernel no estimable con estos datos: se muestra el conteo por celda de 1°. "
        else:
            X, Y, Z = res
            ck = config["figura"]["kde"]
            piso = float(ck["fraccion_minima_nivel"]) * Z.max()
            niveles = np.linspace(piso, Z.max(), int(ck["niveles"]) + 1)
            cs_ = ax.contourf(X, Y, Z, levels=niveles, cmap=cmap, alpha=float(ce["alfa_densidad"]), zorder=2)
            ax.contour(X, Y, Z, levels=niveles, colors="#2b1d3a", linewidths=0.35, alpha=0.6, zorder=2.1)
            fig.colorbar(cs_, cax=ax_cb, format="%.1f").set_label("Tiempo de permanencia (muestras / grado²)")
        # |σ|: círculo centrado en la postura media; |R|: flecha desde el cero hasta ella.
        ax.add_patch(Circle((m.media_lateral, m.media_frontal), m.dispersion, fill=False,
                            edgecolor=ce["color_dispersion"], linewidth=2, linestyle=(0, (2, 2)), zorder=6))
        if m.desplazamiento > 0:
            ax.add_patch(FancyArrowPatch((0, 0), (m.media_lateral, m.media_frontal), arrowstyle="-|>",
                                         mutation_scale=14, color=ce["color_desplazamiento"], linewidth=2.2,
                                         zorder=7))
        ax.plot([m.media_lateral], [m.media_frontal], marker="o", markersize=4,
                color=ce["color_desplazamiento"], zorder=8)

    ax.set_xlabel("Inclinación lateral β (°)   ← izquierda · derecha →")
    ax.set_ylabel("Flexión / extensión α (°)   ← atrás · adelante →")
    extra = [Line2D([], [], color=ce["color_dispersion"], linestyle=(0, (2, 2)), linewidth=2,
                    label=f"|σ| dispersión = {m.dispersion}°" if n else "|σ| dispersión"),
             Line2D([], [], color=ce["color_desplazamiento"], linewidth=2.2, marker=">",
                    label=f"|R| desplazamiento = {m.desplazamiento}°" if n else "|R| desplazamiento")]
    ax.legend(handles=_leyenda_zonas(config, m.pct_zona if n else None) + extra, loc="lower left",
              fontsize=cs["tamano_pie"], framealpha=0.92, edgecolor=cs["color_malla"])

    # Marginales coloreados por zona de cada eje
    if n:
        _hist_por_zona(ax_sup, lat, "lateral", config, lim, vertical=True)
        _hist_por_zona(ax_der, fro, "frontal", config, lim, vertical=False)
    ax_sup.tick_params(labelbottom=False)
    ax_der.tick_params(labelleft=False)
    ax_sup.set_ylabel("Muestras")
    ax_der.set_xlabel("Muestras")
    ax_sup.grid(axis="x", visible=False)
    ax_der.grid(axis="y", visible=False)

    fig.suptitle(titulo, x=0.09, ha="left", fontsize=cs["tamano_titulo"], fontweight="bold")
    sg = erg._cociente(m.dispersion, g.dispersion)
    rg = erg._cociente(m.desplazamiento, g.desplazamiento)
    pie = (f"Zonas según {ce['fuente']}. " + nota_metodo
           + (f"Postura media: α̅ = {m.media_frontal}°, β̅ = {m.media_lateral}°; σα = {m.sd_frontal}°, "
              f"σβ = {m.sd_lateral}°. |σ| = √(σα² + σβ²) = {m.dispersion}°; |R| = √(α̅² + β̅²) = "
              f"{m.desplazamiento}°. Respecto al conjunto completo: σG = {sg}, RG = {rg}. " if n else "")
           + "La silueta es solo una referencia de orientación (persona vista desde arriba, frente al "
             "escritorio) y no está a escala.\n"
           + _nota_calibracion(int(datos["sesion_id"].nunique())))
    _pie_texto(fig, config, pie, 0.16)
    return ResultadoErgonomia(fig, titulo, pie, n)


def _hist_por_zona(ax, valores, eje, config, lim, vertical: bool) -> None:
    ce = config["ergonomia"]
    bordes = _bordes_enteros(lim, 1.0)
    conteo, _ = np.histogram(valores, bins=bordes)
    centros = (bordes[:-1] + bordes[1:]) / 2
    colores = [ce["colores_barra"][z] for z in erg.clasificar_eje(centros, eje, config)]
    borde = config["estilo"]["color_fondo"]
    if vertical:
        ax.bar(centros, conteo, width=1.0, color=colores, edgecolor=borde, linewidth=0.3)
    else:
        ax.barh(centros, conteo, height=1.0, color=colores, edgecolor=borde, linewidth=0.3)


# ---------------------------------------------------------------- paneles pequeños

def _panel_pct(ax, datos, config, lim, vmax, cmap, titulo):
    ce, cs = config["ergonomia"], config["estilo"]
    _pintar_zonas(ax, config, lim, alfa=0.18)
    _ejes_angulares(ax, lim)
    ax.grid(False)
    _silueta(ax, config, alfa=0.55)
    lat = datos["lateral"].to_numpy(dtype=float)
    fro = datos["frontal"].to_numpy(dtype=float)
    mapa = None
    if lat.size == 0:
        ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes, ha="center", va="center",
                color=cs["color_texto_secundario"], fontsize=cs["tamano_rotulo"], zorder=9)
    else:
        res = _densidad_pct(lat, fro, config, lim)
        if res is None:
            ax.scatter(lat, fro, s=8, color=cmap(0.85), zorder=4)
        else:
            X, Y, Z = res
            piso = float(config["figura"]["kde"]["fraccion_minima_nivel"]) * vmax
            niveles = np.linspace(piso, vmax, int(config["figura"]["kde"]["niveles"]) + 1)
            mapa = ax.contourf(X, Y, np.clip(Z, None, vmax), levels=niveles, cmap=cmap,
                               alpha=float(ce["alfa_densidad"]), zorder=2)
            m = erg.metricas(datos, config)
            ax.add_patch(Circle((m.media_lateral, m.media_frontal), m.dispersion, fill=False,
                                edgecolor=ce["color_dispersion"], linewidth=1.2,
                                linestyle=(0, (2, 2)), zorder=6))
            if m.desplazamiento > 0:
                ax.add_patch(FancyArrowPatch((0, 0), (m.media_lateral, m.media_frontal), arrowstyle="-|>",
                                             mutation_scale=9, color=ce["color_desplazamiento"],
                                             linewidth=1.4, zorder=7))
    ax.set_title(titulo, fontsize=cs["tamano_rotulo"])
    return mapa


def _vmax_pct(grupos, config, lim) -> float:
    vmax = 0.0
    for d in grupos:
        if len(d):
            res = _densidad_pct(d["lateral"].to_numpy(float), d["frontal"].to_numpy(float), config, lim)
            if res is not None:
                vmax = max(vmax, float(res[2].max()))
    return vmax or 1.0


def _figura_momentos(datos, config, alcance) -> ResultadoErgonomia:
    ce, cs = config["ergonomia"], config["estilo"]
    momento = erg.momento_del_dia(datos["t"], config)
    claves = list(ce["momentos"])
    grupos = [datos[momento == c] for c in claves]
    lim = _vista(config, datos["lateral"], datos["frontal"])
    cmap = colormaps[ce["paleta"]]
    vmax = _vmax_pct(grupos, config, lim)
    titulo = f"Postura por momento del día · {alcance} · n = {len(datos)}"

    fig = plt.figure(figsize=tuple(ce["tamanos_pulgadas"]["momentos"]))
    gs = fig.add_gridspec(1, len(claves) + 1, width_ratios=[1] * len(claves) + [0.05],
                          left=0.06, right=0.93, top=0.84, bottom=0.30, wspace=0.18)
    mapa = None
    for i, (c, d) in enumerate(zip(claves, grupos)):
        ax = fig.add_subplot(gs[0, i])
        m = ce["momentos"][c]
        etiqueta = f"{m['nombre']} ({m['desde']:02d}–{m['hasta']:02d} h) · n = {len(d)}"
        mapa = _panel_pct(ax, d, config, lim, vmax, cmap, etiqueta) or mapa
        ax.set_xlabel("Lateral β (°)")
        if i == 0:
            ax.set_ylabel("Frontal α (°)")
    ax_cb = fig.add_subplot(gs[0, -1])
    if mapa is not None:
        fig.colorbar(mapa, cax=ax_cb, format="%.2f").set_label("% del tiempo del panel / grado²")
    else:
        ax_cb.set_axis_off()
    fig.suptitle(titulo, x=0.06, ha="left", fontsize=cs["tamano_titulo"], fontweight="bold")
    vacios = [ce["momentos"][c]["nombre"] for c, d in zip(claves, grupos) if not len(d)]
    pie = ("Cada panel muestra el porcentaje de su propio tiempo por grado², con la misma escala de color en "
           "los tres, para comparar la forma de la postura aunque tengan distinta cantidad de muestras. "
           "Círculo rojo: |σ| (dispersión); flecha verde: |R| (desplazamiento desde el cero). "
           "Momento del día según la hora de cada medición. "
           + (f"Sin mediciones en: {', '.join(vacios)}. " if vacios else "")
           + "\n" + _nota_calibracion(int(datos["sesion_id"].nunique())))
    _pie_texto(fig, config, pie, 0.13)
    return ResultadoErgonomia(fig, titulo, pie, len(datos))


def _figura_sujetos(global_, config, por_defecto_usuario) -> ResultadoErgonomia:
    ce, cs = config["ergonomia"], config["estilo"]
    personas = [u for u in sorted(global_["usuario_id"].astype(str).unique())]
    identificadas = [u for u in personas if u != por_defecto_usuario]
    if len(identificadas) >= 2 or (len(personas) >= 2):
        clave_col, unidad = "usuario_id", "persona"
        claves = personas
    else:
        clave_col, unidad = "sesion_id", "sesión"
        claves = list(global_["sesion_id"].astype(str).unique())
    max_p = int(ce["max_paneles"])
    omitidos = max(0, len(claves) - max_p)
    claves = claves[:max_p]
    grupos = [global_[global_[clave_col].astype(str) == k] for k in claves]
    lim = _vista(config, global_["lateral"], global_["frontal"])
    cmap = colormaps[ce["paleta"]]
    vmax = _vmax_pct(grupos, config, lim)

    cols = max(1, min(int(ce["columnas_cuadricula"]), len(claves)))
    filas = max(1, math.ceil(len(claves) / cols))
    ancho_p, alto_p = ce["tamanos_pulgadas"]["sujeto_panel"]
    fig = plt.figure(figsize=(ancho_p * cols + 1.0, alto_p * filas + 1.9))
    alto_total = alto_p * filas + 1.9
    gs = fig.add_gridspec(filas, cols + 1, width_ratios=[1] * cols + [0.06],
                          left=0.07, right=0.92, top=1 - 0.75 / alto_total, bottom=1.05 / alto_total,
                          wspace=0.25, hspace=0.35)
    titulo = f"Postura por {unidad} · {len(claves)} panel{'es' if len(claves) != 1 else ''}"
    mapa = None
    for i, (k, d) in enumerate(zip(claves, grupos)):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        nombre = "Sin identificar" if (unidad == "persona" and k == por_defecto_usuario) else k
        mapa = _panel_pct(ax, d, config, lim, vmax, cmap, f"{nombre} · n = {len(d)}") or mapa
        ax.tick_params(labelsize=cs["tamano_pie"])
    ax_cb = fig.add_subplot(gs[:, -1])
    if mapa is not None:
        fig.colorbar(mapa, cax=ax_cb, format="%.2f").set_label("% del tiempo del panel / grado²")
    else:
        ax_cb.set_axis_off()
    fig.suptitle(titulo, x=0.07, ha="left", fontsize=cs["tamano_titulo"], fontweight="bold")
    pie = (f"Un panel por {unidad}" + (" (no hay al menos dos personas, así que se separa por sesión)"
                                       if unidad == "sesión" else "")
           + ". Misma escala de color en todos: porcentaje del tiempo de cada panel por grado². "
           "Ejes: lateral β (X) y frontal α (Y), en grados. Círculo rojo: |σ|; flecha verde: |R|. "
           + (f"Se omiten {omitidos} paneles (máximo {max_p}). " if omitidos else "")
           + "\n" + _nota_calibracion(int(global_["sesion_id"].nunique())))
    _pie_texto(fig, config, pie, 0.85 / alto_total)
    return ResultadoErgonomia(fig, titulo, pie, len(global_))


def _figura_histogramas(datos, config, alcance) -> ResultadoErgonomia:
    ce, cs = config["ergonomia"], config["estilo"]
    n = len(datos)
    titulo = f"Histogramas de frecuencia por zona · {alcance} · n = {n}"
    fig = plt.figure(figsize=tuple(ce["tamanos_pulgadas"]["histogramas"]))
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.84, bottom=0.34, wspace=0.15)
    textos = []
    for i, (eje, nombre) in enumerate((("frontal", "Frontal α (flexión / extensión)"),
                                       ("lateral", "Lateral β (inclinación)"))):
        ax = fig.add_subplot(gs[0, i])
        v = datos[eje].to_numpy(dtype=float)
        lim = _vista(config, v)
        if n:
            _hist_por_zona(ax, v, eje, config, lim, vertical=True)
            zonas = erg.clasificar_eje(v, eje, config)
            pct = {z: 100 * float((zonas == z).sum()) / n for z in erg.NIVELES}
            ax.legend(handles=_leyenda_zonas(config, pct, "colores_barra"), loc="upper right", fontsize=cs["tamano_pie"],
                      framealpha=0.92, edgecolor=cs["color_malla"])
            lims = ce["zonas_1d"][eje]
            textos.append(f"{nombre.split(' (')[0]}: óptima {lims['optima'][0]}° a {lims['optima'][1]}°, "
                          f"aceptable {lims['aceptable'][0]}° a {lims['aceptable'][1]}°, extrema fuera")
        else:
            ax.text(0.5, 0.5, "Sin muestras", transform=ax.transAxes, ha="center", va="center",
                    color=cs["color_texto_secundario"])
        ax.set_xlim(lim)
        ax.set_title(nombre, fontsize=cs["tamano_rotulo"])
        ax.set_xlabel("Ángulo (°)")
        if i == 0:
            ax.set_ylabel("Muestras (≈ segundos)")
        ax.grid(axis="x", visible=False)
    fig.suptitle(titulo, x=0.06, ha="left", fontsize=cs["tamano_titulo"], fontweight="bold")
    pie = (f"Barras de 1° (resolución del sensor) coloreadas según la zona de cada ángulo por separado ({ce['fuente']}). "
           + ("; ".join(textos) + ". " if textos else "")
           + "\n" + _nota_calibracion(int(datos["sesion_id"].nunique()) if n else 0))
    _pie_texto(fig, config, pie, 0.17)
    return ResultadoErgonomia(fig, titulo, pie, n)


# ---------------------------------------------------------------- entrada

def figura_ergonomia(mediciones: pd.DataFrame, config: dict, figura: str = "zonas",
                     usuario: str | None = None, sesion: str | None = None) -> ResultadoErgonomia:
    if figura not in FIGURAS:
        raise ValueError(f"Figura desconocida: {figura!r} (use {', '.join(FIGURAS)})")
    datos, global_ = erg.seleccionar(mediciones, config, usuario, sesion)
    por_defecto = config["usuarios"]["por_defecto"]
    partes = []
    if usuario:
        partes.append("persona " + ("Sin identificar" if usuario == por_defecto else usuario))
    if sesion:
        partes.append("sesión " + sesion)
    alcance = " · ".join(partes) or "conjunto completo"
    with estilo.aplicar(config):
        if figura == "zonas":
            return _figura_zonas(datos, global_, config, alcance)
        if figura == "momentos":
            return _figura_momentos(datos, config, alcance)
        if figura == "sujetos":
            return _figura_sujetos(global_, config, por_defecto)
        return _figura_histogramas(datos, config, alcance)
