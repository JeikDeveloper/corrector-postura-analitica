"""Figuras del análisis descriptivo de patrones angulares (BIEN vs MAL).

Todas son deterministas: no usan azar (sin dispersión aleatoria de puntos), así
que los mismos datos producen siempre la misma imagen.

- ``histogramas``: distribución de F y L, BIEN y MAL superpuestos, barras de 1°.
- ``cajas``: box plots de F y L para Todos, BIEN y MAL (bigotes a 1,5 × RIC).
- ``violin``: violin plots de F y L para Todos, BIEN y MAL, con mediana y cuartiles.
- ``series``: F y L en el tiempo, una fila por sesión, con los tramos MAL sombreados.

``reporte_pdf`` arma un PDF de varias páginas con las tablas y las cuatro figuras.
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .. import estadistica as est
from . import estilo
from .densidad import _alcance, _bordes_enteros, _como_lista, seleccion
from .estilo import plt

FIGURAS = ("histogramas", "cajas", "violin", "series")
NOMBRES_ANGULO = {"frontal": "Frontal F (°)  ← atrás · adelante →", "lateral": "Lateral L (°)  ← izquierda · derecha →"}


@dataclass
class ResultadoEstadistica:
    figura: Figure
    titulo: str
    pie: str
    n_muestras: int


# ---------------------------------------------------------------- piezas comunes

def _umbrales(config: dict, angulo: str) -> tuple[float, float]:
    u = config["umbrales"]
    return (u["FRONTAL_MIN"], u["FRONTAL_MAX"]) if angulo == "frontal" else (u["LATERAL_MIN"], u["LATERAL_MAX"])


def _pie(fig, config: dict, texto: str, y: float) -> None:
    ce = config["estilo"]
    ancho = int(config["figura"]["ancho_pie_caracteres"])
    fig.text(0.06, y, "\n".join(textwrap.fill(p, width=ancho) for p in texto.split("\n")), ha="left", va="top",
             fontsize=ce["tamano_pie"], color=ce["color_texto_secundario"])


def _nota_comun(datos: pd.DataFrame, n_pre: int, n_sc: int) -> str:
    n_ses = int(datos["sesion_id"].nunique()) if len(datos) else 0
    texto = (f"Estado BIEN/MAL: el que reporta el dispositivo en cada línea. Solo muestras posteriores al CALIBRADO; "
             f"excluidas {n_pre} previas a la calibración" + (f" y {n_sc} de sesiones sin calibrar" if n_sc else "") + ".")
    if n_ses > 1:
        texto += (f" AGREGACIÓN: combina {n_ses} sesiones; cada una tiene su propio cero por la recalibración, así que "
                  "los ángulos no son comparables en términos absolutos entre sesiones.")
    return texto


def _grupos(datos: pd.DataFrame, angulo: str) -> list[tuple[str, np.ndarray]]:
    return [("Todos", datos[angulo].to_numpy(float))] + [
        (e, datos.loc[datos["estado"] == e, angulo].to_numpy(float)) for e in est.ESTADOS]


def _color_grupo(config: dict, grupo: str) -> str:
    return config["estadistica"]["colores_estado"].get(grupo, config["estilo"]["color_histograma"])


def _lineas_umbral(ax, config: dict, angulo: str, vertical: bool) -> None:
    ce, cf = config["estilo"], config["figura"]
    for v in _umbrales(config, angulo):
        kw = dict(color=ce["color_umbral"], linewidth=1.0, linestyle=(0, tuple(cf["umbral_rectangulo"]["patron"])), zorder=1)
        (ax.axvline if vertical else ax.axhline)(v, **kw)


# ---------------------------------------------------------------- figuras

def _histogramas(datos, config, alcance, nota):
    cs, ce = config["estadistica"], config["estilo"]
    n = len(datos)
    fig = plt.figure(figsize=tuple(cs["tamanos_pulgadas"]["histogramas"]))
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.84, bottom=0.33, wspace=0.16)
    for i, angulo in enumerate(est.ANGULOS):
        ax = fig.add_subplot(gs[0, i])
        v = datos[angulo].to_numpy(float)
        lo, hi = _umbrales(config, angulo)
        lim = (min(lo, v.min() if n else lo) - 3, max(hi, v.max() if n else hi) + 3)
        bordes = _bordes_enteros(lim, float(cs["ancho_bin_grados"]))
        for e in est.ESTADOS:
            sub = datos.loc[datos["estado"] == e, angulo].to_numpy(float)
            if sub.size:
                ax.hist(sub, bins=bordes, color=_color_grupo(config, e), alpha=float(cs["alfa_histograma"]),
                        edgecolor=ce["color_fondo"], linewidth=0.3,
                        label=f"{e} · n = {sub.size} · media {sub.mean():.1f}°")
        _lineas_umbral(ax, config, angulo, vertical=True)
        ax.plot([], [], color=ce["color_umbral"], linestyle=(0, (4, 3)), label=f"Umbrales ({lo}° y {hi}°)")
        if not n:
            ax.text(0.5, 0.5, "Sin muestras", transform=ax.transAxes, ha="center", color=ce["color_texto_secundario"])
        ax.set_xlim(lim)
        ax.set_xlabel(NOMBRES_ANGULO[angulo])
        ax.set_ylabel("Muestras (≈ segundos)" if i == 0 else "")
        ax.legend(fontsize=ce["tamano_pie"], loc="upper right", framealpha=0.92, edgecolor=ce["color_malla"])
        ax.grid(axis="x", visible=False)
    titulo = f"Distribución de los ángulos, BIEN vs MAL · {alcance} · n = {n}"
    pie = f"Barras de {cs['ancho_bin_grados']}° (resolución del sensor); BIEN y MAL superpuestos con transparencia.\n{nota}"
    return fig, titulo, pie, 0.17


def _cajas(datos, config, alcance, nota, violin=False):
    cs, ce = config["estadistica"], config["estilo"]
    n = len(datos)
    fig = plt.figure(figsize=tuple(cs["tamanos_pulgadas"]["violin" if violin else "cajas"]))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.84, bottom=0.3, wspace=0.22)
    for i, angulo in enumerate(est.ANGULOS):
        ax = fig.add_subplot(gs[0, i])
        grupos = _grupos(datos, angulo)
        pos = np.arange(1, len(grupos) + 1)
        for p, (g, v) in zip(pos, grupos):
            color = _color_grupo(config, g)
            if not v.size:
                ax.text(p, 0, "sin\nmuestras", ha="center", va="center", fontsize=ce["tamano_pie"],
                        color=ce["color_texto_secundario"])
                continue
            if violin and v.size >= int(cs["min_muestras_violin"]) and np.ptp(v) > 0:
                partes = ax.violinplot([v], positions=[p], widths=0.8, showextrema=False, showmedians=False)
                for cuerpo in partes["bodies"]:
                    cuerpo.set_facecolor(color)
                    cuerpo.set_edgecolor(color)
                    cuerpo.set_alpha(0.45)
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.vlines(p, q1, q3, color=ce["color_texto"], linewidth=4, zorder=3)
                ax.scatter([p], [med], color="#ffffff", edgecolor=ce["color_texto"], s=28, zorder=4)
                ax.vlines(p, v.min(), v.max(), color=ce["color_texto"], linewidth=0.8, zorder=2)
            elif violin:   # pocas muestras o sin variación: se muestran los valores, no una densidad
                ax.scatter(np.full(v.size, p), v, color=color, s=18, zorder=3)
            else:
                ax.boxplot([v], positions=[p], widths=0.55, patch_artist=True, showmeans=True, whis=1.5,
                           boxprops=dict(facecolor=color, alpha=0.5, edgecolor=ce["color_texto"]),
                           medianprops=dict(color=ce["color_texto"], linewidth=1.6),
                           meanprops=dict(marker="D", markerfacecolor="#ffffff", markeredgecolor=ce["color_texto"], markersize=5),
                           whiskerprops=dict(color=ce["color_texto"]), capprops=dict(color=ce["color_texto"]),
                           flierprops=dict(marker="o", markersize=3, markerfacecolor=color, markeredgecolor="none", alpha=0.7))
        _lineas_umbral(ax, config, angulo, vertical=False)
        ax.set_xticks(pos, [f"{g}\nn = {v.size}" for g, v in grupos])
        ax.set_xlim(0.4, len(grupos) + 0.6)
        ax.set_ylabel(NOMBRES_ANGULO[angulo])
        ax.grid(axis="x", visible=False)
    if violin:
        titulo = f"Violin plots de los ángulos, BIEN vs MAL · {alcance} · n = {n}"
        pie = ("Ancho: densidad kernel de cada grupo (cada violín normalizado a su propio máximo). Barra negra: rango "
               "intercuartílico (P25–P75); punto blanco: mediana; línea fina: mínimo a máximo. Con menos de "
               f"{cs['min_muestras_violin']} muestras se dibujan los valores uno a uno en lugar de una densidad. "
               f"Líneas punteadas: umbrales de alerta.\n{nota}")
    else:
        titulo = f"Box plots de los ángulos, BIEN vs MAL · {alcance} · n = {n}"
        pie = ("Caja: P25 a P75; línea: mediana; rombo: media; bigotes hasta 1,5 × el rango intercuartílico; puntos: "
               f"valores más allá de los bigotes. Líneas punteadas: umbrales de alerta.\n{nota}")
    return fig, titulo, pie, 0.15


def _series(datos, config, alcance, nota):
    cs, ce = config["estadistica"], config["estilo"]
    hueco = float(config["calidad"]["umbral_hueco_s"])
    cadencia = float(config["calidad"]["cadencia_nominal_s"])
    sesiones = list(dict.fromkeys(datos.sort_values("t")["sesion_id"].astype(str)))
    maximo = int(cs["max_sesiones_serie"])
    omitidas = max(0, len(sesiones) - maximo)
    sesiones = sesiones[:maximo]
    filas = max(1, len(sesiones))
    ancho, alto_fila = cs["tamanos_pulgadas"]["serie_fila"]
    alto = alto_fila * filas + 2.4
    fig = plt.figure(figsize=(ancho, alto))
    gs = fig.add_gridspec(filas, 1, left=0.07, right=0.98, top=1 - 0.85 / alto, bottom=2.0 / alto, hspace=0.55)
    rachas = est.rachas(datos, cadencia)
    color_mal = cs["colores_estado"]["MAL"]
    if not sesiones:
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.5, 0.5, "Sin muestras analizables para este filtro", transform=ax.transAxes, ha="center",
                color=ce["color_texto_secundario"])
        ax.set_axis_off()
    for i, sid in enumerate(sesiones):
        ax = fig.add_subplot(gs[i, 0])
        ses = datos[datos["sesion_id"].astype(str) == sid].sort_values("t")
        t = ses["t_rel_calibracion"].to_numpy(float) / 60.0
        corte = np.r_[False, np.diff(ses["t_rel_calibracion"].to_numpy(float)) > hueco]
        for angulo in est.ANGULOS:
            y = ses[angulo].to_numpy(float).copy()
            tt = t.copy()
            if corte.any():   # se corta la línea en los huecos: no se interpola entre ellos
                idx = np.flatnonzero(corte)
                tt = np.insert(tt, idx, np.nan)
                y = np.insert(y, idx, np.nan)
            ax.plot(tt, y, color=cs["colores_angulo"][angulo], linewidth=1.0, label=angulo.capitalize())
        for _, r in rachas[(rachas["sesion_id"] == sid) & (rachas["estado"] == "MAL")].iterrows():
            ax.axvspan(r["inicio_s"] / 60.0, (r["inicio_s"] + r["duracion_s"]) / 60.0, color=color_mal, alpha=0.18,
                       linewidth=0, zorder=0)
        ax.axhline(0, color=ce["color_malla"], linewidth=0.8, zorder=0)
        pct = 100 * (ses["estado"] == "MAL").mean()
        persona = str(ses["usuario_id"].iloc[0])
        ax.set_title(f"{sid} · {persona} · n = {len(ses)} · MAL {pct:.1f} %", fontsize=ce["tamano_rotulo"], loc="left")
        ax.set_ylabel("Ángulo (°)")
        if i == len(sesiones) - 1:
            ax.set_xlabel("Minutos desde el CALIBRADO de la sesión")
    if sesiones:
        manejadores = [plt.Line2D([], [], color=cs["colores_angulo"][a], label=n)
                       for a, n in (("frontal", "Frontal F"), ("lateral", "Lateral L"))]
        manejadores.append(Patch(facecolor=color_mal, alpha=0.18, label="Tramos MAL"))
        fig.legend(handles=manejadores, loc="upper right", ncol=3, fontsize=ce["tamano_pie"], frameon=False,
                   bbox_to_anchor=(0.98, 1 - 0.35 / alto))
    titulo = f"Series temporales de los ángulos · {alcance} · n = {len(datos)}"
    pie = ("Una fila por sesión. Sombreado: tramos consecutivos con estado MAL. La línea se corta en los huecos "
           f"mayores a {hueco:g} s; no se interpola."
           + (f" Se omiten {omitidas} sesiones (máximo {maximo})." if omitidas else "") + f"\n{nota}")
    return fig, titulo, pie, 1.35 / alto


def figura_estadistica(mediciones: pd.DataFrame, config: dict, figura: str = "histogramas",
                       usuario: str | None = None, sesion: str | None = None) -> ResultadoEstadistica:
    if figura not in FIGURAS:
        raise ValueError(f"Figura desconocida: {figura!r} (use {', '.join(FIGURAS)})")
    usuarios, sesiones = _como_lista(usuario), _como_lista(sesion)
    datos, n_pre, n_sc = seleccion(mediciones, config, usuarios, sesiones)
    alcance = _alcance(usuarios, sesiones)
    nota = _nota_comun(datos, n_pre, n_sc)
    with estilo.aplicar(config):
        if figura == "histogramas":
            fig, titulo, pie, y = _histogramas(datos, config, alcance, nota)
        elif figura == "series":
            fig, titulo, pie, y = _series(datos, config, alcance, nota)
        else:
            fig, titulo, pie, y = _cajas(datos, config, alcance, nota, violin=figura == "violin")
        fig.suptitle(titulo, x=0.06, ha="left", fontsize=config["estilo"]["tamano_titulo"], fontweight="bold")
        _pie(fig, config, pie, y)
    return ResultadoEstadistica(fig, titulo, pie, len(datos))


# ---------------------------------------------------------------- reporte

def _tabla(ax, encabezado, filas, config, anchos=None):
    ce = config["estilo"]
    ax.set_axis_off()
    t = ax.table(cellText=filas, colLabels=encabezado, loc="upper left", cellLoc="right", colLoc="right",
                 colWidths=anchos)
    t.auto_set_font_size(False)
    t.set_fontsize(ce["tamano_pie"] + 0.5)
    t.scale(1, 1.25)
    for (fila, col), celda in t.get_celld().items():
        celda.set_edgecolor(ce["color_malla"])
        if fila == 0:
            celda.set_text_props(fontweight="bold", color=ce["color_texto_secundario"])
        if col == 0:
            celda.set_text_props(ha="left")
            celda._loc = "left"


def reporte_pdf(mediciones: pd.DataFrame, config: dict, usuario: str | None = None,
                sesion: str | None = None) -> bytes:
    """PDF de varias páginas: resumen y tablas, luego las cuatro figuras."""
    usuarios, sesiones = _como_lista(usuario), _como_lista(sesion)
    datos, n_pre, n_sc = seleccion(mediciones, config, usuarios, sesiones)
    a = est.analisis(datos, config)
    alcance = _alcance(usuarios, sesiones)
    ce = config["estilo"]
    num = lambda x, s="": "—" if x is None else f"{x}{s}"   # noqa: E731
    buf = io.BytesIO()
    with estilo.aplicar(config), PdfPages(buf) as pdf:
        fig = plt.figure(figsize=tuple(config["estadistica"]["tamanos_pulgadas"]["reporte_pagina"]))
        fig.text(0.05, 0.95, "Análisis descriptivo de patrones angulares", fontsize=16, fontweight="bold")
        fig.text(0.05, 0.915, f"{alcance} · {a['n']} muestras · {a['n_sesiones']} sesiones · {a['n_personas']} "
                 f"persona{'s' if a['n_personas'] != 1 else ''} · MAL {num(a['pct_mal'], ' %')}",
                 fontsize=ce["tamano_base"] + 1, color=ce["color_texto_secundario"])
        p = [f"P{int(x) if float(x).is_integer() else x}" for x in a["percentiles"]]
        filas = [[f"{r['angulo'].capitalize()} · {r['grupo']}", r["n"], num(r["media"]), num(r["sd"]), num(r["min"]),
                  *[num(r["percentiles"][k]) for k in r["percentiles"]], num(r["max"]), num(r["iqr"]),
                  num(r["asimetria"]), num(r["curtosis"])] for r in a["descriptivos"]]
        _tabla(fig.add_axes([0.05, 0.52, 0.9, 0.36]),
               ["Ángulo · grupo", "n", "Media", "DE", "Mín", *p, "Máx", "RIC", "Asim.", "Curt."], filas, config)
        ep = a["episodios"]
        _tabla(fig.add_axes([0.05, 0.27, 0.42, 0.18]),
               ["Episodios", "Cantidad", "Media (s)", "Mediana (s)", "Máx (s)", "Total (s)"],
               [[e, ep[e]["n_episodios"], num(ep[e]["duracion_media_s"]), num(ep[e]["duracion_mediana_s"]),
                 num(ep[e]["duracion_max_s"]), num(ep[e]["tiempo_total_s"])] for e in est.ESTADOS], config)
        _tabla(fig.add_axes([0.52, 0.22, 0.43, 0.23]), ["Causa de las muestras MAL", "n", "%"],
               [[c["descripcion"], c["n"], num(c["pct"], " %")] for c in a["causas_mal"]], config, [0.72, 0.12, 0.16])
        fig.text(0.05, 0.12, "\n".join(textwrap.wrap(
            "DE: desviación estándar muestral. RIC: rango intercuartílico (P75 − P25). Asim.: asimetría; Curt.: curtosis "
            "en exceso (normal = 0). Episodio: racha de muestras consecutivas con el mismo estado dentro de una sesión. "
            "Una muestra MAL puede exceder dos umbrales a la vez. " + _nota_comun(datos, n_pre, n_sc), 150)),
            fontsize=ce["tamano_pie"], color=ce["color_texto_secundario"], va="top")
        pdf.savefig(fig)
        plt.close(fig)
        for tipo in FIGURAS:
            r = figura_estadistica(mediciones, config, tipo, usuario=usuario, sesion=sesion)
            pdf.savefig(r.figura)
            plt.close(r.figura)
        info = pdf.infodict()
        info["Title"] = "Análisis descriptivo de patrones angulares"
        info["Subject"] = alcance
    return buf.getvalue()
