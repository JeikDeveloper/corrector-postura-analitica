"""Figuras e informe de confiabilidad (sección 4.5)."""

from __future__ import annotations

import io
import math
import textwrap

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from .. import confiabilidad as co
from . import estilo
from .estilo import plt

FIGURAS = ("fallos", "weibull", "anomalias", "prediccion")
COLOR_TIPO = {"Pérdida de conexión": "#c2410c", "Conexión fallida": "#7a4fa3", "Hueco de datos": "#b07d00",
              "Sesión sin cierre": "#59636e", "Episodio MAL": "#c2410c"}
COLOR_SEV = {"critica": "#a40e26", "alta": "#d1242f", "media": "#d9a400", "normal": "#9aa4af"}
NOMBRE_SEV = {"critica": "Crítica", "alta": "Alta", "media": "Media", "normal": "Normal"}
GRIS = "#9aa4af"


def _tam(config, clave):
    return tuple(config["confiabilidad"]["tamanos_pulgadas"][clave])


def _cerrar(fig, config, titulo, pie, y=0.1):
    ce = config["estilo"]
    fig.suptitle(titulo, x=0.06, ha="left", fontsize=ce["tamano_titulo"], fontweight="bold")
    fig.text(0.06, y, "\n".join(textwrap.fill(t, 150) for t in pie.split("\n")), ha="left", va="top",
             fontsize=ce["tamano_pie"], color=ce["color_texto_secundario"])


def _mensaje(ax, config, texto):
    ax.text(0.5, 0.5, "\n".join(textwrap.wrap(texto, 60)), transform=ax.transAxes, ha="center", va="center",
            color=config["estilo"]["color_texto_secundario"], fontsize=config["estilo"]["tamano_base"])
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(False)


def _modo(a):
    return "fallos del dispositivo" if a["modo"] == "dispositivo" else "fallos posturales (episodios MAL)"


# ---------------------------------------------------------------- fallos en el tiempo

def _fallos(a, config):
    f, seg = a["_fallos"], a["_segmentos"]
    fig = plt.figure(figsize=_tam(config, "fallos"))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], left=0.17, right=0.97, top=0.84, bottom=0.27, wspace=0.3)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    archivos = list(dict.fromkeys(seg["archivo"])) if len(seg) else []
    for i, arch in enumerate(archivos):
        sg = seg[seg["archivo"] == arch]
        for s in sg.itertuples():
            ax.barh(i, s.operacion_s / 60, left=s.reloj_inicio_s / 60, height=0.35, color="#dfe3e8", edgecolor="#c9ced4")
        ff = f[f["archivo"] == arch]
        for fila in ff.itertuples():
            ax.plot(fila.reloj_s / 60, i, marker="v", markersize=9, color=COLOR_TIPO[co.TIPOS[fila.tipo]], zorder=3)
    ax.set_yticks(range(len(archivos)), archivos, fontsize=config["estilo"]["tamano_pie"] + 0.5)
    ax.set_ylim(-0.7, max(len(archivos), 1) - 0.3)
    ax.set_xlabel("Minutos de operación acumulados" + (" (con datos)" if a["modo"] == "dispositivo" else " (en BIEN)"))
    ax.grid(axis="y", visible=False)
    tipos = list(dict.fromkeys(co.TIPOS[t] for t in f["tipo"])) if len(f) else []
    ax.legend(handles=[Line2D([], [], marker="v", linestyle="", color=COLOR_TIPO[t], label=t) for t in tipos]
              + [Line2D([], [], color="#dfe3e8", linewidth=8, label="Tiempo de operación")],
              fontsize=config["estilo"]["tamano_pie"], loc="upper right", framealpha=0.9)
    m = a["metricas"]
    if m["por_tipo"]:
        nombres = list(m["por_tipo"])
        ax2.barh(range(len(nombres)), [m["por_tipo"][n] for n in nombres], color=[COLOR_TIPO[n] for n in nombres])
        ax2.set_yticks(range(len(nombres)), nombres, fontsize=config["estilo"]["tamano_pie"] + 0.5)
        ax2.set_xlabel("Fallos")
        ax2.grid(axis="y", visible=False)
    else:
        _mensaje(ax2, config, "Sin fallos en la selección.")
    _cerrar(fig, config, f"Fallos en el tiempo · {_modo(a)} · {m['n_fallos']} fallos",
            f"MTBF {m['mtbf_min'] or '—'} min · MTTR {m['mttr_s'] or '—'} s · disponibilidad {m['disponibilidad_pct'] or '—'} % · "
            f"tasa de fallos {m['tasa_fallos_h'] or '—'} por hora. Cada triángulo es un fallo sobre el reloj de operación de su "
            "archivo (barras grises).", 0.13)
    return fig


# ---------------------------------------------------------------- Weibull

def _weibull(a, config):
    w = a["weibull"]
    fig = plt.figure(figsize=_tam(config, "weibull"))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.84, bottom=0.27, wspace=0.25)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    obs = np.sort(np.array(w["observados_min"], float))
    if not w["ajustado"]:
        if obs.size:
            ax.plot(obs, np.zeros_like(obs), "v", color="#c2410c", markersize=9, label="Tiempo entre fallos observado")
            cens = np.array(w["censurados_min"], float)
            ax.plot(cens, np.zeros_like(cens), "o", mfc="none", color=GRIS, markersize=8, label="Tramo final sin fallo (censurado)")
            ax.set_yticks([])
            ax.set_xlabel("Minutos de operación")
            ax.legend(fontsize=config["estilo"]["tamano_pie"])
            ax.set_title("Tiempos disponibles", fontsize=config["estilo"]["tamano_rotulo"], loc="left")
        else:
            _mensaje(ax, config, "No hay tiempos entre fallos en la selección.")
        _mensaje(ax2, config, f"Weibull no ajustado. {w['motivo']}")
        _cerrar(fig, config, f"Weibull · {_modo(a)} · no ajustable con estos datos",
                "El ajuste necesita más fallos observados; con más registros esta figura muestra el gráfico de probabilidad "
                "de Weibull y la curva de confiabilidad con su proyección.", 0.13)
        return fig
    b, e, h = w["beta"], w["eta_min"], float(w["horizonte_min"])
    n = obs.size
    rangos = (np.arange(1, n + 1) - 0.3) / (n + 0.4)            # rangos medianos de Bernard
    ax.plot(np.log(obs), np.log(-np.log(1 - rangos)), "o", color="#3b528b", label="Fallos observados")
    xs = np.linspace(np.log(obs.min()) - 0.5, np.log(max(obs.max(), h)) + 0.5, 50)
    ax.plot(xs, b * (xs - math.log(e)), color="#c2410c", label=f"Ajuste: β = {b}, η = {e} min")
    ax.set_xlabel("ln(tiempo entre fallos, min)")
    ax.set_ylabel("ln(−ln(1 − F))")
    ax.legend(fontsize=config["estilo"]["tamano_pie"])
    ax.set_title("Gráfico de probabilidad de Weibull", fontsize=config["estilo"]["tamano_rotulo"], loc="left")
    t = np.linspace(0, max(obs.max(), 2 * h, w["mttf_min"] * 2), 300)
    R = np.exp(-(t / e) ** b)
    ax2.plot(t, R, color="#3b528b", label="Confiabilidad R(t)")
    ax2.fill_between(t, 0, R, where=t <= h, color="#c2410c", alpha=0.12, label=f"Horizonte de {h:g} min")
    ax2.axvline(w["b10_min"], color="#b07d00", linestyle="--", label=f"B10 = {w['b10_min']} min")
    ax2.axvline(w["mttf_min"], color="#2f8f46", linestyle=":", label=f"Tiempo medio = {w['mttf_min']} min")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Minutos de operación desde el último fallo")
    ax2.set_ylabel("Probabilidad de no haber fallado")
    ax2.legend(fontsize=config["estilo"]["tamano_pie"])
    ax2.set_title(f"Proyección: {100 * w['prob_fallo_horizonte']:.1f} % de fallar en {h:g} min",
                  fontsize=config["estilo"]["tamano_rotulo"], loc="left")
    _cerrar(fig, config, f"Weibull · {_modo(a)} · {n} fallos, {w['n_censurados']} tramos censurados",
            f"{w['lectura']} Ajuste por máxima verosimilitud con censura por la derecha (los tramos finales sin fallo cuentan "
            "como «sobrevivió al menos hasta aquí»). B10: tiempo en que falla el 10 %.", 0.13)
    return fig


# ---------------------------------------------------------------- anomalías

def _anomalias(a, config):
    an = a["anomalias"]
    fig = plt.figure(figsize=_tam(config, "anomalias"))
    ax = fig.add_axes([0.07, 0.3, 0.88, 0.55])
    if not an["aplicado"]:
        _mensaje(ax, config, f"Isolation Forest no aplicado. {an['motivo']}")
        _cerrar(fig, config, "Anomalías (Isolation Forest)", "Sin ventanas suficientes.", 0.15)
        return fig
    vs = an["ventanas"]
    x = np.arange(len(vs))
    ax.scatter(x, [v["puntaje"] for v in vs], c=[COLOR_SEV[v["severidad"]] for v in vs], s=40, zorder=3)
    for nivel in ("media", "alta", "critica"):
        ax.axhline(an["umbrales"][nivel], color=COLOR_SEV[nivel], linestyle="--", linewidth=1,
                   label=f"{NOMBRE_SEV[nivel]} ≥ {an['umbrales'][nivel]}")
    cambios = [i for i in range(1, len(vs)) if vs[i]["sesion_id"] != vs[i - 1]["sesion_id"]]
    for c in cambios:
        ax.axvline(c - 0.5, color="#c9ced4", linewidth=0.8)
    for i, v in enumerate(vs):
        if v["severidad"] in ("alta", "critica"):
            ax.annotate(f"{v['sesion_id'][-12:]} · min {v['inicio_s'] / 60:.1f}", (i, v["puntaje"]), xytext=(4, 6),
                        textcoords="offset points", fontsize=6.5, color=COLOR_SEV[v["severidad"]])
    ax.set_xlabel("Ventanas (en orden de archivo y sesión; las líneas grises separan sesiones)")
    ax.set_ylabel("Puntaje de anomalía")
    ax.legend(fontsize=config["estilo"]["tamano_pie"], loc="upper right")
    c = an["conteo"]
    _cerrar(fig, config, f"Anomalías (Isolation Forest) · {len(vs)} ventanas · {c['critica']} críticas, {c['alta']} altas, {c['media']} medias",
            "Puntaje de anomalía de Isolation Forest (0 a 1): cerca de 0,5 es normal; más alto, más atípica la ventana respecto "
            "a las demás. Cada ventana se describe con 12 características de calidad de datos y de comportamiento angular. "
            "Semilla fija: el mismo archivo da siempre el mismo resultado.", 0.15)
    return fig


# ---------------------------------------------------------------- predicción

def _prediccion(a, config):
    pr = a["prediccion"]
    fig = plt.figure(figsize=_tam(config, "prediccion"))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.84, bottom=0.27, wspace=0.35)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cp = config["confiabilidad"]["prediccion"]
    if not pr["entrenado"]:
        x = [0, 1]
        ax.bar(x, [pr["positivos"], pr["negativos"]], color=["#c2410c", GRIS])
        ax.plot([-0.4, 0.4], [cp["min_positivos"]] * 2, color="#1f2328", linestyle="--")
        ax.plot([0.6, 1.4], [cp["min_negativos"]] * 2, color="#1f2328", linestyle="--")
        ax.set_xticks(x, [f"Seguidas de un fallo\n(mínimo {cp['min_positivos']})", f"Sin fallo\n(mínimo {cp['min_negativos']})"])
        ax.set_ylabel("Ventanas")
        ax.set_title("Datos disponibles para entrenar", fontsize=config["estilo"]["tamano_rotulo"], loc="left")
        _mensaje(ax2, config, f"Modelo no entrenado. {pr['motivo']}")
        _cerrar(fig, config, f"Predicción de fallo inminente (regresión logística) · no entrenable con estos datos",
                f"Etiqueta de cada ventana: si empieza un fallo en los {pr['horizonte_s']} s siguientes. Línea punteada: mínimo "
                "configurado para entrenar.", 0.13)
        return fig
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(np.array(pr["etiquetas"]), np.array(pr["prob_validacion"], float))
    ax.plot(fpr, tpr, color="#3b528b", label="Validación cruzada")
    ax.plot([0, 1], [0, 1], color=GRIS, linestyle="--", label="Azar")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdaderos positivos")
    ax.set_title(f"Curva ROC · AUC validación cruzada = {pr['auc']}", fontsize=config["estilo"]["tamano_rotulo"], loc="left")
    ax.legend(fontsize=config["estilo"]["tamano_pie"])
    coef = pr["coeficientes"][::-1]
    yy = np.arange(len(coef))
    ax2.barh(yy, [c["coef"] for c in coef], color=["#c2410c" if c["coef"] > 0 else "#3b528b" for c in coef])
    ax2.set_yticks(yy, [c["caracteristica"] for c in coef], fontsize=config["estilo"]["tamano_pie"] + 0.5)
    ax2.axvline(0, color="#1f2328", linewidth=0.8)
    ax2.set_xlabel("Coeficiente (por desviación estándar)  ← protege · aumenta el riesgo →")
    ax2.grid(axis="y", visible=False)
    _cerrar(fig, config, f"Predicción de fallo inminente (regresión logística) · {pr['positivos']} positivos, {pr['negativos']} negativos",
            f"Etiqueta: empieza un fallo en los {pr['horizonte_s']} s siguientes a la ventana. AUC estimado con validación cruzada "
            f"de {pr['pliegues']} pliegues: cada ventana se evalúa con un modelo que no la vio. AUC 0,5 = azar; 1 = perfecto. Sensibilidad {pr['sensibilidad']}, "
            f"especificidad {pr['especificidad']} con umbral {pr['umbral_alerta']}. Variables estandarizadas; clases balanceadas.", 0.13)
    return fig


_DIBUJOS = {"fallos": _fallos, "weibull": _weibull, "anomalias": _anomalias, "prediccion": _prediccion}


def figura_confiabilidad(a: dict, figura: str, config: dict):
    if figura not in FIGURAS:
        raise ValueError(f"Figura desconocida: {figura!r} (use {', '.join(FIGURAS)})")
    with estilo.aplicar(config):
        return _DIBUJOS[figura](a, config)


def informe_pdf(a: dict, config: dict, alcance: str) -> bytes:
    m, w, an, pr = a["metricas"], a["weibull"], a["anomalias"], a["prediccion"]
    lineas = [
        ("Confiabilidad y tasa de fallos", 16, True),
        (f"Lectura de fallo: {_modo(a)} · selección: {alcance}", 10, False),
        ("", 6, False),
        ("Métricas", 12, True),
        (f"Fallos: {m['n_fallos']} ({', '.join(f'{k}: {v}' for k, v in m['por_tipo'].items()) or 'ninguno'}) · "
         f"tiempo de operación: {m['operacion_min']} min", 10, False),
        (f"MTBF: {m['mtbf_min'] or '—'} min · MTTR: {m['mttr_s'] or '—'} s ({m['reparaciones_conocidas']} reparaciones conocidas) · "
         f"disponibilidad: {m['disponibilidad_pct'] or '—'} % · tasa de fallos: {m['tasa_fallos_h'] or '—'} por hora", 10, False),
        ("", 6, False),
        ("Weibull", 12, True),
        ((f"β = {w['beta']}, η = {w['eta_min']} min, tiempo medio = {w['mttf_min']} min, B10 = {w['b10_min']} min. "
          f"Probabilidad de fallar en {w['horizonte_min']} min: {100 * w['prob_fallo_horizonte']:.1f} %. {w['lectura']}")
         if w["ajustado"] else f"No ajustado. {w['motivo']}", 10, False),
        ("", 6, False),
        ("Anomalías (Isolation Forest)", 12, True),
        ((f"{len(an['ventanas'])} ventanas: {an['conteo']['critica']} críticas, {an['conteo']['alta']} altas, "
          f"{an['conteo']['media']} medias, {an['conteo']['normal']} normales.") if an["aplicado"] else f"No aplicado. {an['motivo']}", 10, False),
        ("", 6, False),
        ("Predicción de fallo inminente (regresión logística)", 12, True),
        ((f"AUC (validación cruzada) {pr['auc']}, sensibilidad {pr['sensibilidad']}, especificidad {pr['especificidad']}. "
          f"Variables con más peso: {', '.join(c['caracteristica'] for c in pr['coeficientes'][:3])}.")
         if pr["entrenado"] else f"No entrenado. {pr['motivo']}", 10, False),
        ("", 6, False),
        ("Alertas", 12, True),
    ] + ([(f"[{NOMBRE_SEV[x['severidad']]}] {x['origen']}: {x['texto']}", 9, False) for x in a["alertas"][:14]]
         or [("Sin alertas.", 10, False)])
    buf = io.BytesIO()
    with estilo.aplicar(config), PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        y = 0.95
        for texto, tam, negrita in lineas:
            partes = textwrap.wrap(texto, 120) or [""]
            fig.text(0.06, y, "\n".join(partes), fontsize=tam, fontweight="bold" if negrita else "normal", va="top")
            y -= 0.02 * len(partes) * (tam / 9) + 0.008
        pdf.savefig(fig)
        plt.close(fig)
        for nombre in FIGURAS:
            f = figura_confiabilidad(a, nombre, config)
            pdf.savefig(f)
            plt.close(f)
        pdf.infodict()["Title"] = "Confiabilidad y tasa de fallos"
    return buf.getvalue()
