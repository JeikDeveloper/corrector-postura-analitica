"""Figuras e informe de la analítica diagnóstica (sección 4.4).

- ``narrativa``: el gráfico que acompaña a la narración de datos. Todo en gris salvo lo
  que el relato señala (un color y una anotación), con el titular como título.
  Con varias sesiones compara el tiempo en postura correcta de cada una y resalta la
  peor; con una sola, muestra su línea de tiempo BIEN/MAL y señala el episodio MAL
  más largo y la primera alerta.
- ``indicadores``: los ocho indicadores por sesión.
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from .. import estadistica as est
from ..diagnostica import _n
from .. import indicadores as ind
from . import estilo
from .estilo import plt

FIGURAS = ("narrativa", "indicadores")
GRIS = "#b8bec6"
COLORES = ["#3b528b", "#2f8f46", "#b07d00", "#c2410c", "#7a4fa3", "#59636e", "#1f8a8a", "#a3476b"]


def _titulo(fig, config, titular: str, subtitulo: str) -> None:
    ce = config["estilo"]
    fig.text(0.05, 0.965, "\n".join(textwrap.wrap(titular, 95)), ha="left", va="top",
             fontsize=ce["tamano_titulo"] + 1, fontweight="bold", color=ce["color_texto"])
    fig.text(0.05, 0.86, subtitulo, ha="left", va="top", fontsize=ce["tamano_base"], color=ce["color_texto_secundario"])


def _pie(fig, config, texto: str) -> None:
    ce = config["estilo"]
    fig.text(0.05, 0.06, "\n".join(textwrap.wrap(texto, 150)), ha="left", va="top", fontsize=ce["tamano_pie"],
             color=ce["color_texto_secundario"])


def _narrativa(datos, config, analisis, alcance, con_titulo=True):
    """Con ``con_titulo=False`` (vista en la página, donde el titular ya está arriba) no repite el
    titular ni el valor global, que ya muestra la tarjeta de indicadores."""
    ce = config["estilo"]
    acento = config["estadistica"]["colores_estado"]["MAL"]
    nar, total = analisis["narracion"], analisis["total"]
    ancho, alto = config["diagnostica"]["tamanos_pulgadas"]["narrativa"]
    fig = plt.figure(figsize=(ancho, alto if con_titulo else alto * 0.82))
    ax = fig.add_axes([0.22, 0.2, 0.7, 0.56] if con_titulo else [0.22, 0.22, 0.7, 0.72])
    sesiones = [s for s in analisis["por_sesion"] if s["tiempo_correcto"] is not None]
    if not len(datos):
        ax.text(0.5, 0.5, "Sin mediciones analizables para este filtro", transform=ax.transAxes, ha="center",
                color=ce["color_texto_secundario"])
        ax.set_axis_off()
        if con_titulo:
            _titulo(fig, config, nar["titular"], f"Selección: {alcance}")
        return fig
    if len(sesiones) >= 2:
        orden = sorted(sesiones, key=lambda s: s["tiempo_correcto"])
        y = np.arange(len(orden))
        colores = [acento if s["sesion_id"] == nar["destacada"] else GRIS for s in orden]
        ax.barh(y, [s["tiempo_correcto"] for s in orden], color=colores, height=0.62)
        ax.set_yticks(y, [s["sesion_id"] for s in orden], fontsize=ce["tamano_pie"] + 0.5)
        ax.axvline(total["tiempo_correcto"], color=ce["color_texto"], linestyle="--", linewidth=1)
        if con_titulo:
            ax.text(total["tiempo_correcto"], len(orden) - 0.4, f" selección: {total['tiempo_correcto']} %".replace(".", ","),
                    fontsize=ce["tamano_pie"] + 0.5, va="bottom", ha="center", color=ce["color_texto"])
        ax.set_xlim(0, 112)          # desde 0: recortar el eje exageraría las diferencias
        for yi, s in zip(y, orden):
            fuerte = s["sesion_id"] == nar["destacada"]
            txt = f" {s['tiempo_correcto']} %".replace(".", ",")
            if fuerte and s["alertas_hora"] is not None:
                txt += f" · {s['alertas_hora']} alertas/h".replace(".", ",")
            ax.text(s["tiempo_correcto"], yi, txt, va="center", fontsize=ce["tamano_pie"] + (1 if fuerte else 0),
                    color=acento if fuerte else ce["color_texto_secundario"], fontweight="bold" if fuerte else "normal")
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Tiempo en postura correcta (%)")
        ax.grid(axis="y", visible=False)
        pie = ("Cada barra es una sesión. En color, la que la narración señala por tener más tiempo fuera de postura; "
               "la línea punteada es el valor de toda la selección.")
    else:
        sid = datos["sesion_id"].iloc[0]
        ses = datos[datos["sesion_id"] == sid].sort_values("t")
        r = est.rachas(ses, float(config["calidad"]["cadencia_nominal_s"]))
        for _, x in r.iterrows():
            ax.barh(0, x["duracion_s"] / 60, left=x["inicio_s"] / 60, height=0.5,
                    color=acento if x["estado"] == "MAL" else GRIS, linewidth=0)
        mal = r[r["estado"] == "MAL"]
        if len(mal):
            larga = mal.loc[mal["duracion_s"].idxmax()]
            xm = (larga["inicio_s"] + larga["duracion_s"] / 2) / 60
            primera = mal["inicio_s"].min() / 60
            es_primera = primera == larga["inicio_s"] / 60
            ax.annotate("Primera alerta y episodio MAL más largo" if es_primera else "Episodio MAL más largo", xy=(xm, 0.25),
                        xytext=(xm, 0.75), ha="center", fontsize=ce["tamano_base"], color=acento, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=acento))
            if not es_primera:
                ax.annotate("Primera alerta", xy=(primera, -0.25),
                            xytext=(primera, -0.75), ha="center", fontsize=ce["tamano_pie"] + 0.5,
                            color=ce["color_texto"], arrowprops=dict(arrowstyle="-|>", color=ce["color_texto"]))
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([0], [str(sid)], fontsize=ce["tamano_pie"] + 0.5)
        ax.set_xlabel("Minutos desde el CALIBRADO")
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Patch(color=GRIS, label="BIEN"), Patch(color=acento, label="MAL")], loc="upper right",
                  fontsize=ce["tamano_pie"], frameon=False, ncol=2)
        pie = "Línea de tiempo de la sesión: gris = BIEN, color = MAL. Se señalan el episodio MAL más largo y la primera alerta."
    if con_titulo:
        _titulo(fig, config, nar["titular"], f"Selección: {alcance} · {total['n']} mediciones de {total['sesiones']} "
                                             f"sesi{'ón' if total['sesiones'] == 1 else 'ones'}")
        _pie(fig, config, pie + " Gráfico generado a partir de los mismos datos que la narración.")
    else:
        fig.text(0.05, 0.07, pie, ha="left", va="top", fontsize=ce["tamano_pie"], color=ce["color_texto_secundario"])
    return fig


def _indicadores(datos, config, analisis, alcance):
    ce = config["estilo"]
    total, sesiones = analisis["total"], analisis["por_sesion"]
    fig = plt.figure(figsize=tuple(config["diagnostica"]["tamanos_pulgadas"]["indicadores"]))
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, top=0.84, bottom=0.2, hspace=0.6, wspace=0.35)
    x = np.arange(len(sesiones))
    for k, i in enumerate(ind.INDICADORES):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        vals = [s[i["clave"]] for s in sesiones]
        ax.bar(x, [v if v is not None else 0 for v in vals], color=COLORES[k % len(COLORES)], alpha=0.85)
        for xi, val in zip(x, vals):
            if val is None:
                ax.text(xi, 0, "—", ha="center", va="bottom", fontsize=7)
        if total[i["clave"]] is not None:
            ax.axhline(total[i["clave"]], color=ce["color_texto"], linestyle="--", linewidth=1)
        nombre = i["nombre"] if len(i["nombre"]) <= 34 else i["nombre"][:33] + "…"
        ax.set_title(f"{nombre}\n({i['unidad']})", fontsize=ce["tamano_pie"] + 0.5, loc="left")
        ax.set_xticks(x, [f"S{j + 1}" for j in x] if len(x) <= 15 else [""] * len(x), fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Indicadores por sesión · {alcance} · {len(sesiones)} sesiones", x=0.06, ha="left",
                 fontsize=ce["tamano_titulo"], fontweight="bold")
    texto = ("Barras: valor de cada sesión (S1, S2… en el orden de la tabla «Indicadores por sesión»). Línea punteada: "
             "valor de toda la selección. — = no aplica (por ejemplo, sesión sin alertas).\n"
             + "  ".join(f"S{k + 1}: {s['sesion_id']}" for k, s in enumerate(sesiones[:15])))
    fig.text(0.06, 0.1, "\n".join(textwrap.fill(p, 150) for p in texto.split("\n")), ha="left", va="top",
             fontsize=ce["tamano_pie"], color=ce["color_texto_secundario"])
    return fig


def figura_diagnostica(datos, config: dict, analisis: dict, figura: str, alcance: str, con_titulo: bool = True):
    if figura not in FIGURAS:
        raise ValueError(f"Figura desconocida: {figura!r} (use {', '.join(FIGURAS)})")
    with estilo.aplicar(config):
        if figura == "narrativa":
            return _narrativa(datos, config, analisis, alcance, con_titulo)
        return _indicadores(datos, config, analisis, alcance)


# ---------------------------------------------------------------- informe

def _pagina_texto(pdf, config, bloques):
    """Página carta horizontal con bloques (titulo, [parrafos]) que fluyen hacia abajo."""
    ce = config["estilo"]
    fig = plt.figure(figsize=(11, 8.5))
    y = 0.94
    for titulo, parrafos, tamano in bloques:
        if titulo:
            fig.text(0.06, y, titulo, fontsize=tamano + 2, fontweight="bold", va="top")
            y -= 0.045
        for p in parrafos:
            lineas = textwrap.wrap(p, 120 if tamano <= 10 else 95)
            fig.text(0.06, y, "\n".join(lineas), fontsize=tamano, va="top", color=ce["color_texto"])
            y -= 0.022 * len(lineas) * (tamano / 9) + 0.012
        y -= 0.015
    pdf.savefig(fig)
    plt.close(fig)


def informe_pdf(datos, config: dict, analisis: dict, alcance: str) -> bytes:
    nar = analisis["narracion"]
    buf = io.BytesIO()
    with estilo.aplicar(config), PdfPages(buf) as pdf:
        # Cada cifra aparece una vez: los valores están en «Indicadores»; el resto del texto no los repite.
        valor = (lambda i: "—" if i["valor"] is None else f"{str(i['valor']).replace('.', ',')} {i['unidad']}")
        _pagina_texto(pdf, config, [
            ("Narración de datos", [], 14),
            (None, [nar["titular"]], 13),
            ("Contexto", [nar["contexto"]], 10),
            ("Indicadores", [f"• {i['nombre']}: {valor(i)}. {i['lectura'] or ''}" for i in analisis["indicadores"]], 9.5),
            ("Otros hallazgos", [f"• {h['tema']}: {h['texto']}" for h in nar["hallazgos"]] or ["• Ninguno."], 9.5),
        ])
        _pagina_texto(pdf, config, [
            ("Qué significa", [nar["significado"]] if nar["significado"] else [], 10),
            ("Qué revisar", [f"• {r}" for r in nar["recomendaciones"]] or ["• Nada en particular con estos datos."], 9.5),
            ("Limitaciones", [f"• {x}" for x in nar["limitaciones"]] or ["• Ninguna relevante."], 9.5),
        ])
        for figura in FIGURAS:
            f = figura_diagnostica(datos, config, analisis, figura, alcance, con_titulo=False)
            pdf.savefig(f)
            plt.close(f)
        _pagina_texto(pdf, config, [("Cómo se calcula cada indicador", [], 14)] + [
            (None, [i["nombre"], f"Definición: {i['definicion']}  Fórmula: {i['formula']}  "
                                 f"Interpretación general: {i['interpretacion']}"], 8.5)
            for i in analisis["indicadores"]])
        info = pdf.infodict()
        info["Title"] = "Analítica diagnóstica"
        info["Subject"] = alcance
    return buf.getvalue()
