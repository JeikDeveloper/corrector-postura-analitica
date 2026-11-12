import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from postura import estadistica as est
from postura.figuras.densidad import exportar, seleccion
from postura.figuras.estadistica import FIGURAS, figura_estadistica, reporte_pdf
from postura.web import crear_app
from conftest import DATOS


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    yield
    plt.close("all")


def _datos(referencia, config):
    return seleccion(referencia.mediciones, config, None, None)[0]


def test_resumen_formulas():
    r = est.resumen(np.array([1, 2, 3, 4, 10]), [5, 25, 50, 75, 95])
    assert r["n"] == 5 and r["media"] == 4.0 and r["min"] == 1.0 and r["max"] == 10.0
    assert r["sd"] == round(float(np.std([1, 2, 3, 4, 10], ddof=1)), 2)
    assert r["percentiles"]["50"] == 3.0 and r["iqr"] == 2.0
    assert r["asimetria"] > 0                         # cola hacia la derecha
    vacio = est.resumen(np.array([]), [50])
    assert vacio["n"] == 0 and vacio["media"] is None and vacio["percentiles"] == {"50": None}
    uno = est.resumen(np.array([7]), [50])
    assert uno["sd"] is None and uno["asimetria"] is None


def test_descriptivos_por_estado(referencia, config):
    filas = est.descriptivos(_datos(referencia, config), config)
    assert [(f["angulo"], f["grupo"]) for f in filas] == [
        ("frontal", "Todos"), ("frontal", "BIEN"), ("frontal", "MAL"),
        ("lateral", "Todos"), ("lateral", "BIEN"), ("lateral", "MAL")]
    assert filas[0]["n"] == 339 and filas[1]["n"] + filas[2]["n"] == 339
    assert set(filas[0]["percentiles"]) == {"5", "25", "50", "75", "95"}


def test_rachas_y_episodios(config):
    datos = pd.DataFrame({
        "sesion_id": ["a"] * 6 + ["b"] * 2,
        "t": pd.date_range("2026-01-01 10:00", periods=8, freq="s"),
        "t_rel_calibracion": [0, 1, 2, 3, 4, 5, 0, 1],
        "estado": ["BIEN", "BIEN", "MAL", "MAL", "MAL", "BIEN", "MAL", "MAL"],
    })
    r = est.rachas(datos, 1.0)
    assert list(r["estado"]) == ["BIEN", "MAL", "BIEN", "MAL"]
    assert list(r["n"]) == [2, 3, 1, 2]
    assert list(r["duracion_s"]) == [2.0, 3.0, 1.0, 2.0]          # último − primero + una cadencia
    e = est.episodios(datos, config)
    assert e["MAL"]["n_episodios"] == 2 and e["MAL"]["duracion_max_s"] == 3.0 and e["MAL"]["tiempo_total_s"] == 5.0


def test_causas_mal(config):
    datos = pd.DataFrame({"estado": ["MAL", "MAL", "MAL", "BIEN"], "frontal": [30, -12, 0, 50],
                          "lateral": [20, 0, 0, 0]})
    c = {x["clave"]: x for x in est.causas_mal(datos, config)}
    assert c["frontal_adelante"]["n"] == 1 and c["lateral_derecha"]["n"] == 1   # la primera excede dos
    assert c["frontal_atras"]["n"] == 1 and c["sin_umbral"]["n"] == 1
    assert c["frontal_adelante"]["pct"] == round(100 / 3, 1)


def test_analisis_referencia(referencia, config):
    a = est.analisis(_datos(referencia, config), config)
    assert a["n"] == 339 and a["conteo"] == {"BIEN": 336, "MAL": 3} and a["pct_mal"] == 0.9
    assert a["episodios"]["MAL"]["n_episodios"] == 2
    assert sum(c["n"] for c in a["causas_mal"] if c["clave"] != "sin_umbral") >= 3


@pytest.mark.parametrize("tipo", FIGURAS)
def test_figuras_deterministas(referencia, config, tipo):
    a = figura_estadistica(referencia.mediciones, config, tipo)
    b = figura_estadistica(referencia.mediciones, config, tipo)
    assert a.n_muestras == 339 and "CALIBRADO" in a.pie
    svg_a = exportar(a.figura, "svg", config)
    svg_b = exportar(b.figura, "svg", config)
    assert svg_a.startswith(b"<?xml") and b"<text" in svg_a      # texto editable
    # Mismos datos → misma imagen (se ignoran los ids y la fecha que matplotlib genera al guardar)
    import re
    limpiar = lambda s: re.sub(rb'(id|href|clip-path)="[^"]*"|<dc:date>[^<]*</dc:date>', b"", s)  # noqa: E731
    assert limpiar(svg_a) == limpiar(svg_b)


def test_figura_sin_datos(referencia, config):
    for tipo in FIGURAS:
        assert figura_estadistica(referencia.mediciones, config, tipo, usuario="nadie").n_muestras == 0


def test_reporte_pdf_varias_paginas(referencia, config):
    import re
    pdf = reporte_pdf(referencia.mediciones, config)
    paginas = re.findall(rb"/Type\s*/Page(?!s)", pdf)
    assert pdf.startswith(b"%PDF") and len(paginas) == 1 + len(FIGURAS)   # resumen + una por figura


# ---------------------------------------------------------------- web

def _post(cliente, ruta, **campos):
    datos = (DATOS / "referencia.txt").read_bytes()
    return cliente.post(ruta, data={"archivos": [(io.BytesIO(datos), "referencia.txt")], **campos},
                        content_type="multipart/form-data")


@pytest.fixture
def cliente(config):
    return crear_app(config).test_client()


def test_api_estadistica(cliente, referencia):
    j = _post(cliente, "/api/analizar/estadistica").get_json()
    assert j["n"] == 339 and j["formatos"] == ["png", "svg", "pdf"] and len(j["descriptivos"]) == 6
    sid = referencia.sesiones["sesion_id"].iloc[1]
    assert _post(cliente, "/api/analizar/estadistica", sesion=sid).get_json()["n"] == 199


@pytest.mark.parametrize("formato, firma", [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")])
def test_api_figura(cliente, formato, firma):
    r = _post(cliente, f"/api/analizar/estadistica.{formato}", figura="cajas", descargar="1")
    assert r.status_code == 200 and r.data.startswith(firma)
    assert f"estadistica_cajas_referencia_completo.{formato}" in r.headers["Content-Disposition"]


def test_api_png_300_ppp(cliente):
    from PIL import Image
    r = _post(cliente, "/api/analizar/estadistica.png", figura="histogramas")
    dpi = Image.open(io.BytesIO(r.data)).info["dpi"]
    assert round(dpi[0]) == 300


def test_api_errores(cliente):
    assert _post(cliente, "/api/analizar/estadistica.jpg").status_code == 404     # solo png, svg, pdf
    assert _post(cliente, "/api/analizar/estadistica.png", figura="torta").status_code == 400


def test_api_reporte_y_csv(cliente):
    r = _post(cliente, "/api/analizar/estadistica/reporte.pdf")
    assert r.status_code == 200 and r.data.startswith(b"%PDF")
    assert "reporte_estadistico_referencia_completo.pdf" in r.headers["Content-Disposition"]
    c = _post(cliente, "/api/analizar/estadistica.csv").data.decode("utf-8-sig").splitlines()
    assert c[0].startswith("angulo,grupo,n,media,sd,min,p5,p25,p50,p75,p95,max") and len(c) == 7
