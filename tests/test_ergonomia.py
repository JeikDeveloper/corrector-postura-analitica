import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from postura import ergonomia as erg
from postura.configuracion import fusionar
from postura.figuras.densidad import exportar
from postura.figuras.ergonomia import FIGURAS, figura_ergonomia
from postura.web import crear_app
from conftest import DATOS


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    yield
    plt.close("all")


@pytest.mark.parametrize("frontal, lateral, zona", [
    (5, 0, "optima"), (0, -10, "optima"), (20, 10, "optima"),       # bordes cerrados
    (10, 15, "aceptable"), (10, -20, "aceptable"),                   # costados
    (30, 0, "aceptable"), (45, 25, "aceptable"),                     # flexión 20°–60°
    (-1, 0, "extrema"), (61, 0, "extrema"), (10, 21, "extrema"),
])
def test_zonas_iso(config, frontal, lateral, zona):
    assert erg.clasificar_zona([frontal], [lateral], config)[0] == zona


def test_zonas_por_eje(config):
    assert list(erg.clasificar_eje([-5, 10, 40, 70], "frontal", config)) == ["extrema", "optima", "aceptable", "extrema"]
    assert list(erg.clasificar_eje([0, -15, 25], "lateral", config)) == ["optima", "aceptable", "extrema"]


def test_momentos_del_dia_cruza_medianoche(config):
    t = pd.Series(pd.to_datetime(["2026-01-01 05:59", "2026-01-01 06:00", "2026-01-01 12:00",
                                  "2026-01-01 17:59", "2026-01-01 18:00", "2026-01-01 23:30"]))
    assert list(erg.momento_del_dia(t, config)) == ["noche", "manana", "tarde", "tarde", "noche", "noche"]


def test_metricas_formulas(config):
    datos = pd.DataFrame({"frontal": [0, 6, 0, 6], "lateral": [0, 0, 8, 8]})
    m = erg.metricas(datos, config)
    assert (m.media_frontal, m.media_lateral) == (3.0, 4.0)
    assert m.desplazamiento == 5.0                              # √(3² + 4²)
    assert m.dispersion == round(math.hypot(3, 4), 2)           # σα = 3, σβ = 4
    assert m.pct_zona == {"optima": 100.0, "aceptable": 0.0, "extrema": 0.0}   # todos con |β| ≤ 10°
    mixto = erg.metricas(pd.DataFrame({"frontal": [5, 5, 30, -2], "lateral": [0, 15, 0, 0]}), config)
    assert mixto.pct_zona == {"optima": 25.0, "aceptable": 50.0, "extrema": 25.0}
    assert erg.metricas(datos.iloc[:0], config).a_dict()["dispersion"] is None


def test_tabla_y_cocientes(referencia, config):
    datos, global_ = erg.seleccionar(referencia.mediciones, config)
    filas = erg.tabla_metricas(global_, config, "desconocido")
    assert filas[0]["tipo"] == "conjunto" and filas[0]["n"] == 339
    assert filas[0]["sigma_g"] == 1.0 and filas[0]["r_g"] == 1.0
    assert [f["tipo"] for f in filas].count("sesion") == 3
    assert sum(f["n"] for f in filas if f["tipo"] == "sesion") == 339


def test_anova_entre_momentos(config):
    rng = np.random.default_rng(0)
    filas = []
    for k, hora in enumerate([8, 9, 14, 15, 20, 21]):       # dos sesiones por momento
        inicio = pd.Timestamp(f"2026-01-01 {hora:02d}:00")
        for i in range(40):
            filas.append({"sesion_id": f"s{k}", "t": inicio + pd.Timedelta(seconds=i),
                          "frontal": int(rng.integers(0, 10 + 5 * k)), "lateral": int(rng.integers(-3, 4))})
    datos = pd.DataFrame(filas)
    r = erg.comparar_momentos(datos, datos, config)
    assert [m["n_tramos"] for m in r["momentos"]] == [2, 2, 2]
    assert r["anova_r_g"]["p"] is not None and 0 <= r["anova_r_g"]["p"] <= 1
    # Con un solo momento no hay comparación posible: se dice por qué
    una = erg.comparar_momentos(datos, datos[datos["t"].dt.hour < 12], config)
    assert una["anova_sigma_g"]["p"] is None and "al menos 2 momentos" in una["anova_sigma_g"]["motivo"]


@pytest.mark.parametrize("tipo", FIGURAS)
def test_figuras_ergonomicas(referencia, config, tipo):
    r = figura_ergonomia(referencia.mediciones, config, tipo)
    assert r.n_muestras == 339 and "CALIBRADO" in r.pie
    assert exportar(r.figura, "png", config).startswith(b"\x89PNG")


def test_figura_sin_datos_lo_dice(referencia, config):
    r = figura_ergonomia(referencia.mediciones, config, "zonas", usuario="nadie")
    assert r.n_muestras == 0 and "n = 0" in r.titulo


def test_figura_desconocida(referencia, config):
    with pytest.raises(ValueError):
        figura_ergonomia(referencia.mediciones, config, "torta")


# ---------------------------------------------------------------- web

@pytest.fixture
def cliente(config):
    app = crear_app(config)
    app.testing = True
    return app.test_client()


def _post(cliente, ruta, **campos):
    datos = (DATOS / "referencia.txt").read_bytes()
    return cliente.post(ruta, data={"archivos": [(io.BytesIO(datos), "referencia.txt")], **campos},
                        content_type="multipart/form-data")


@pytest.mark.parametrize("tipo", FIGURAS)
def test_web_figura_ergonomica(cliente, tipo):
    r = _post(cliente, "/api/ergonomia.png", figura=tipo, descargar="1")
    assert r.status_code == 200 and r.data.startswith(b"\x89PNG")
    assert f"ergonomia_{tipo}_referencia_" in r.headers["Content-Disposition"]


def test_web_figura_ergonomica_invalida(cliente):
    r = _post(cliente, "/api/ergonomia.svg", figura="torta")
    assert r.status_code == 400 and "zonas" in r.get_json()["error"]


def test_web_metricas(cliente):
    j = _post(cliente, "/api/ergonomia/metricas").get_json()
    assert j["seleccion"]["n"] == 339 and j["seleccion"]["sigma_g"] == 1.0
    assert set(j["seleccion"]["pct_zona"]) == {"optima", "aceptable", "extrema"}
    assert [m["clave"] for m in j["momentos"]["momentos"]] == ["manana", "tarde", "noche"]
    assert j["momentos"]["anova_r_g"]["p"] is None          # la referencia solo tiene tarde
    sid = j["filas"][-1]["clave"]
    sel = _post(cliente, "/api/ergonomia/metricas", sesion=sid).get_json()["seleccion"]
    assert sel["n"] == j["filas"][-1]["n"]


def test_web_metricas_csv(cliente):
    r = _post(cliente, "/api/ergonomia/metricas.csv")
    texto = r.data.decode("utf-8-sig")
    assert r.status_code == 200 and texto.startswith("tipo,etiqueta,n,")
    assert "pct_extrema" in texto.splitlines()[0] and len(texto.strip().splitlines()) == 1 + 1 + 1 + 3


def test_config_por_archivo_cambia_zonas(referencia, config):
    estricta = fusionar(config, {"ergonomia": {"zonas": [{"nivel": "optima", "frontal": [0, 1], "lateral": [0, 1]}]}})
    assert erg.metricas(referencia.mediciones, estricta).pct_zona["optima"] < \
        erg.metricas(referencia.mediciones, config).pct_zona["optima"]
