import io

import matplotlib.pyplot as plt
import numpy as np
import pytest

from postura.figuras.densidad import exportar
from postura.figuras.densidad3d import figura_densidad_3d, malla_3d
from postura.web import crear_app
from conftest import DATOS


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    yield
    plt.close("all")


def test_malla_misma_seleccion_que_el_mapa_2d(referencia, config):
    m = malla_3d(referencia.mediciones, config)
    r = config["figura"]["tres_d"]["resolucion_malla"]
    assert m["metodo"] == "kde" and m["n"] == 339 and m["n_sesiones"] == 3
    assert len(m["x"]) == len(m["y"]) == r and len(m["z"]) == r and len(m["z"][0]) == r
    assert min(min(f) for f in m["z"]) >= 0 and "AGREGACIÓN" in m["pie"]
    assert m["umbrales"] == {"lateral": [-15, 15], "frontal": [-10, 25]}


def test_malla_una_muestra_y_sin_datos(ingerir, referencia, config):
    una = malla_3d(ingerir("una_muestra.txt").mediciones, config)
    assert una["metodo"] == "histograma2d" and np.isclose(np.sum(una["z"]), 1)
    vacia = malla_3d(referencia.mediciones, config, usuario="nadie")
    assert vacia["metodo"] == "vacio" and vacia["n"] == 0 and "no se rellena" in vacia["pie"]


@pytest.mark.parametrize("formato, firma", [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")])
def test_imagen_fija(referencia, config, formato, firma):
    sid = referencia.sesiones["sesion_id"].iloc[1]
    fig, s = figura_densidad_3d(referencia.mediciones, config, sesion=sid)
    assert s.n == 199 and sid in s.titulo
    assert exportar(fig, formato, config).startswith(firma)


def _post(cliente, ruta, **campos):
    datos = (DATOS / "referencia.txt").read_bytes()
    return cliente.post(ruta, data={"archivos": [(io.BytesIO(datos), "referencia.txt")], **campos},
                        content_type="multipart/form-data")


def test_web_3d(config):
    cliente = crear_app(config).test_client()
    j = _post(cliente, "/api/densidad3d").get_json()
    assert j["n"] == 339 and j["plotly_cdn"].startswith("https://")
    r = _post(cliente, "/api/figura3d.jpg", descargar="1")
    assert r.status_code == 200 and r.data.startswith(b"\xff\xd8\xff")
    assert "densidad3d_referencia_completo.jpg" in r.headers["Content-Disposition"]
    assert _post(cliente, "/api/figura3d.gif").status_code == 404
