import io
import re

import matplotlib.pyplot as plt
import pytest

from postura import diagnostica as diag
from postura import indicadores as ind
from postura.figuras.densidad import exportar, seleccion
from postura.figuras.diagnostica import FIGURAS, figura_diagnostica, informe_pdf
from postura.ingesta import ingerir_contenidos
from postura.web import crear_app
from conftest import DATOS


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    yield
    plt.close("all")


def _sesion(estados, f_mal=30, nombre="s.txt", config=None):
    """Registro sintético: una medición por segundo con el estado pedido."""
    lineas = ["10:00:00.000 Connected", "10:00:00.500 CALIBRADO"]
    for i, e in enumerate(estados):
        s = 1 + i
        lineas.append(f"10:{s // 60:02d}:{s % 60:02d}.000 F = {f_mal if e == 'MAL' else 2}º, L = 1º, {e}")
    s = 2 + len(estados)
    lineas.append(f"10:{s // 60:02d}:{s % 60:02d}.000 Disconnected")
    m = ingerir_contenidos([(nombre, "\n".join(lineas).encode())], config).mediciones
    return m[m["post_calibracion"]]


def _datos(referencia, config):
    return seleccion(referencia.mediciones, config, None, None)[0]


# ---------------------------------------------------------------- indicadores

def test_indicadores_definidos(config):
    assert len(ind.INDICADORES) >= 6
    assert all(i["definicion"] and i["formula"] and i["interpretacion"] and i["unidad"] for i in ind.INDICADORES)


def test_indicadores_valores(config):
    r = ind.calcular(_sesion(["BIEN"] * 10 + ["MAL"] * 2 + ["BIEN"] * 8, config=config), config)
    assert r["tiempo_correcto"] == 90.0 and r["duracion_mal"] == 2.0
    assert r["alertas_hora"] == round(1 / (20 / 3600), 1)          # 1 episodio en 20 s
    assert r["primera_alerta"] == round(10.5 / 60, 1)              # el primer MAL, a 10,5 s del CALIBRADO
    vacio = ind.calcular(_sesion(["BIEN"] * 3, config=config).iloc[:0], config)
    assert vacio["n"] == 0 and vacio["tiempo_correcto"] is None


# ---------------------------------------------------------------- narración

def test_narracion_referencia(referencia, config):
    n = diag.narracion(_datos(referencia, config), config, "conjunto completo")
    assert n["titular"] == ("La postura se mantuvo correcta la mayor parte del tiempo, pero las alertas fueron "
                            "frecuentes, aunque breves.")
    assert [h["tema"] for h in n["hallazgos"]] == ["Episodios", "Causa principal", "Diferencias entre sesiones"]
    assert n["hallazgos"][0]["texto"] == "2 episodios MAL en total; el más largo duró 2 s."
    causa = n["hallazgos"][1]["texto"]
    assert "inclinarse hacia atrás (F < -10°)" in causa and "66,7 %" in causa
    assert n["lecturas"]["tiempo_correcto"].startswith("Alto") and n["lecturas"]["alertas_hora"].startswith("Frecuentes")
    assert n["lecturas"]["zona_optima"].startswith("6,5 puntos por debajo")
    assert n["destacada"] == "referencia-s02-161657879"
    assert any("respaldo" in r for r in n["recomendaciones"])
    assert not any("causa más común" in r for r in n["recomendaciones"])   # la causa se dice una vez, en los hallazgos
    assert any("poco representativos" in x for x in n["limitaciones"])   # solo 2 episodios MAL


def test_narracion_sin_alertas_y_baja(config):
    sin = diag.narracion(_sesion(["BIEN"] * 30, config=config), config, "x")
    assert sin["titular"].endswith("y no hubo alertas.") and sin["recomendaciones"] == []
    mala = diag.narracion(_sesion(["BIEN"] * 10 + ["MAL"] * 30, config=config), config, "x")
    assert "fuera de los umbrales una parte importante" in mala["titular"]
    assert "sostenido" in mala["significado"]
    assert "inclinarse hacia adelante (F > 25°)" in mala["hallazgos"][1]["texto"]
    assert mala["lecturas"]["tiempo_correcto"].startswith("Bajo")


def test_cada_cifra_aparece_una_vez(referencia, config):
    """Los valores de los indicadores solo están en los indicadores: ni el titular, ni los hallazgos,
    ni las lecturas, ni las limitaciones los repiten."""
    a = diag.analisis(_datos(referencia, config), config, "conjunto completo")
    n = a["narracion"]
    texto = " ".join([n["titular"], n["significado"] or "", *n["recomendaciones"], *n["limitaciones"],
                      *(h["texto"] for h in n["hallazgos"]), *n["lecturas"].values()])
    for i in a["indicadores"]:
        if i["valor"] is not None:                                    # el número con su unidad (p. ej. «1,5 s»)
            unidad = i["unidad"].split("/")[0].split(" ")[0]
            assert f"{diag._n(i['valor'], 3)} {unidad}" not in texto, (i["clave"], i["valor"])
    assert not re.search(r"\d", n["titular"])                       # el titular cuenta el mensaje sin cifras
    assert "5,7 min" in n["contexto"] and "5,7 min" not in texto     # la duración solo en el contexto


def test_narracion_sin_datos(referencia, config):
    n = diag.narracion(_datos(referencia, config).iloc[:0], config, "persona nadie")
    assert n["hallazgos"] == [] and "No hay mediciones" in n["titular"]


def test_narracion_usa_umbrales_de_config(referencia, config):
    from postura.configuracion import fusionar
    exigente = fusionar(config, {"diagnostica": {"narracion": {"tiempo_correcto_alto": 99.5}}})
    n = diag.narracion(_datos(referencia, config), exigente, "x")
    assert n["titular"].startswith("La postura fue correcta en buena parte del tiempo")


# ---------------------------------------------------------------- figuras e informe

@pytest.mark.parametrize("figura", FIGURAS)
def test_figuras(referencia, config, figura):
    datos = _datos(referencia, config)
    a = diag.analisis(datos, config, "conjunto completo")
    svg = exportar(figura_diagnostica(datos, config, a, figura, "conjunto completo"), "svg", config).decode()
    assert "<text" in svg
    if figura == "narrativa":
        assert "referencia-s02-161657879" in svg and "selección: 99,1" in svg      # descarga: con título y valor global
        pagina = exportar(figura_diagnostica(datos, config, a, figura, "x", con_titulo=False), "svg", config).decode()
        assert "selección: 99,1" not in pagina and a["narracion"]["titular"][:30] not in pagina


def test_figura_narrativa_una_sesion(config):
    datos = _sesion(["BIEN"] * 20 + ["MAL"] * 3 + ["BIEN"] * 10 + ["MAL"] * 6 + ["BIEN"] * 5, config=config)
    a = diag.analisis(datos, config, "x")
    svg = exportar(figura_diagnostica(datos, config, a, "narrativa", "x"), "svg", config).decode()
    assert "Episodio MAL más largo" in svg and "Primera alerta" in svg
    assert "el más largo duró 6 s" in a["narracion"]["hallazgos"][0]["texto"]     # la cifra está en el texto, no en el gráfico


def test_informe_pdf(referencia, config):
    datos = _datos(referencia, config)
    pdf = informe_pdf(datos, config, diag.analisis(datos, config, "x"), "x")
    assert pdf.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page(?!s)", pdf)) == 5


# ---------------------------------------------------------------- API

def _post(cliente, ruta, **campos):
    datos = (DATOS / "referencia.txt").read_bytes()
    return cliente.post(ruta, data={"archivos": [(io.BytesIO(datos), "referencia.txt")], **campos},
                        content_type="multipart/form-data")


def test_api_diagnostica(config, referencia):
    c = crear_app(config).test_client()
    j = _post(c, "/api/diagnostica").get_json()
    assert len(j["indicadores"]) == len(ind.INDICADORES) and j["narracion"]["titular"]
    assert len(j["por_sesion"]) == 3 and j["formatos"] == ["png", "svg", "pdf"]
    sid = referencia.sesiones["sesion_id"].iloc[0]
    assert _post(c, "/api/diagnostica", sesion=sid).get_json()["total"]["n"] == 80
    r = _post(c, "/api/diagnostica/figura.png", figura="narrativa", descargar="1")
    assert r.status_code == 200 and "diagnostica_narrativa_referencia_completo.png" in r.headers["Content-Disposition"]
    assert _post(c, "/api/diagnostica/figura.png", figura="torta").status_code == 400
    assert _post(c, "/api/diagnostica/figura.jpg").status_code == 404
    r = _post(c, "/api/diagnostica/informe.pdf")
    assert r.status_code == 200 and r.data.startswith(b"%PDF")
    csv = _post(c, "/api/diagnostica/indicadores.csv").data.decode("utf-8-sig").splitlines()
    assert csv[0].startswith("sesion_id,usuario_id,n,tiempo_correcto") and len(csv) == 1 + 1 + 3
