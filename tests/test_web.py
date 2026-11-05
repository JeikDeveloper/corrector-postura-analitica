import io

import pytest

from postura.web import crear_app
from conftest import DATOS

FIRMAS = {
    "png": b"\x89PNG",
    "jpg": b"\xff\xd8\xff",
    "pdf": b"%PDF",
    "svg": b"<?xml",
}


@pytest.fixture
def cliente(config):
    app = crear_app(config)
    app.testing = True
    return app.test_client()


def _subir(cliente, *nombres, contenido=None):
    if contenido is not None:
        archivos = [(io.BytesIO(contenido), nombres[0])]
    else:
        archivos = [(io.BytesIO((DATOS / n).read_bytes()), n) for n in nombres]
    return cliente.post("/api/analizar", data={"archivos": archivos}, content_type="multipart/form-data")


def test_pagina(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and "Subir registro" in r.get_data(as_text=True)


def test_referencia_estructura_correcta(cliente):
    r = _subir(cliente, "referencia.txt")
    assert r.status_code == 200
    j = r.get_json()
    est = j["estructura"]
    # Solo hay un aviso informativo (sin fecha en el nombre), que no cambia el veredicto
    assert j["api"] == 4
    assert est["veredicto"] == "correcta"
    assert est["resumen"]["lineas"] == 363 and est["resumen"]["mediciones"] == 347
    assert est["resumen"]["dispositivos"] == ["WLAB CP"]
    assert len(j["sesiones"]) == 3 and j["graficable"]
    assert j["n_discrepancias_estado"] == 0 and j["n_analizables"] == 339


def test_advertencias_con_linea(cliente):
    j = _subir(cliente, "grado_alternativo.txt").get_json()
    est = j["estructura"]
    assert est["veredicto"] == "con_advertencias"
    desconocido = [h for h in est["hallazgos"] if h["codigo"] == "evento_desconocido"]
    assert desconocido[0]["linea"] == 6 and "BATERIA BAJA" in desconocido[0]["texto"]
    assert est["conteo"]["linea_invalida"] == 1


def test_discrepancia_detectada_por_linea(cliente):
    texto = ("10:00:00.000 Connected\n10:00:01.000 CALIBRADO\n"
             "10:00:02.000 F = 30º, L = 0º, BIEN\n10:00:03.000 Disconnected\n").encode()
    est = _subir(cliente, "malo.txt", contenido=texto).get_json()["estructura"]
    h = [x for x in est["hallazgos"] if x["codigo"] == "discrepancia_estado"]
    assert h and h[0]["linea"] == 3 and "derivado MAL" in h[0]["detalle"]


def test_archivo_sin_formato_se_rechaza_sin_procesar(cliente):
    r = _subir(cliente, "otro.txt", contenido="hola\nesto no es un registro\n".encode())
    j = r.get_json()
    assert r.status_code == 422
    assert "id" not in j and "estructura" not in j
    assert j["rechazados"][0]["archivo"] == "otro.txt"
    assert "marca de tiempo" in j["rechazados"][0]["motivo"]


def test_pdf_se_rechaza(cliente):
    r = _subir(cliente, "informe.pdf", contenido=b"%PDF-1.4\n%\xe2\xe3\n1 0 obj\n")
    assert r.status_code == 422
    assert "extensión .pdf" in r.get_json()["rechazados"][0]["motivo"]


def test_lote_mixto_procesa_solo_los_validos(cliente):
    archivos = [
        (io.BytesIO((DATOS / "referencia.txt").read_bytes()), "referencia.txt"),
        (io.BytesIO(b"%PDF-1.4\n..."), "disfrazado.txt"),
    ]
    r = cliente.post("/api/analizar", data={"archivos": archivos}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200
    assert [a["archivo"] for a in j["por_archivo"]] == ["referencia.txt"]
    assert j["rechazados"][0]["archivo"] == "disfrazado.txt" and "PDF" in j["rechazados"][0]["motivo"]
    assert j["estructura"]["resumen"]["lineas"] == 363 and len(j["sesiones"]) == 3


def test_sin_archivos(cliente):
    r = cliente.post("/api/analizar", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "archivo" in r.get_json()["error"]


@pytest.mark.parametrize("formato", ["png", "svg", "pdf", "jpg"])
def test_descarga_formatos(cliente, formato):
    clave = _subir(cliente, "referencia.txt").get_json()["id"]
    r = cliente.get(f"/api/figura/{clave}.{formato}?descargar=1&metodo=hex")
    assert r.status_code == 200
    assert r.data.startswith(FIRMAS[formato])
    assert "attachment" in r.headers["Content-Disposition"]


def test_figura_por_sesion_y_csv(cliente):
    j = _subir(cliente, "referencia.txt").get_json()
    sid = j["sesiones"][0]["sesion_id"]
    r = cliente.get(f"/api/figura/{j['id']}.svg?sesion={sid}")
    assert r.status_code == 200 and b"n = 80 muestras" in r.data
    csv = cliente.get(f"/api/datos/{j['id']}.csv")
    assert csv.status_code == 200 and csv.data.decode("utf-8-sig").count("\n") == 348


def test_analisis_inexistente(cliente):
    r = cliente.get("/api/figura/no-existe.png")
    assert r.status_code == 404 and "error" in r.get_json()


def test_persona_por_archivo_y_filtro(cliente):
    datos = (DATOS / "una_muestra.txt").read_bytes()
    r = cliente.post("/api/analizar", data={
        "archivos": [(io.BytesIO(datos), "a.txt"), (io.BytesIO(datos), "b.txt")],
        "personas": ["Ana", ""],
    }, content_type="multipart/form-data")
    j = r.get_json()
    personas = {p["usuario_id"]: p["persona"] for p in j["personas"]}
    assert personas == {"Ana": "Ana", "desconocido": None}   # None = sin identificar
    assert [a["persona"] for a in j["por_archivo"]] == ["Ana", None]
    assert [s["persona"] for s in j["sesiones"]] == ["Ana", None]
    assert j["por_archivo"][0]["lineas"] == 5 and j["por_archivo"][0]["mediciones"] == 1
    svg = cliente.get(f"/api/figura/{j['id']}.svg?usuario=Ana").data
    assert b"persona Ana" in svg and b"n = 1 muestra" in svg


def test_persona_se_sanea(cliente):
    datos = (DATOS / "una_muestra.txt").read_bytes()
    j = cliente.post("/api/analizar", data={
        "archivos": [(io.BytesIO(datos), "a.txt")], "personas": ["$x^2$ <b>José</b>"],
    }, content_type="multipart/form-data").get_json()
    assert j["personas"][0]["usuario_id"] == "x2 bJosé/b".replace("/", "")
