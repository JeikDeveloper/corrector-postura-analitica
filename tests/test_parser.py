import pytest

from postura import parser as p
from conftest import DATOS, REFERENCIA


# 1. Signo pegado y signo separado por espacio dan el mismo valor
@pytest.mark.parametrize("pegado, separado", [("-2", "- 2"), ("-14", "-  14"), ("-1", "−1")])
def test_signo_pegado_y_separado_equivalen(pegado, separado):
    assert p.parsear_angulo(pegado) == p.parsear_angulo(separado) < 0


def test_linea_con_signo_separado():
    a = p.parsear_linea("16:17:05.820 F = - 2º, L = 0º, BIEN")
    b = p.parsear_linea("16:17:05.820 F = -2º, L = 0º, BIEN")
    assert a.tipo == b.tipo == p.MEDICION
    assert (a.frontal, a.lateral) == (b.frontal, b.lateral) == (-2, 0)


# 2. º (U+00BA) y, por tolerancia, ° (U+00B0)
def test_caracter_ordinal_oficial():
    ev = p.parsear_linea("16:18:06.290 F = -14º, L = 9º, MAL")
    assert (ev.tipo, ev.frontal, ev.lateral, ev.estado) == (p.MEDICION, -14, 9, "MAL")
    assert not ev.grado_alternativo


def test_archivo_con_simbolo_de_grado_alternativo():
    r = p.parsear_archivo(DATOS / "grado_alternativo.txt")
    med = r.de_tipo(p.MEDICION)
    assert [(e.frontal, e.lateral) for e in med] == [(2, -3), (-14, 9), (-2, 0)]
    assert all(e.grado_alternativo for e in med)


def test_referencia_conteos():
    r = p.parsear_archivo(REFERENCIA)
    assert r.n_lineas == 363
    assert len(r.de_tipo(p.MEDICION)) == 347
    assert {e.dispositivo for e in r.de_tipo(p.CONECTANDO)} == {"WLAB CP"}
    assert not r.de_tipo(p.DESCONOCIDO, p.INVALIDA)


def test_eventos_de_sistema():
    casos = {
        "Connecting to WLAB CP ...": p.CONECTANDO,
        "Connected": p.CONECTADO,
        "CALIBRADO": p.CALIBRADO,
        "APAGADO": p.APAGADO,
        "Disconnected": p.DESCONECTADO,
        "Connection lost": p.CONEXION_PERDIDA,
        "Connection failed: gatt status 8": p.CONEXION_FALLIDA,
    }
    for texto, tipo in casos.items():
        assert p.parsear_linea(f"10:00:00.000 {texto}").tipo == tipo
    assert p.parsear_linea("10:00:00.000 Connection failed: gatt status 133").gatt_status == 133


def test_evento_desconocido_no_aborta_y_se_registra(caplog):
    with caplog.at_level("WARNING", logger="postura.ingesta"):
        r = p.parsear_archivo(DATOS / "grado_alternativo.txt")
    assert [e.texto for e in r.de_tipo(p.DESCONOCIDO)] == ["10:00:05.000 BATERIA BAJA 15%"]
    assert len(r.de_tipo(p.INVALIDA)) == 1
    assert "BATERIA BAJA 15%" in caplog.text
    assert "linea sin marca de tiempo" in caplog.text


def test_medicion_malformada_es_desconocida():
    assert p.parsear_linea("10:00:00.000 F = 2º, L = 1º, REGULAR").tipo == p.DESCONOCIDO
    assert p.parsear_linea("10:00:00.000 F = 2.5º, L = 1º, BIEN").tipo == p.DESCONOCIDO
