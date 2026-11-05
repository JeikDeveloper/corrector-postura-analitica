import pytest

from postura.validacion import revisar_archivo
from conftest import DATOS


def test_referencia_se_acepta(config):
    parseo, motivo = revisar_archivo("referencia.txt", (DATOS / "referencia.txt").read_bytes(), config)
    assert motivo is None and parseo.n_lineas == 363 and parseo.n_bytes > 0


def test_advertencias_no_son_rechazo(config):
    # Tiene un evento desconocido y una línea sin marca, pero es un registro: se acepta y se advierte.
    assert revisar_archivo("x.txt", (DATOS / "grado_alternativo.txt").read_bytes(), config)[1] is None


@pytest.mark.parametrize("nombre, crudo, fragmento", [
    ("a.pdf", b"10:00:00.000 Connected\n", "extensión .pdf"),
    ("a", b"10:00:00.000 Connected\n", "(ninguna)"),
    ("a.txt", b"  \n\n", "vacío"),
    ("a.txt", b"%PDF-1.7\n", "PDF"),
    ("a.txt", b"PK\x03\x04resto", "Office"),
    ("a.txt", b"10:00:00.000 Connected\x00\x01", "binario"),
    ("a.txt", "10:00:00.000 Connected\n".encode("utf-16"), "binario"),
    ("a.txt", b"10:00:00.000 F = 2\xba, L = 1\xba, BIEN\n", "UTF-8"),
    ("a.txt", b"hola\nmundo\n", "marca de tiempo"),
    ("a.txt", b"10:00:00.000 Connected\n10:00:01.000 Disconnected\n", "medición"),
    ("a.txt", "10:00:00.000 F = 2º, L = 1º, BIEN\n".encode(), "Connected"),
])
def test_motivos_de_rechazo(config, nombre, crudo, fragmento):
    parseo, motivo = revisar_archivo(nombre, crudo, config)
    assert parseo is None and motivo and fragmento in motivo
