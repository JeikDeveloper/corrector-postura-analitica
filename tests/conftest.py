from pathlib import Path

import pytest

from postura.configuracion import cargar_configuracion
from postura.ingesta import ingerir_archivos

DATOS = Path(__file__).parent / "datos"
REFERENCIA = DATOS / "referencia.txt"


@pytest.fixture
def config():
    return cargar_configuracion()


@pytest.fixture
def ingerir(config):
    def _ingerir(*nombres, **sobrescrituras):
        cfg = cargar_configuracion(**sobrescrituras) if sobrescrituras else config
        return ingerir_archivos([DATOS / n for n in nombres], cfg)
    return _ingerir


@pytest.fixture
def referencia(ingerir):
    return ingerir("referencia.txt")
