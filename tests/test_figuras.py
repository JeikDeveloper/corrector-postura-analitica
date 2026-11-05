import matplotlib.pyplot as plt
import pytest

from postura.cli import principal
from postura.figuras.densidad import figura_densidad
from conftest import DATOS


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    yield
    plt.close("all")


# 9. Figura con el conjunto completo y con una sola muestra
@pytest.mark.parametrize("metodo", ["kde", "hex"])
def test_figura_conjunto_completo(referencia, config, tmp_path, metodo):
    r = figura_densidad(referencia.mediciones, config, metodo=metodo, salida=tmp_path)
    assert r.n_muestras == 339 and r.n_sesiones == 3
    assert r.metodo == metodo
    assert "n = 339" in r.titulo and "conjunto completo" in r.titulo
    assert "3 sesiones" in r.pie and "propio cero" in r.pie
    extensiones = {p.suffix for p in r.rutas}
    assert {".png", ".svg"} <= extensiones
    assert all(p.stat().st_size > 0 for p in r.rutas)


def test_figura_una_muestra(ingerir, config, tmp_path):
    datos = ingerir("una_muestra.txt")
    r = figura_densidad(datos.mediciones, config, salida=tmp_path)
    assert r.n_muestras == 1
    assert r.metodo == "histograma2d"          # KDE no estimable con 1 muestra: se dice en el pie
    assert "no estimable" in r.pie
    assert "AGREGACIÓN" not in r.pie


def test_figura_por_sesion(referencia, config):
    sid = referencia.sesiones["sesion_id"].iloc[1]
    r = figura_densidad(referencia.mediciones, config, sesion=sid)
    assert r.n_muestras == 199 and r.n_sesiones == 1
    assert sid in r.titulo and "AGREGACIÓN" not in r.pie


def test_figura_sin_datos_lo_dice(referencia, config):
    r = figura_densidad(referencia.mediciones, config, usuario="U01")
    assert r.n_muestras == 0 and r.metodo == "vacio"
    assert "No hay datos" in r.pie


def test_figura_sin_calibrar_excluida(ingerir, config):
    datos = ingerir("sin_calibrar.txt")
    r = figura_densidad(datos.mediciones, config)
    assert r.n_muestras == 0 and "3 de sesiones sin calibrar" in r.pie


def test_cli_extremo_a_extremo(tmp_path, capsys):
    parquet = tmp_path / "mediciones.parquet"
    assert principal(["ingerir", str(DATOS / "referencia.txt"), "--salida", str(parquet)]) == 0
    assert (tmp_path / "ingesta.log").is_file()
    assert principal(["calidad", str(parquet)]) == 0
    assert principal(["figura", "densidad", str(parquet), "--salida", str(tmp_path)]) == 0
    salida = capsys.readouterr().out
    assert "Discrepancias estado reportado vs derivado: 0 de 347" in salida
    assert list(tmp_path.glob("densidad_conjunto_completo_kde.png"))


def test_svg_con_texto_editable(referencia, config):
    from postura.figuras.densidad import exportar
    r = figura_densidad(referencia.mediciones, config)
    svg = exportar(r.figura, "svg", config).decode("utf-8")
    assert "<text" in svg and "Espacio postural" in svg   # texto real, no trazos
