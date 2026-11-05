import pandas as pd

from postura.ingesta import ingerir_archivos
from postura.modelo import seleccion_analizable
from postura.parser import parsear_archivo, parsear_texto
from postura.sesiones import construir
from conftest import DATOS


# 3. Tres sesiones en el archivo de referencia
def test_referencia_tres_sesiones(referencia):
    s = referencia.sesiones
    assert len(s) == 3
    assert list(s["indice"]) == [1, 2, 3]
    assert set(s["dispositivo"]) == {"WLAB CP"}
    assert list(s["evento_cierre"]) == ["apagado", "desconectado", "desconectado"]
    assert s["n_muestras"].sum() == len(referencia.mediciones) == 347
    assert not s["sin_calibrar"].any() and not s["cierre_implicito"].any()


def test_sesion_id_determinista(ingerir):
    a = ingerir("referencia.txt").sesiones["sesion_id"].tolist()
    b = ingerir("referencia.txt").sesiones["sesion_id"].tolist()
    assert a == b
    assert len(set(a)) == 3


# 4. Muestras previas a CALIBRADO
def test_precalibracion_marcada(referencia):
    m = referencia.mediciones
    pre = m[~m["post_calibracion"]]
    assert len(pre) == 8                       # 2 + 3 + 3
    assert (pre["t_rel_calibracion"] < 0).all()
    assert (m.loc[m["post_calibracion"], "t_rel_calibracion"] > 0).all()
    # Las dos primeras muestras (F = 29º, 28º) son montaje, no postura
    assert list(m["frontal"].iloc[:2]) == [29, 28]
    assert not m["post_calibracion"].iloc[:2].any()
    assert list(referencia.sesiones["n_precalibracion"]) == [2, 3, 3]


# 5. Sesión sin CALIBRADO: marcada y excluida por defecto
def test_sin_calibrar_marcada_y_excluida(ingerir):
    r = ingerir("sin_calibrar.txt")
    assert r.sesiones["sin_calibrar"].tolist() == [True]
    assert pd.isna(r.sesiones["calibracion"].iloc[0])
    assert r.mediciones["t_rel_calibracion"].isna().all()
    assert seleccion_analizable(r.mediciones).sum() == 0
    assert seleccion_analizable(r.mediciones, incluir_sin_calibrar=True).sum() == 3


# 6. Connection failed no abre sesión
def test_conexion_fallida_no_abre_sesion(ingerir, referencia, config):
    rec = construir(parsear_archivo(DATOS / "conexion_fallida.txt"), config)
    assert rec.filas_sesiones == [] and rec.filas_mediciones == []
    assert rec.contadores.n_conexiones_fallidas == 2
    assert referencia.contadores.n_conexiones_fallidas == 1
    # En la ingesta, un archivo sin sesiones ni mediciones se rechaza y no se procesa
    r = ingerir("conexion_fallida.txt")
    assert not r.parseos and len(r.rechazados) == 1
    assert "ninguna línea de medición" in r.rechazados[0].motivo


def test_incidencias_con_linea(config):
    texto = "\n".join([
        "10:00:00.000 F = 1º, L = 1º, BIEN",   # 1: medición antes de Connected
        "10:00:01.000 CALIBRADO",              # 2: CALIBRADO fuera de sesión
        "10:00:02.000 Connected",
        "10:00:04.000 F = 1º, L = 1º, BIEN",
        "10:00:03.000 F = 1º, L = 1º, BIEN",   # 5: la hora retrocede
        "10:00:05.000 Disconnected",
    ])
    rec = construir(parsear_texto(texto, "x.txt"), config)
    assert [(i.codigo, i.linea) for i in rec.incidencias] == [
        ("medicion_fuera_sesion", 1), ("calibrado_fuera_sesion", 2), ("fuera_de_orden", 5)]
    assert rec.contadores.n_mediciones_fuera_sesion == 1


def test_cli_rechaza_lo_que_no_es_registro(tmp_path, capsys):
    from postura.cli import principal
    (tmp_path / "bueno.txt").write_bytes((DATOS / "una_muestra.txt").read_bytes())
    (tmp_path / "malo.txt").write_bytes(b"%PDF-1.4 ...")
    salida = tmp_path / "m.parquet"
    assert principal(["ingerir", str(tmp_path / "*.txt"), "--salida", str(salida)]) == 0
    err = capsys.readouterr().err
    assert "RECHAZADO malo.txt" in err and "PDF" in err
    assert principal(["ingerir", str(tmp_path / "malo.txt"), "--salida", str(salida)]) == 1


# 8. Cierre implícito
def test_cierre_implicito(ingerir):
    r = ingerir("sin_cierre.txt")
    s = r.sesiones.iloc[0]
    assert bool(s["cierre_implicito"]) and s["evento_cierre"] == "implicito"
    assert s["fin"] == r.mediciones["t"].max()
    assert s["n_muestras"] == 4 and s["n_precalibracion"] == 1


def test_varios_archivos_y_usuarios(tmp_path, config):
    contenido = (DATOS / "una_muestra.txt").read_text(encoding="utf-8")
    rutas = []
    for nombre in ("U01_2026-10-31.txt", "U02_2026-11-01.txt"):
        (tmp_path / nombre).write_text(contenido, encoding="utf-8")
        rutas.append(tmp_path / nombre)
    r = ingerir_archivos(rutas, config)
    assert sorted(r.mediciones["usuario_id"].unique()) == ["U01", "U02"]
    assert r.sesiones["sesion_id"].is_unique
    assert not r.sesiones["fecha_desconocida"].any()
    assert r.mediciones["t"].dt.date.astype(str).tolist() == ["2026-10-31", "2026-11-01"]


def test_usuario_desconocido_por_defecto(referencia):
    assert set(referencia.mediciones["usuario_id"]) == {"desconocido"}
