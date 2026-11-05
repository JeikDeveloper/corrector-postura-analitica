import pandas as pd
import pytest

from postura import calidad, ingesta, modelo


def test_esquema_mediciones(referencia):
    m = referencia.mediciones
    assert list(m.columns) == list(modelo.ESQUEMA_MEDICIONES)
    assert m["frontal"].dtype == "int16" and m["lateral"].dtype == "int16"
    assert isinstance(m["estado"].dtype, pd.CategoricalDtype)
    assert m["post_calibracion"].dtype == bool
    modelo.validar_mediciones(m)


def test_validacion_detecta_tipo_incorrecto(referencia):
    m = referencia.mediciones.copy()
    m["frontal"] = m["frontal"].astype("float64")
    with pytest.raises(modelo.ErrorEsquema, match="frontal"):
        modelo.validar_mediciones(m)


# 7. Estado derivado == reportado en la referencia (0 discrepancias sobre 347)
def test_estado_derivado_coincide_en_referencia(referencia, config):
    m = referencia.mediciones
    assert len(m) == 347
    assert (m["estado"] == m["estado_derivado"]).all()
    informe = calidad.evaluar(m, referencia.sesiones, config)
    assert informe.n_discrepancias_estado == 0
    assert informe.alertas == []


def test_discrepancias_se_reportan(referencia, config):
    m = referencia.mediciones.copy()
    m.loc[m.index[10], "estado"] = "MAL" if m["estado"].iloc[10] == "BIEN" else "BIEN"
    informe = calidad.evaluar(m, referencia.sesiones, config)
    assert informe.n_discrepancias_estado == 1
    assert any("discrepancias" in a for a in informe.alertas)


def test_umbrales_son_estrictos(config):
    u = config["umbrales"]
    estados = modelo.derivar_estado([25, 26, -10, -11, 0, 0, 0, 0], [0, 0, 0, 0, 15, 16, -15, -16], u)
    assert list(estados) == ["BIEN", "MAL", "BIEN", "MAL", "BIEN", "MAL", "BIEN", "MAL"]


def test_calidad_cadencia_referencia(referencia, config):
    informe = calidad.evaluar(referencia.mediciones, referencia.sesiones, config)
    assert informe.n_analizables == 339
    for q in informe.sesiones:
        assert 0.9 < q.cadencia_mediana_s < 1.1
        assert q.n_huecos == 0 and q.n_duplicados_tiempo == 0 and q.n_retrocesos_tiempo == 0
    assert "INFORME DE CALIDAD" in calidad.formatear(informe, config)


def test_ida_y_vuelta_parquet(referencia, tmp_path):
    ruta = tmp_path / "mediciones.parquet"
    ingesta.guardar(referencia, ruta)
    m, s = ingesta.cargar(ruta)
    pd.testing.assert_frame_equal(m, referencia.mediciones)
    pd.testing.assert_frame_equal(s, referencia.sesiones)
