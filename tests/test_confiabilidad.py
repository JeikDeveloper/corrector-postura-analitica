import io

import matplotlib.pyplot as plt
import numpy as np
import pytest

from postura import confiabilidad as co
from postura.figuras.confiabilidad import FIGURAS, figura_confiabilidad, informe_pdf
from postura.figuras.densidad import exportar
from postura.ingesta import ingerir_contenidos
from postura.web import crear_app
from conftest import DATOS


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    yield
    plt.close("all")


def _hora(s: float) -> str:
    s_ent = int(s)
    return f"{s_ent // 3600:02d}:{(s_ent // 60) % 60:02d}:{s_ent % 60:02d}.{int(round((s % 1) * 1000)):03d}"


def _sesion(inicio_s, estados, cierre="Disconnected", huecos=()):
    """Sesión sintética: Connected, CALIBRADO y una medición por segundo; `huecos` son índices tras los que faltan 6 s."""
    lineas = [f"{_hora(inicio_s)} Connecting to WLAB CP ...", f"{_hora(inicio_s + 1)} Connected",
              f"{_hora(inicio_s + 1.5)} CALIBRADO"]
    t = inicio_s + 2
    for i, e in enumerate(estados):
        lineas.append(f"{_hora(t)} F = {30 if e == 'MAL' else 2}º, L = 1º, {e}")
        t += 1 + (6 if i in huecos else 0)
    lineas.append(f"{_hora(t)} {cierre}")
    return lineas, t


def _ingerir(lineas, config, nombre="s.txt"):
    return ingerir_contenidos([(nombre, "\n".join(lineas).encode())], config)


def _sintetico(config, n_sesiones=4, minutos=25, semilla=1):
    """Registro largo con muchos episodios MAL (precedidos de inclinación) y huecos ocasionales."""
    rng = np.random.default_rng(semilla)
    lineas, t0 = [], 8 * 3600.0
    for k in range(n_sesiones):
        estados, huecos, prox, mal = [], set(), rng.weibull(1.8) * 90, 0
        for i in range(minutos * 60):
            if rng.random() < 0.004:
                huecos.add(i)
            if mal == 0 and prox <= 0:
                mal, prox = int(rng.integers(2, 8)), rng.weibull(1.8) * 90
            estados.append("MAL" if mal else "BIEN")
            mal, prox = (mal - 1, prox) if mal else (0, prox - 1)
        ses, t0 = _sesion(t0, estados, "Connection lost" if k % 2 == 0 else "Disconnected", huecos)
        lineas += ses
        t0 += 600
    return _ingerir(lineas, config, "sintetico.txt")


# ---------------------------------------------------------------- fallos y métricas

def test_apagado_seguido_de_connection_lost_no_es_fallo(referencia, config):
    f, seg, _ = co.fallos(referencia, config, "dispositivo")
    assert "conexion_perdida" not in set(f["tipo"])       # en la referencia, el Connection lost sigue a un APAGADO
    assert list(f["tipo"]) == ["conexion_fallida"]
    assert f["reparacion_s"].iloc[0] == pytest.approx(9.83, abs=0.01)   # 16:20:35.930 → Connected 16:20:45.765


def test_tipos_de_fallo_del_dispositivo(config):
    a, t = _sesion(0, ["BIEN"] * 20, "Connection lost", huecos={9})
    b, _ = _sesion(t + 30, ["BIEN"] * 20, "Disconnected")
    c = [f"{_hora(t + 60)} Connected", f"{_hora(t + 60.5)} CALIBRADO",
         *(f"{_hora(t + 61 + i)} F = 2º, L = 1º, BIEN" for i in range(5))]           # sin cierre
    r = _ingerir(a + b + c, config)
    f, seg, _ = co.fallos(r, config, "dispositivo")
    assert sorted(f["tipo"]) == ["conexion_perdida", "hueco", "sin_cierre"]
    perdida = f[f["tipo"] == "conexion_perdida"].iloc[0]
    assert perdida["reparacion_s"] == pytest.approx(31, abs=0.01)     # hasta el siguiente Connected
    assert f.loc[f["tipo"] == "hueco", "reparacion_s"].iloc[0] == pytest.approx(7)
    m = co.metricas(f, seg)
    assert m["n_fallos"] == 3 and m["mtbf_min"] == round(m["operacion_min"] / 3, 2)


def test_metricas_postura(config):
    lineas, _ = _sesion(0, ["BIEN"] * 10 + ["MAL"] * 3 + ["BIEN"] * 7)
    f, seg, _ = co.fallos(_ingerir(lineas, config), config, "postura")
    m = co.metricas(f, seg)
    assert m["n_fallos"] == 1 and m["mttr_s"] == 3.0
    assert m["operacion_min"] == round(17 / 60, 2)                    # 17 s en BIEN
    assert m["disponibilidad_pct"] == round(100 * 17 / 20, 2)
    assert m["tasa_fallos_h"] == round(1 / (17 / 3600), 2)


# ---------------------------------------------------------------- Weibull

def test_weibull_recupera_parametros():
    from scipy.stats import weibull_min
    obs = np.random.default_rng(0).weibull(2.0, 400) * 10.0
    b, e = co.ajustar_weibull(obs, np.array([]))
    c, _, escala = weibull_min.fit(obs, floc=0)                      # sin censura coincide con el ajuste de scipy
    assert b == pytest.approx(c, rel=1e-3) and e == pytest.approx(escala, rel=1e-3)
    grande = np.random.default_rng(1).weibull(2.0, 5000) * 10.0
    b, e = co.ajustar_weibull(grande, np.array([]))
    assert b == pytest.approx(2.0, rel=0.05) and e == pytest.approx(10.0, rel=0.03)


def test_weibull_con_censura_no_subestima():
    rng = np.random.default_rng(1)
    t = rng.weibull(1.5, 600) * 20
    corte = 25.0
    obs, cens = t[t <= corte], np.full((t > corte).sum(), corte)
    _, e_cens = co.ajustar_weibull(obs, cens)
    _, e_sin = co.ajustar_weibull(obs, np.array([]))
    assert e_cens == pytest.approx(20, rel=0.08) and e_sin < e_cens   # ignorar la censura sesga hacia abajo


def test_weibull_insuficiente(referencia, config):
    f, seg, _ = co.fallos(referencia, config, "postura")
    w = co.weibull(f, seg, config)
    assert not w["ajustado"] and "al menos 5" in w["motivo"]


# ---------------------------------------------------------------- anomalías, predicción y alertas

@pytest.fixture(scope="module")
def sintetico():
    from postura.configuracion import cargar_configuracion
    cfg = cargar_configuracion()
    return _sintetico(cfg), cfg


def test_isolation_forest_determinista(sintetico):
    r, cfg = sintetico
    a1, a2 = co.analisis(r, cfg, "postura"), co.analisis(r, cfg, "postura")
    an = a1["anomalias"]
    assert an["aplicado"] and len(an["ventanas"]) == a1["n_ventanas"] > 8
    assert [x["puntaje"] for x in an["ventanas"]] == [x["puntaje"] for x in a2["anomalias"]["ventanas"]]
    assert sum(an["conteo"].values()) == len(an["ventanas"])
    assert all(0 < x["puntaje"] < 1 for x in an["ventanas"])


def test_prediccion_entrenada(sintetico):
    r, cfg = sintetico
    pr = co.analisis(r, cfg, "postura")["prediccion"]
    assert pr["entrenado"] and pr["positivos"] >= 5 and pr["negativos"] >= 5
    assert 0 <= pr["auc"] <= 1 and len(pr["coeficientes"]) == len(co.NOMBRES)
    assert len(pr["prob_validacion"]) == len(pr["etiquetas"]) == pr["positivos"] + pr["negativos"]
    assert {x["sesion_id"] for x in pr["riesgo"]} == set(r.sesiones["sesion_id"])


def test_prediccion_insuficiente(referencia, config):
    pr = co.analisis(referencia, config, "dispositivo")["prediccion"]
    assert not pr["entrenado"] and "hacen falta" in pr["motivo"]


def test_alertas_ordenadas(sintetico):
    r, cfg = sintetico
    al = co.analisis(r, cfg, "dispositivo")["alertas"]
    orden = {"critica": 0, "alta": 1, "media": 2}
    assert [orden[x["severidad"]] for x in al] == sorted(orden[x["severidad"]] for x in al)


@pytest.mark.parametrize("modo", co.MODOS)
def test_figuras_e_informe(sintetico, referencia, config, modo):
    r, cfg = sintetico
    for datos, c in ((r, cfg), (referencia, config)):         # con modelos ajustados y sin datos suficientes
        a = co.analisis(datos, c, modo)
        for nombre in FIGURAS:
            assert exportar(figura_confiabilidad(a, nombre, c), "png", c).startswith(b"\x89PNG")
        assert informe_pdf(a, c, "x").startswith(b"%PDF")


# ---------------------------------------------------------------- API

def _post(cliente, ruta, **campos):
    datos = (DATOS / "referencia.txt").read_bytes()
    return cliente.post(ruta, data={"archivos": [(io.BytesIO(datos), "referencia.txt")], **campos},
                        content_type="multipart/form-data")


def test_api_confiabilidad(config):
    c = crear_app(config).test_client()
    j = _post(c, "/api/confiabilidad/resumen", modo="postura").get_json()
    assert j["modo"] == "postura" and j["metricas"]["n_fallos"] == 2 and "_ventanas" not in j
    for parte, clave in (("metricas", "metricas"), ("weibull", "weibull"), ("anomalias", "anomalias"),
                         ("prediccion", "prediccion"), ("alertas", "alertas")):
        assert clave in _post(c, f"/api/confiabilidad/{parte}").get_json()
    assert _post(c, "/api/confiabilidad/otra").status_code == 404
    assert _post(c, "/api/confiabilidad/resumen", modo="x").status_code == 400
    r = _post(c, "/api/confiabilidad/figura.pdf", figura="weibull", descargar="1", modo="postura")
    assert r.status_code == 200 and r.data.startswith(b"%PDF") and "confiabilidad_weibull_postura_" in r.headers["Content-Disposition"]
    assert _post(c, "/api/confiabilidad/figura.png", figura="torta").status_code == 400
    assert _post(c, "/api/confiabilidad/informe.pdf").data.startswith(b"%PDF")
    csv = _post(c, "/api/confiabilidad/fallos.csv").data.decode("utf-8-sig").splitlines()
    assert csv[0] == "archivo,sesion_id,tipo,reloj_min,reparacion_s,linea,detalle" and len(csv) == 2
