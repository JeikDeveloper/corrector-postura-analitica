"""Confiabilidad y tasa de fallos: MTBF, MTTR, disponibilidad, Weibull, anomalías y predicción.

Dos lecturas de "fallo" (``modo``):

- ``dispositivo``: pérdida de conexión (``Connection lost``), conexión fallida
  (``Connection failed``), hueco de datos dentro de una sesión (intervalo mayor a
  ``calidad.umbral_hueco_s``) y sesión sin evento de cierre. Un ``Connection lost`` que
  llega después de un ``APAGADO`` no cuenta: es la desconexión esperada al apagar. El tiempo de operación es
  el tiempo con datos; la reparación, lo que tarda en volver a haber datos.
- ``postura``: cada episodio MAL. El tiempo de operación es el tiempo en BIEN y la
  reparación, lo que tarda en volver a BIEN.

Métricas: MTBF = tiempo de operación / fallos; MTTR = media de los tiempos de
reparación conocidos; disponibilidad = operación / (operación + reparación);
tasa de fallos λ = fallos / hora de operación.

Weibull: ajuste por máxima verosimilitud de los tiempos entre fallos, con los
tramos finales sin fallo como datos censurados. β < 1: los fallos se concentran al
principio; β ≈ 1: ocurren al azar; β > 1: el riesgo crece con el uso (degradación).

Isolation Forest (anomalías) y regresión logística (fallo inminente) trabajan sobre
ventanas de ``confiabilidad.ventana_s`` segundos. Si no hay datos suficientes, no se
entrena nada: se devuelve el motivo.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma

from . import parser as p
from .estadistica import rachas
from .modelo import discrepancias, seleccion_analizable

MODOS = ("dispositivo", "postura")
TIPOS = {
    "conexion_perdida": "Pérdida de conexión",
    "conexion_fallida": "Conexión fallida",
    "hueco": "Hueco de datos",
    "sin_cierre": "Sesión sin cierre",
    "mal": "Episodio MAL",
}
CARACTERISTICAS = [
    ("completitud", "%", "Muestras presentes respecto a las esperadas a la cadencia nominal."),
    ("hueco_max_s", "s", "Mayor intervalo sin datos dentro de la ventana."),
    ("irregularidad_s", "s", "Desviación estándar del intervalo entre muestras (cadencia irregular)."),
    ("salto_frontal", "°", "Mayor cambio de F entre dos muestras seguidas."),
    ("salto_lateral", "°", "Mayor cambio de L entre dos muestras seguidas."),
    ("sd_frontal", "°", "Variabilidad de F."),
    ("sd_lateral", "°", "Variabilidad de L."),
    ("abs_frontal", "°", "Inclinación frontal media en valor absoluto."),
    ("abs_lateral", "°", "Inclinación lateral media en valor absoluto."),
    ("pct_mal", "%", "Porcentaje de muestras MAL."),
    ("transiciones", "cambios", "Cambios entre BIEN y MAL."),
    ("discrepancias", "muestras", "Estado reportado distinto del derivado de los umbrales."),
]
NOMBRES = [c[0] for c in CARACTERISTICAS]


def _r(x, d=2):
    if x is None:
        return None
    x = float(x)
    return round(x, d) if math.isfinite(x) else None


def _c(x) -> str:
    """Número con coma decimal, para los textos de la página."""
    return str(x).replace(".", ",")


def _seg_dia(h) -> float:
    return h.hour * 3600 + h.minute * 60 + h.second + h.microsecond / 1e6


# ---------------------------------------------------------------- fallos y tiempo de operación

def _fallos_dispositivo(resultado, config, usuario=None, sesion=None):
    """(fallos, segmentos de operación). Segmentos: una fila por sesión con su tiempo con datos."""
    hueco = float(config["calidad"]["umbral_hueco_s"])
    med, ses = resultado.mediciones, resultado.sesiones
    if usuario:
        ses = ses[ses["usuario_id"] == usuario]
    if sesion:
        ses = ses[ses["sesion_id"] == sesion]
    fallos, segmentos = [], []
    for parseo in resultado.parseos:
        s_arch = ses[ses["archivo"] == parseo.archivo].sort_values("indice")
        if s_arch.empty:
            continue
        m_arch = med[med["archivo"] == parseo.archivo]
        primera_linea = {sid: int(g["linea"].min()) for sid, g in m_arch.groupby("sesion_id", observed=True)}
        reloj = 0.0                                   # tiempo de operación acumulado del archivo
        filas = list(s_arch.itertuples())
        for k, s in enumerate(filas):
            m = m_arch[m_arch["sesion_id"] == s.sesion_id].sort_values("linea")
            t = m["t_rel_sesion"].to_numpy(float)
            difs = np.diff(t)
            huecos = [(t[i], difs[i], int(m["linea"].iloc[i + 1])) for i in np.flatnonzero(difs > hueco)]
            dur = float(s.duracion_s)
            perdido = sum(d for _, d, _ in huecos)
            siguiente = filas[k + 1] if k + 1 < len(filas) else None
            # Conexiones fallidas entre la sesión anterior y esta (por número de línea)
            if not sesion:
                desde = primera_linea.get(filas[k - 1].sesion_id, 0) if k else 0
                hasta = primera_linea.get(s.sesion_id, 10 ** 9)
                for e in parseo.eventos:
                    if e.tipo == p.CONEXION_FALLIDA and desde < e.linea < hasta and (k == 0 or e.linea > desde):
                        espera = (_seg_dia(s.inicio.time()) - _seg_dia(e.hora)) % 86400
                        fallos.append({"archivo": parseo.archivo, "sesion_id": None, "tipo": "conexion_fallida",
                                       "reloj_s": reloj, "t_sesion_s": None, "reparacion_s": espera, "linea": e.linea,
                                       "detalle": e.texto})
            for inicio, d, linea in huecos:
                previo = sum(dd for ii, dd, _ in huecos if ii < inicio)
                fallos.append({"archivo": parseo.archivo, "sesion_id": s.sesion_id, "tipo": "hueco",
                               "reloj_s": reloj + inicio - previo, "t_sesion_s": inicio, "reparacion_s": d,
                               "linea": linea, "detalle": f"{d:.1f} s sin datos"})
            if s.evento_cierre in ("conexion_perdida", "implicito"):
                rep = ((siguiente.inicio - s.fin).total_seconds() if siguiente is not None else None)
                fallos.append({"archivo": parseo.archivo, "sesion_id": s.sesion_id,
                               "tipo": "conexion_perdida" if s.evento_cierre == "conexion_perdida" else "sin_cierre",
                               "reloj_s": reloj + dur - perdido, "t_sesion_s": dur, "reparacion_s": rep,
                               "linea": int(m["linea"].max()) if len(m) else None,
                               "detalle": "Connection lost" if s.evento_cierre == "conexion_perdida" else "sin evento de cierre"})
            segmentos.append({"archivo": parseo.archivo, "sesion_id": s.sesion_id, "usuario_id": s.usuario_id,
                              "operacion_s": max(0.0, dur - perdido), "reloj_inicio_s": reloj})
            reloj += max(0.0, dur - perdido)
    return pd.DataFrame(fallos, columns=["archivo", "sesion_id", "tipo", "reloj_s", "t_sesion_s", "reparacion_s",
                                         "linea", "detalle"]), pd.DataFrame(segmentos)


def _fallos_postura(datos, config):
    cad = float(config["calidad"]["cadencia_nominal_s"])
    r = rachas(datos, cad)
    fallos, segmentos = [], []
    reloj_archivo: dict[str, float] = {}
    for sid, g in r.groupby("sesion_id", sort=False):
        ses = datos[datos["sesion_id"].astype(str) == sid]
        archivo = str(ses["archivo"].iloc[0])
        reloj = reloj_archivo.get(archivo, 0.0)
        inicio = reloj
        for fila in g.itertuples():
            if fila.estado == "MAL":
                lin = ses.loc[np.isclose(ses["t_rel_calibracion"].to_numpy(float), fila.inicio_s), "linea"]
                fallos.append({"archivo": archivo, "sesion_id": sid, "tipo": "mal", "reloj_s": reloj,
                               "t_sesion_s": fila.inicio_s, "reparacion_s": fila.duracion_s,
                               "linea": int(lin.iloc[0]) if len(lin) else None,
                               "detalle": f"{fila.n} muestra{'s' if fila.n != 1 else ''} MAL"})
            else:
                reloj += fila.duracion_s
        segmentos.append({"archivo": archivo, "sesion_id": sid, "usuario_id": str(ses["usuario_id"].iloc[0]),
                          "operacion_s": reloj - inicio, "reloj_inicio_s": inicio})
        reloj_archivo[archivo] = reloj
    return pd.DataFrame(fallos, columns=["archivo", "sesion_id", "tipo", "reloj_s", "t_sesion_s", "reparacion_s",
                                         "linea", "detalle"]), pd.DataFrame(segmentos)


def fallos(resultado, config, modo, usuario=None, sesion=None):
    """(fallos, segmentos de operación, mediciones usadas para las ventanas)."""
    if modo not in MODOS:
        raise ValueError(f"Modo desconocido: {modo!r} (use {', '.join(MODOS)})")
    med = resultado.mediciones
    sel = pd.Series(True, index=med.index)
    if usuario:
        sel &= med["usuario_id"] == usuario
    if sesion:
        sel &= med["sesion_id"] == sesion
    if modo == "dispositivo":
        f, seg = _fallos_dispositivo(resultado, config, usuario, sesion)
        return f, seg, med[sel].assign(t_ventana=med.loc[sel, "t_rel_sesion"])
    datos = med[sel & seleccion_analizable(med, bool(config["sesiones"]["incluir_sin_calibrar"]))]
    f, seg = _fallos_postura(datos, config)
    return f, seg, datos.assign(t_ventana=datos["t_rel_calibracion"])


# ---------------------------------------------------------------- métricas

def metricas(f: pd.DataFrame, seg: pd.DataFrame) -> dict:
    n = len(f)
    operacion = float(seg["operacion_s"].sum()) if len(seg) else 0.0
    rep = f["reparacion_s"].dropna().to_numpy(float)
    reparacion = float(rep.sum())
    horas = operacion / 3600
    return {
        "n_fallos": n,
        "por_tipo": {TIPOS[t]: int(c) for t, c in f["tipo"].value_counts().items()} if n else {},
        "operacion_min": _r(operacion / 60, 2),
        "mtbf_min": _r(operacion / 60 / n, 2) if n else None,
        "mttr_s": _r(rep.mean(), 1) if rep.size else None,
        "reparaciones_conocidas": int(rep.size),
        "disponibilidad_pct": _r(100 * operacion / (operacion + reparacion), 2) if operacion + reparacion > 0 else None,
        "tasa_fallos_h": _r(n / horas, 2) if horas > 0 else None,
    }


def metricas_por_segmento(f: pd.DataFrame, seg: pd.DataFrame) -> list[dict]:
    filas = []
    for s in seg.itertuples():
        fs = f[(f["sesion_id"] == s.sesion_id)]
        filas.append({"sesion_id": s.sesion_id, "archivo": s.archivo, "usuario_id": s.usuario_id,
                      **metricas(fs, seg[seg["sesion_id"] == s.sesion_id])})
    return filas


# ---------------------------------------------------------------- Weibull

def tiempos_entre_fallos(f: pd.DataFrame, seg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(tiempos hasta cada fallo, tramos finales sin fallo = censurados), en minutos de operación."""
    obs, cens = [], []
    for archivo, sg in seg.groupby("archivo", sort=False):
        total = float(sg["operacion_s"].sum() + sg["reloj_inicio_s"].min())
        relojes = np.sort(f.loc[f["archivo"] == archivo, "reloj_s"].to_numpy(float))
        previo = float(sg["reloj_inicio_s"].min())
        for r in relojes:
            if r - previo > 0:                      # fallos simultáneos cuentan como un solo intervalo
                obs.append((r - previo) / 60)
            previo = r
        if total - previo > 0:
            cens.append((total - previo) / 60)
    return np.array(obs), np.array(cens)


def ajustar_weibull(obs: np.ndarray, cens: np.ndarray) -> tuple[float, float]:
    """(β forma, η escala) por máxima verosimilitud con censura por la derecha."""
    def nll(theta):
        b, e = np.exp(theta)
        z_o, z_c = obs / e, cens / e
        return -(np.sum(np.log(b / e) + (b - 1) * np.log(z_o) - z_o ** b) - np.sum(z_c ** b))
    inicio = np.log([1.0, max(np.mean(np.r_[obs, cens]), 1e-6)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(nll, inicio, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    b, e = np.exp(res.x)
    return float(b), float(e)


def weibull(f: pd.DataFrame, seg: pd.DataFrame, config: dict) -> dict:
    cw = config["confiabilidad"]["weibull"]
    obs, cens = tiempos_entre_fallos(f, seg)
    base = {"n_observados": int(obs.size), "n_censurados": int(cens.size), "observados_min": np.round(obs, 3).tolist(),
            "censurados_min": np.round(cens, 3).tolist(), "horizonte_min": cw["horizonte_min"]}
    if obs.size < int(cw["min_fallos"]):
        return {**base, "ajustado": False,
                "motivo": f"Se necesitan al menos {cw['min_fallos']} tiempos entre fallos observados para ajustar Weibull; "
                          f"hay {obs.size}."}
    b, e = ajustar_weibull(obs, cens)
    h = float(cw["horizonte_min"])
    if b < 0.9:
        lectura = "β < 1: los fallos se concentran al principio (fallos tempranos); el riesgo baja con el uso."
    elif b <= 1.1:
        lectura = "β ≈ 1: los fallos ocurren al azar, con riesgo constante; no hay señal de degradación."
    else:
        lectura = "β > 1: el riesgo de fallo crece con el tiempo de uso; hay señal de degradación."
    return {**base, "ajustado": True, "motivo": None, "beta": _r(b, 3), "eta_min": _r(e, 3),
            "mttf_min": _r(e * gamma(1 + 1 / b), 3), "b10_min": _r(e * (-math.log(0.9)) ** (1 / b), 3),
            "prob_fallo_horizonte": _r(1 - math.exp(-((h / e) ** b)), 4), "lectura": lectura}


# ---------------------------------------------------------------- ventanas

def ventanas(datos: pd.DataFrame, f: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Características por ventana y etiqueta de fallo inminente (empieza un fallo en el horizonte)."""
    cc = config["confiabilidad"]
    dur = float(cc["ventana_s"])
    horizonte = float(cc["prediccion"]["horizonte_s"])
    cad = float(config["calidad"]["cadencia_nominal_s"])
    disc = discrepancias(datos) if len(datos) else pd.Series(dtype=bool)
    filas = []
    for sid, ses in datos.sort_values(["archivo", "linea"]).groupby("sesion_id", observed=True, sort=False):
        t = ses["t_ventana"].to_numpy(float)
        if not t.size:
            continue
        ev = f.loc[(f["sesion_id"] == str(sid)) & f["t_sesion_s"].notna(), "t_sesion_s"].to_numpy(float)
        fin_sesion = t.max()
        for k, ini in enumerate(np.arange(np.floor(t.min() / dur) * dur, fin_sesion + 1e-9, dur)):
            sel = (t >= ini) & (t < ini + dur)
            sub = ses[sel]
            if len(sub) < int(cc["min_muestras_ventana"]):
                continue
            tt = t[sel]
            dt = np.diff(tt) if tt.size > 1 else np.array([0.0])
            fr, la = sub["frontal"].to_numpy(float), sub["lateral"].to_numpy(float)
            est = sub["estado"].astype(str).to_numpy()
            fin = ini + dur
            proximo = bool(((ev >= fin) & (ev < fin + horizonte)).any())
            conocido = proximo or fin + horizonte <= fin_sesion or bool((ev >= fin).any())
            filas.append({
                "sesion_id": str(sid), "archivo": str(sub["archivo"].iloc[0]), "ventana": k, "inicio_s": ini,
                "linea_inicio": int(sub["linea"].iloc[0]),
                "completitud": min(100.0, 100 * len(sub) / (dur / cad)), "hueco_max_s": float(dt.max()),
                "irregularidad_s": float(dt.std()), "salto_frontal": float(np.abs(np.diff(fr)).max()) if fr.size > 1 else 0.0,
                "salto_lateral": float(np.abs(np.diff(la)).max()) if la.size > 1 else 0.0,
                "sd_frontal": float(fr.std()), "sd_lateral": float(la.std()),
                "abs_frontal": float(np.abs(fr).mean()), "abs_lateral": float(np.abs(la).mean()),
                "pct_mal": 100 * float((est == "MAL").mean()), "transiciones": int((est[1:] != est[:-1]).sum()),
                "discrepancias": int(disc.loc[sub.index].sum()),
                "fallo_proximo": proximo if conocido else None,
            })
    return pd.DataFrame(filas, columns=["sesion_id", "archivo", "ventana", "inicio_s", "linea_inicio", *NOMBRES,
                                        "fallo_proximo"])


# ---------------------------------------------------------------- Isolation Forest

def anomalias(v: pd.DataFrame, config: dict) -> dict:
    from sklearn.ensemble import IsolationForest

    ca = config["confiabilidad"]["anomalias"]
    sev = ca["severidad"]
    if len(v) < int(ca["min_ventanas"]):
        return {"aplicado": False, "motivo": f"Isolation Forest necesita al menos {ca['min_ventanas']} ventanas; hay {len(v)}.",
                "ventanas": [], "conteo": {}}
    X = v[NOMBRES].to_numpy(float)
    modelo = IsolationForest(n_estimators=int(ca["arboles"]), random_state=int(ca["semilla"]), contamination="auto")
    modelo.fit(X)
    puntaje = -modelo.score_samples(X)              # puntaje de anomalía del artículo original: 0 a 1
    media, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    z = (X - media) / sd
    salida = []
    for i, fila in enumerate(v.itertuples()):
        s = float(puntaje[i])
        nivel = ("critica" if s >= sev["critica"] else "alta" if s >= sev["alta"] else "media" if s >= sev["media"] else "normal")
        orden = np.argsort(-np.abs(z[i]))[:2]
        motivos = [f"{NOMBRES[j]} {'alto' if z[i, j] > 0 else 'bajo'} ({_c(_r(X[i, j], 2))})" for j in orden if abs(z[i, j]) >= 1]
        salida.append({"sesion_id": fila.sesion_id, "archivo": fila.archivo, "ventana": fila.ventana,
                       "inicio_s": _r(fila.inicio_s, 1), "linea_inicio": fila.linea_inicio, "puntaje": _r(s, 4),
                       "severidad": nivel, "motivos": motivos})
    conteo = {k: sum(1 for x in salida if x["severidad"] == k) for k in ("critica", "alta", "media", "normal")}
    return {"aplicado": True, "motivo": None, "ventanas": salida, "conteo": conteo, "umbrales": sev}


# ---------------------------------------------------------------- regresión logística

def prediccion(v: pd.DataFrame, config: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cp = config["confiabilidad"]["prediccion"]
    conocidas = v[v["fallo_proximo"].notna()]
    y = conocidas["fallo_proximo"].astype(bool).to_numpy().astype(int)
    pos, neg = int(y.sum()), int((1 - y).sum())
    base = {"horizonte_s": cp["horizonte_s"], "n_ventanas": len(conocidas), "positivos": pos, "negativos": neg,
            "umbral_alerta": cp["umbral_alerta"]}
    if pos < int(cp["min_positivos"]) or neg < int(cp["min_negativos"]):
        return {**base, "entrenado": False, "riesgo": [],
                "motivo": f"Para entrenar hacen falta al menos {cp['min_positivos']} ventanas seguidas de un fallo y "
                          f"{cp['min_negativos']} sin fallo; hay {pos} y {neg}."}
    X = conocidas[NOMBRES].to_numpy(float)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000,
                                                                random_state=int(cp["semilla"])))
    pliegues = max(2, min(int(cp["pliegues"]), pos, neg))
    cv = StratifiedKFold(n_splits=pliegues, shuffle=True, random_state=int(cp["semilla"]))
    prob_cv = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = prob_cv >= float(cp["umbral_alerta"])
    modelo.fit(X, y)
    coef = modelo[-1].coef_[0]
    todas = modelo.predict_proba(v[NOMBRES].to_numpy(float))[:, 1]
    riesgo = []
    for sid, g in v.assign(prob=todas).groupby("sesion_id", sort=False):
        ult = g.iloc[-1]
        riesgo.append({"sesion_id": sid, "ventana": int(ult["ventana"]), "inicio_s": _r(ult["inicio_s"], 1),
                       "probabilidad": _r(ult["prob"], 4), "alerta": bool(ult["prob"] >= float(cp["umbral_alerta"])),
                       "max_probabilidad": _r(g["prob"].max(), 4)})
    return {**base, "entrenado": True, "motivo": None, "pliegues": pliegues,
            "auc": _r(roc_auc_score(y, prob_cv), 3) if 0 < pos < len(y) else None,
            "sensibilidad": _r(float((pred & (y == 1)).sum()) / pos, 3),
            "especificidad": _r(float((~pred & (y == 0)).sum()) / neg, 3),
            "coeficientes": sorted([{"caracteristica": n, "odds_ratio": _r(math.exp(c), 3), "coef": _r(c, 3)}
                                    for n, c in zip(NOMBRES, coef)], key=lambda x: -abs(x["coef"])),
            "probabilidades": [_r(x, 4) for x in todas], "prob_validacion": [_r(x, 4) for x in prob_cv],
            "etiquetas": y.tolist(), "riesgo": riesgo}


# ---------------------------------------------------------------- alertas

def alertas(anom: dict, pred: dict, w: dict, modo: str) -> list[dict]:
    """Alertas para la página, de la más a la menos severa."""
    orden = {"critica": 0, "alta": 1, "media": 2}
    salida = []
    for a in anom.get("ventanas", []):
        if a["severidad"] in orden:
            salida.append({"severidad": a["severidad"], "origen": "Anomalía (Isolation Forest)",
                           "texto": f"Sesión {a['sesion_id']}, minuto {_c(_r(a['inicio_s'] / 60, 1))} (línea {a['linea_inicio']}): "
                                    f"comportamiento atípico, puntaje {_c(a['puntaje'])}"
                                    + (f" — {', '.join(a['motivos'])}" if a["motivos"] else "") + "."})
    for r in pred.get("riesgo", []):
        if r["alerta"]:
            nivel = "critica" if r["probabilidad"] >= 0.8 else "alta"
            salida.append({"severidad": nivel, "origen": "Predicción (regresión logística)",
                           "texto": f"Sesión {r['sesion_id']}: probabilidad {_c(_r(100 * r['probabilidad'], 1))} % de "
                                    f"{'un fallo' if modo == 'dispositivo' else 'un episodio MAL'} en los próximos "
                                    f"{pred['horizonte_s']} s tras su última ventana."})
    if w.get("ajustado") and w["beta"] > 1.1:
        salida.append({"severidad": "media", "origen": "Weibull",
                       "texto": f"β = {_c(w['beta'])}: el riesgo de fallo crece con el uso (degradación). Probabilidad de fallar en "
                                f"los próximos {w['horizonte_min']} min: {_c(_r(100 * w['prob_fallo_horizonte'], 1))} %."})
    salida.sort(key=lambda x: orden.get(x["severidad"], 9))
    return salida


def analisis(resultado, config: dict, modo: str, usuario=None, sesion=None) -> dict:
    f, seg, datos = fallos(resultado, config, modo, usuario, sesion)
    v = ventanas(datos, f, config)
    w = weibull(f, seg, config)
    an = anomalias(v, config)
    pr = prediccion(v, config)
    lista = [{**x, "tipo": TIPOS[x["tipo"]], "reloj_min": _r(x["reloj_s"] / 60, 3),
              "reparacion_s": _r(x["reparacion_s"], 2)} for x in f.drop(columns=["t_sesion_s"]).to_dict("records")]
    return {
        "modo": modo, "metricas": metricas(f, seg), "por_sesion": metricas_por_segmento(f, seg), "fallos": lista,
        "weibull": w, "anomalias": an, "prediccion": pr, "alertas": alertas(an, pr, w, modo),
        "n_ventanas": len(v), "caracteristicas": [{"nombre": n, "unidad": u, "descripcion": d} for n, u, d in CARACTERISTICAS],
        "_ventanas": v, "_fallos": f, "_segmentos": seg,
    }

