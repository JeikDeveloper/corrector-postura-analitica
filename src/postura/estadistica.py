"""Análisis descriptivo de patrones angulares: F (frontal) y L (lateral) en BIEN vs MAL.

Todo es estadística descriptiva sobre las muestras analizables (posteriores al
CALIBRADO de su sesión) del filtro elegido; no se ajustan modelos ni se infiere nada.

- ``descriptivos``: n, media, desviación estándar, mínimo, percentiles, máximo,
  rango intercuartílico, asimetría y curtosis de cada ángulo, por estado.
- ``episodios``: rachas consecutivas de BIEN o MAL dentro de cada sesión (cuántas
  y cuánto duran).
- ``causas_mal``: qué umbral excede cada muestra MAL (puede ser más de uno).

El estado es el que reporta el dispositivo en cada línea (columna ``estado``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ESTADOS = ("BIEN", "MAL")
ANGULOS = ("frontal", "lateral")


def _r(x, d: int = 2):
    """Redondeo que deja None en lugar de NaN (para JSON)."""
    if x is None:
        return None
    x = float(x)
    return round(x, d) if math.isfinite(x) else None


def resumen(valores: np.ndarray, percentiles: list[float]) -> dict:
    v = np.asarray(valores, dtype=float)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "media": None, "sd": None, "min": None, "max": None, "iqr": None,
                "asimetria": None, "curtosis": None, "percentiles": {f"{p:g}": None for p in percentiles}}
    pct = np.percentile(v, percentiles)
    p25, p75 = np.percentile(v, [25, 75])
    varia = n > 2 and np.ptp(v) > 0
    return {
        "n": n,
        "media": _r(v.mean()),
        "sd": _r(v.std(ddof=1)) if n > 1 else None,     # desviación estándar muestral
        "min": _r(v.min()), "max": _r(v.max()),
        "iqr": _r(p75 - p25),
        "asimetria": _r(skew(v), 3) if varia else None,
        "curtosis": _r(kurtosis(v), 3) if n > 3 and np.ptp(v) > 0 else None,   # exceso (normal = 0)
        "percentiles": {f"{p:g}": _r(x) for p, x in zip(percentiles, pct)},
    }


def descriptivos(datos: pd.DataFrame, config: dict) -> list[dict]:
    """Una fila por ángulo × grupo (Todos, BIEN, MAL)."""
    pcts = [float(p) for p in config["estadistica"]["percentiles"]]
    filas = []
    for angulo in ANGULOS:
        for grupo in ("Todos",) + ESTADOS:
            sub = datos if grupo == "Todos" else datos[datos["estado"] == grupo]
            filas.append({"angulo": angulo, "grupo": grupo, **resumen(sub[angulo].to_numpy(), pcts)})
    return filas


def rachas(datos: pd.DataFrame, cadencia_s: float) -> pd.DataFrame:
    """Rachas consecutivas del mismo estado dentro de cada sesión.

    Columnas: sesion_id, estado, inicio_s, fin_s (segundos desde la calibración),
    n (muestras), duracion_s (último − primero + una cadencia nominal).
    """
    filas = []
    for sid, ses in datos.sort_values("t").groupby("sesion_id", observed=True, sort=False):
        estado = ses["estado"].astype(str).to_numpy()
        t = ses["t_rel_calibracion"].to_numpy(dtype=float)
        if not len(estado):
            continue
        cortes = np.flatnonzero(estado[1:] != estado[:-1]) + 1
        for ini, fin in zip(np.r_[0, cortes], np.r_[cortes, len(estado)]):
            filas.append({"sesion_id": str(sid), "estado": estado[ini], "inicio_s": t[ini], "fin_s": t[fin - 1],
                          "n": int(fin - ini), "duracion_s": float(t[fin - 1] - t[ini] + cadencia_s)})
    return pd.DataFrame(filas, columns=["sesion_id", "estado", "inicio_s", "fin_s", "n", "duracion_s"])


def episodios(datos: pd.DataFrame, config: dict) -> dict:
    """Cuántas rachas BIEN y MAL hay y cuánto duran (segundos)."""
    r = rachas(datos, float(config["calidad"]["cadencia_nominal_s"]))
    salida = {}
    for e in ESTADOS:
        d = r.loc[r["estado"] == e, "duracion_s"].to_numpy(dtype=float)
        salida[e] = {
            "n_episodios": int(d.size),
            "duracion_media_s": _r(d.mean(), 1) if d.size else None,
            "duracion_mediana_s": _r(np.median(d), 1) if d.size else None,
            "duracion_max_s": _r(d.max(), 1) if d.size else None,
            "tiempo_total_s": _r(d.sum(), 1) if d.size else 0.0,
        }
    return salida


def causas_mal(datos: pd.DataFrame, config: dict) -> list[dict]:
    """Por cada umbral, cuántas muestras MAL lo exceden (una muestra puede exceder dos)."""
    u = config["umbrales"]
    mal = datos[datos["estado"] == "MAL"]
    n = len(mal)
    f, lat = mal["frontal"], mal["lateral"]
    reglas = [
        ("frontal_adelante", f"Frontal > {u['FRONTAL_MAX']}° (inclinado hacia adelante)", f > u["FRONTAL_MAX"]),
        ("frontal_atras", f"Frontal < {u['FRONTAL_MIN']}° (inclinado hacia atrás)", f < u["FRONTAL_MIN"]),
        ("lateral_derecha", f"Lateral > {u['LATERAL_MAX']}° (inclinado a la derecha)", lat > u["LATERAL_MAX"]),
        ("lateral_izquierda", f"Lateral < {u['LATERAL_MIN']}° (inclinado a la izquierda)", lat < u["LATERAL_MIN"]),
    ]
    fuera = pd.Series(False, index=mal.index)
    filas = []
    for clave, texto, mascara in reglas:
        fuera |= mascara
        k = int(mascara.sum())
        filas.append({"clave": clave, "descripcion": texto, "n": k, "pct": _r(100 * k / n, 1) if n else None})
    k = int((~fuera).sum())
    filas.append({"clave": "sin_umbral", "descripcion": "Ningún umbral excedido (MAL reportado dentro de los umbrales)",
                  "n": k, "pct": _r(100 * k / n, 1) if n else None})
    return filas


def analisis(datos: pd.DataFrame, config: dict) -> dict:
    """Todo el análisis descriptivo en un dict serializable a JSON."""
    n = len(datos)
    n_mal = int((datos["estado"] == "MAL").sum())
    return {
        "n": n,
        "n_sesiones": int(datos["sesion_id"].nunique()),
        "n_personas": int(datos["usuario_id"].nunique()),
        "pct_mal": _r(100 * n_mal / n, 1) if n else None,
        "conteo": {"BIEN": n - n_mal, "MAL": n_mal},
        "percentiles": [float(p) for p in config["estadistica"]["percentiles"]],
        "descriptivos": descriptivos(datos, config),
        "episodios": episodios(datos, config),
        "causas_mal": causas_mal(datos, config),
        "umbrales": dict(config["umbrales"]),
    }
