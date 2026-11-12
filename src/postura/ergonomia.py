"""Análisis ergonómico según ISO 11226 (BGIA-Report 2/2007, EN 1005-4).

Implementa lo que plantea "Posture Analysis During Desk Work of Researchers"
(Betancur, Universidad Manuela Beltrán):

- Zonas de postura óptima, aceptable y extrema en el plano lateral (β) × frontal (α).
- Métricas por sujeto i:
    |σ|ᵢ = √(σα² + σβ²)    dispersión de la postura
    |R|ᵢ = √(ᾱ² + β̄²)      desplazamiento de la postura media respecto al cero
  y sus cocientes respecto al conjunto: σGᵢ = |σ|ᵢ / |σ|global, RGᵢ = |R|ᵢ / |R|global.
- Comparación entre mañana, tarde y noche con ANOVA de un factor.

Los ángulos son relativos al CALIBRADO de cada sesión, que el dispositivo toma
como postura erguida: las zonas se aplican sobre ese cero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from .modelo import seleccion_analizable

NIVELES = ("optima", "aceptable", "extrema")


# ---------------------------------------------------------------- clasificación

def clasificar_zona(frontal, lateral, config: dict) -> np.ndarray:
    """Zona de cada punto (frontal, lateral): 'optima', 'aceptable' o 'extrema'."""
    f = np.asarray(frontal, dtype=float)
    lat = np.asarray(lateral, dtype=float)
    zona = np.full(f.shape, "extrema", dtype=object)
    libre = np.ones(f.shape, dtype=bool)
    for z in config["ergonomia"]["zonas"]:   # gana la primera que contiene el punto
        (f0, f1), (l0, l1) = z["frontal"], z["lateral"]
        dentro = libre & (f >= f0) & (f <= f1) & (lat >= l0) & (lat <= l1)
        zona[dentro] = z["nivel"]
        libre &= ~dentro
    return zona


def clasificar_eje(valores, eje: str, config: dict) -> np.ndarray:
    """Zona de un solo ángulo ('frontal' o 'lateral'), para los histogramas."""
    v = np.asarray(valores, dtype=float)
    lim = config["ergonomia"]["zonas_1d"][eje]
    zona = np.full(v.shape, "extrema", dtype=object)
    a0, a1 = lim["aceptable"]
    o0, o1 = lim["optima"]
    zona[(v >= a0) & (v <= a1)] = "aceptable"
    zona[(v >= o0) & (v <= o1)] = "optima"
    return zona


def momento_del_dia(t: pd.Series, config: dict) -> pd.Series:
    """Clave de la franja horaria de cada instante (o None si ninguna la cubre)."""
    hora = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    resultado = pd.Series(None, index=t.index, dtype=object)
    for clave, m in config["ergonomia"]["momentos"].items():
        d, h = float(m["desde"]), float(m["hasta"])
        dentro = (hora >= d) & (hora < h) if d < h else (hora >= d) | (hora < h)
        resultado[dentro & resultado.isna()] = clave
    return resultado


# ---------------------------------------------------------------- selección

def seleccionar(mediciones: pd.DataFrame, config: dict, usuario: str | None = None,
                sesion: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(muestras analizables del filtro, muestras analizables de todo el conjunto)."""
    incluir_sc = bool(config["sesiones"]["incluir_sin_calibrar"])
    global_ = mediciones.loc[seleccion_analizable(mediciones, incluir_sc)]
    filtro = pd.Series(True, index=global_.index)
    if usuario:
        filtro &= global_["usuario_id"] == usuario
    if sesion:
        filtro &= global_["sesion_id"] == sesion
    return global_.loc[filtro], global_


# ---------------------------------------------------------------- métricas

@dataclass
class Metricas:
    n: int
    media_frontal: float      # ᾱ
    media_lateral: float      # β̄
    sd_frontal: float         # σα (desviación estándar poblacional)
    sd_lateral: float         # σβ
    dispersion: float         # |σ|
    desplazamiento: float     # |R|
    pct_zona: dict            # {'optima': %, 'aceptable': %, 'extrema': %}

    def a_dict(self) -> dict:
        def limpio(x):
            return None if isinstance(x, float) and not math.isfinite(x) else x
        return {k: limpio(v) for k, v in self.__dict__.items()}


def metricas(datos: pd.DataFrame, config: dict) -> Metricas:
    n = len(datos)
    if n == 0:
        nan = float("nan")
        return Metricas(0, nan, nan, nan, nan, nan, nan, {z: None for z in NIVELES})
    f = datos["frontal"].to_numpy(dtype=float)
    lat = datos["lateral"].to_numpy(dtype=float)
    ma, mb = float(f.mean()), float(lat.mean())
    sa, sb = float(f.std(ddof=0)), float(lat.std(ddof=0))
    zonas = clasificar_zona(f, lat, config)
    pct = {z: round(100.0 * float((zonas == z).sum()) / n, 1) for z in NIVELES}
    return Metricas(n, round(ma, 2), round(mb, 2), round(sa, 2), round(sb, 2),
                    round(math.hypot(sa, sb), 2), round(math.hypot(ma, mb), 2), pct)


def _cociente(a: float, b: float) -> float | None:
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return None
    return round(a / b, 3)


def tabla_metricas(global_: pd.DataFrame, config: dict, por_defecto_usuario: str) -> list[dict]:
    """Filas: conjunto completo, cada persona y cada sesión, con σG y RG respecto al conjunto."""
    g = metricas(global_, config)
    filas = []

    def fila(tipo: str, clave: str | None, etiqueta: str, datos: pd.DataFrame) -> dict:
        m = metricas(datos, config)
        return {"tipo": tipo, "clave": clave, "etiqueta": etiqueta, **m.a_dict(),
                "sigma_g": _cociente(m.dispersion, g.dispersion),
                "r_g": _cociente(m.desplazamiento, g.desplazamiento)}

    filas.append(fila("conjunto", None, "Conjunto completo", global_))
    for u in sorted(global_["usuario_id"].astype(str).unique()):
        etiqueta = "Sin identificar" if u == por_defecto_usuario else u
        filas.append(fila("persona", u, etiqueta, global_[global_["usuario_id"] == u]))
    for s in global_["sesion_id"].astype(str).unique():
        filas.append(fila("sesion", s, s, global_[global_["sesion_id"] == s]))
    return filas


# ---------------------------------------------------------------- momentos del día

def comparar_momentos(global_: pd.DataFrame, datos: pd.DataFrame, config: dict) -> dict:
    """Resumen por momento del día y ANOVA de σG y RG entre momentos.

    Observaciones del ANOVA: tramos sesión × momento del filtro actual con al menos
    ``anova.min_muestras_tramo`` muestras; σG y RG se calculan respecto al conjunto.
    """
    ce = config["ergonomia"]
    minimo = int(ce["anova"]["min_muestras_tramo"])
    alfa = float(ce["anova"]["alfa"])
    g = metricas(global_, config)
    momento = momento_del_dia(datos["t"], config)

    resumen, grupos_s, grupos_r = [], {}, {}
    for clave, m in ce["momentos"].items():
        en_momento = datos[momento == clave]
        met = metricas(en_momento, config)
        tramos_s, tramos_r = [], []
        for _, tramo in en_momento.groupby("sesion_id", observed=True):
            if len(tramo) >= minimo:
                mt = metricas(tramo, config)
                s, r = _cociente(mt.dispersion, g.dispersion), _cociente(mt.desplazamiento, g.desplazamiento)
                if s is not None and r is not None:
                    tramos_s.append(s)
                    tramos_r.append(r)
        grupos_s[clave], grupos_r[clave] = tramos_s, tramos_r
        resumen.append({
            "clave": clave, "nombre": m["nombre"], "desde": m["desde"], "hasta": m["hasta"],
            "n_muestras": met.n, "n_tramos": len(tramos_s),
            "sigma_g_media": round(float(np.mean(tramos_s)), 3) if tramos_s else None,
            "r_g_media": round(float(np.mean(tramos_r)), 3) if tramos_r else None,
            "dispersion": met.a_dict()["dispersion"], "desplazamiento": met.a_dict()["desplazamiento"],
            "pct_zona": met.a_dict()["pct_zona"],
        })

    def anova(grupos: dict) -> dict:
        validos = [v for v in grupos.values() if len(v) >= 2]
        if len(validos) < 2:
            return {"p": None, "f": None, "significativo": None,
                    "motivo": (f"Se necesitan al menos 2 momentos con 2 tramos de ≥ {minimo} muestras "
                               f"cada uno; hay {len(validos)}.")}
        if all(np.ptp(v) == 0 for v in validos):
            return {"p": None, "f": None, "significativo": None,
                    "motivo": "Sin variación dentro de los momentos: el ANOVA no es calculable."}
        res = f_oneway(*validos)
        p = float(res.pvalue)
        return {"p": round(p, 5), "f": round(float(res.statistic), 4), "significativo": p < alfa,
                "motivo": None}

    return {"momentos": resumen, "alfa": alfa, "min_muestras_tramo": minimo,
            "anova_sigma_g": anova(grupos_s), "anova_r_g": anova(grupos_r)}
