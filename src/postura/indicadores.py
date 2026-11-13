"""Objetivo 4: indicadores descriptivos del comportamiento postural.

Cada indicador tiene definición operativa, fórmula e interpretación para el
usuario final. Se calculan sobre las muestras analizables del filtro elegido y
también por sesión.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .ergonomia import clasificar_zona
from .estadistica import rachas

INDICADORES = [
    {"clave": "tiempo_correcto", "nombre": "Tiempo en postura correcta", "unidad": "%", "sentido": "mayor",
     "definicion": "Proporción del tiempo registrado en que el dispositivo reportó BIEN.",
     "formula": "100 · #(estado = BIEN) / n",
     "interpretacion": "Cuánto del tiempo de trabajo la postura estuvo dentro de los umbrales. Más alto es mejor."},
    {"clave": "alertas_hora", "nombre": "Frecuencia de alertas", "unidad": "episodios/h", "sentido": "menor",
     "definicion": "Número de episodios MAL (rachas consecutivas de MAL) por hora registrada.",
     "formula": "#episodios MAL / (n · cadencia / 3600)",
     "interpretacion": "Con qué frecuencia se pierde la postura correcta. Más bajo es mejor."},
    {"clave": "duracion_mal", "nombre": "Duración media de un episodio MAL", "unidad": "s", "sentido": "menor",
     "definicion": "Tiempo promedio que dura cada episodio MAL antes de volver a BIEN.",
     "formula": "Σ duración(episodios MAL) / #episodios MAL",
     "interpretacion": "Qué tan rápido se corrige la postura tras una alerta. Más corto es mejor."},
    {"clave": "primera_alerta", "nombre": "Tiempo hasta la primera alerta", "unidad": "min", "sentido": "mayor",
     "definicion": "Minutos desde el CALIBRADO hasta el primer MAL de cada sesión, promediado entre sesiones con alguna alerta.",
     "formula": "media_s( t_rel_calibración del primer MAL de la sesión s ) / 60",
     "interpretacion": "Cuánto se sostiene la buena postura al empezar. Más largo es mejor."},
    {"clave": "estabilidad", "nombre": "Índice de estabilidad postural |σ|", "unidad": "°", "sentido": "menor",
     "definicion": "Dispersión conjunta de los ángulos frontal y lateral.",
     "formula": "√(σF² + σL²)",
     "interpretacion": "Cuánto se mueve la postura alrededor de su promedio. Valores bajos = postura estable."},
    {"clave": "desplazamiento", "nombre": "Desplazamiento postural medio |R|", "unidad": "°", "sentido": "menor",
     "definicion": "Distancia entre la postura promedio y el cero fijado en la calibración.",
     "formula": "√(F̄² + L̄²)",
     "interpretacion": "Si la postura habitual se aleja de la posición calibrada. Más bajo es mejor."},
    {"clave": "zona_optima", "nombre": "Tiempo en zona óptima ISO 11226", "unidad": "%", "sentido": "mayor",
     "definicion": "Proporción de muestras en la zona óptima de la norma (frontal 0° a 20°, lateral ±10° por defecto).",
     "formula": "100 · #(zona = óptima) / n",
     "interpretacion": "Criterio ergonómico independiente de los umbrales del dispositivo. Más alto es mejor."},
    {"clave": "movilidad", "nombre": "Movilidad angular", "unidad": "°/s", "sentido": "neutro",
     "definicion": "Rapidez media del cambio de postura entre muestras consecutivas (sin contar huecos).",
     "formula": "(1/m) Σ √(ΔF² + ΔL²) / Δt",
     "interpretacion": "Muy baja: postura estática prolongada; muy alta: movimientos bruscos frecuentes."},
]


def _r(x, d=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), d)


def calcular(datos: pd.DataFrame, config: dict) -> dict:
    """Valor de cada indicador para un conjunto de muestras analizables."""
    n = len(datos)
    if n == 0:
        return {i["clave"]: None for i in INDICADORES} | {"n": 0, "sesiones_con_alerta": 0, "sesiones": 0}
    cad = float(config["calidad"]["cadencia_nominal_s"])
    hueco = float(config["calidad"]["umbral_hueco_s"])
    f = datos["frontal"].to_numpy(float)
    lat = datos["lateral"].to_numpy(float)
    r = rachas(datos, cad)
    mal = r[r["estado"] == "MAL"]
    horas = n * cad / 3600.0
    primeras = []
    movs = []
    for _, ses in datos.sort_values("t").groupby("sesion_id", observed=True):
        t_mal = ses.loc[ses["estado"] == "MAL", "t_rel_calibracion"]
        if len(t_mal):
            primeras.append(float(t_mal.min()) / 60.0)
        t = ses["t_rel_calibracion"].to_numpy(float)
        dt = np.diff(t)
        ok = (dt > 0) & (dt <= hueco)
        if ok.any():
            d = np.hypot(np.diff(ses["frontal"].to_numpy(float)), np.diff(ses["lateral"].to_numpy(float)))
            movs.append(d[ok] / dt[ok])
    mov = np.concatenate(movs) if movs else np.array([])
    return {
        "n": n, "sesiones": int(datos["sesion_id"].nunique()), "sesiones_con_alerta": len(primeras),
        "tiempo_correcto": _r(100.0 * (datos["estado"] == "BIEN").mean(), 1),
        "alertas_hora": _r(len(mal) / horas, 1) if horas else None,
        "duracion_mal": _r(mal["duracion_s"].mean(), 1) if len(mal) else None,
        "primera_alerta": _r(np.mean(primeras), 1) if primeras else None,
        "estabilidad": _r(math.hypot(f.std(ddof=0), lat.std(ddof=0))),
        "desplazamiento": _r(math.hypot(f.mean(), lat.mean())),
        "zona_optima": _r(100.0 * (clasificar_zona(f, lat, config) == "optima").mean(), 1),
        "movilidad": _r(mov.mean(), 3) if mov.size else None,
    }


def por_sesion(datos: pd.DataFrame, config: dict) -> list[dict]:
    filas = []
    for sid, ses in datos.groupby("sesion_id", observed=True, sort=False):
        filas.append({"sesion_id": str(sid), "usuario_id": str(ses["usuario_id"].iloc[0]), **calcular(ses, config)})
    return filas
