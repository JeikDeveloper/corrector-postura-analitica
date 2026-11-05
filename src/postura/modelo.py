"""Esquema del DataFrame de mediciones y de sesiones, y validación de tipos.

Advertencia de dominio, válida para todas las columnas angulares:
``frontal`` y ``lateral`` son relativos al cero fijado por el CALIBRADO de su
propia sesión. Dos sesiones distintas NO son comparables en términos absolutos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ESTADOS = ["BIEN", "MAL"]
TIPO_ESTADO = pd.CategoricalDtype(categories=ESTADOS, ordered=False)

# Columna → tipo esperado. "str" admite object o el tipo string de pandas.
ESQUEMA_MEDICIONES: dict[str, str] = {
    "archivo": "str",
    "dispositivo": "str",
    "usuario_id": "str",
    "sesion_id": "str",
    "t": "datetime64[ns]",
    "t_rel_sesion": "float64",
    "t_rel_calibracion": "float64",   # NaN si la sesión no tiene CALIBRADO
    "frontal": "int16",
    "lateral": "int16",
    "estado": "category",
    "estado_derivado": "category",
    "post_calibracion": "bool",
}

ESQUEMA_SESIONES: dict[str, str] = {
    "sesion_id": "str",
    "archivo": "str",
    "dispositivo": "str",
    "usuario_id": "str",
    "indice": "int64",
    "inicio": "datetime64[ns]",
    "calibracion": "datetime64[ns]",   # NaT si sin_calibrar
    "fin": "datetime64[ns]",
    "evento_cierre": "str",
    "duracion_s": "float64",
    "n_muestras": "int64",
    "n_precalibracion": "int64",
    "n_post_calibracion": "int64",
    "n_calibraciones": "int64",
    "cadencia_mediana_s": "float64",
    "sin_calibrar": "bool",
    "cierre_implicito": "bool",
    "calibracion_multiple": "bool",
    "fecha_desconocida": "bool",
}


class ErrorEsquema(Exception):
    """El DataFrame no cumple el esquema esperado."""


def derivar_estado(frontal, lateral, umbrales: dict) -> pd.Categorical:
    """Recalcula el estado desde los ángulos con los umbrales del firmware.

    MAL si F > FRONTAL_MAX o F < FRONTAL_MIN o L > LATERAL_MAX o L < LATERAL_MIN.
    """
    f = np.asarray(frontal)
    l = np.asarray(lateral)
    mal = (
        (f > umbrales["FRONTAL_MAX"]) | (f < umbrales["FRONTAL_MIN"])
        | (l > umbrales["LATERAL_MAX"]) | (l < umbrales["LATERAL_MIN"])
    )
    return pd.Categorical(np.where(mal, "MAL", "BIEN"), dtype=TIPO_ESTADO)


def discrepancias(mediciones: pd.DataFrame) -> pd.Series:
    """Máscara de mediciones cuyo estado reportado difiere del derivado de los umbrales."""
    return mediciones["estado"] != mediciones["estado_derivado"]


def _cumple(serie: pd.Series, esperado: str) -> bool:
    if esperado == "str":
        return pd.api.types.is_string_dtype(serie) or serie.dtype == object
    if esperado == "category":
        return isinstance(serie.dtype, pd.CategoricalDtype)
    if esperado.startswith("datetime64"):
        return pd.api.types.is_datetime64_dtype(serie)
    return str(serie.dtype) == esperado


def validar(df: pd.DataFrame, esquema: dict[str, str], nombre: str) -> pd.DataFrame:
    """Lanza ErrorEsquema si faltan columnas o algún tipo no coincide."""
    faltan = [c for c in esquema if c not in df.columns]
    if faltan:
        raise ErrorEsquema(f"{nombre}: faltan columnas {faltan}")
    malos = [
        f"{c} (esperado {t}, encontrado {df[c].dtype})"
        for c, t in esquema.items() if not _cumple(df[c], t)
    ]
    if malos:
        raise ErrorEsquema(f"{nombre}: tipos incorrectos en " + "; ".join(malos))
    return df


def validar_mediciones(df: pd.DataFrame) -> pd.DataFrame:
    validar(df, ESQUEMA_MEDICIONES, "mediciones")
    # Al coaccionar, un estado que no sea BIEN/MAL queda como NaN: aquí se detecta.
    for col in ("estado", "estado_derivado"):
        if df[col].isna().any():
            raise ErrorEsquema(f"mediciones.{col}: hay valores vacíos o distintos de {ESTADOS}")
    if df["sesion_id"].isna().any():
        raise ErrorEsquema("mediciones: hay filas sin sesion_id")
    return df


def validar_sesiones(df: pd.DataFrame) -> pd.DataFrame:
    validar(df, ESQUEMA_SESIONES, "sesiones")
    if df["sesion_id"].duplicated().any():
        raise ErrorEsquema("sesiones: sesion_id duplicado")
    return df


def mediciones_vacias() -> pd.DataFrame:
    """DataFrame de mediciones sin filas pero con el esquema correcto."""
    return coaccionar_mediciones(pd.DataFrame({c: [] for c in ESQUEMA_MEDICIONES}))


def sesiones_vacias() -> pd.DataFrame:
    return coaccionar_sesiones(pd.DataFrame({c: [] for c in ESQUEMA_SESIONES}))


def _coaccionar(df: pd.DataFrame, esquema: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for c, t in esquema.items():
        if t == "str":
            df[c] = df[c].astype(object).where(df[c].notna(), None).astype("string")
        elif t == "category":
            df[c] = pd.Categorical(df[c], dtype=TIPO_ESTADO)
        elif t.startswith("datetime64"):
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
        else:
            df[c] = df[c].astype(t)
    return df[list(esquema)]


def coaccionar_mediciones(df: pd.DataFrame) -> pd.DataFrame:
    return _coaccionar(df, ESQUEMA_MEDICIONES)


def coaccionar_sesiones(df: pd.DataFrame) -> pd.DataFrame:
    return _coaccionar(df, ESQUEMA_SESIONES)


def seleccion_analizable(mediciones: pd.DataFrame, incluir_sin_calibrar: bool = False) -> pd.Series:
    """Máscara de muestras aptas para estadística y figuras.

    Siempre excluye las muestras previas al CALIBRADO de su sesión (reflejan el
    montaje, no la postura). Las sesiones sin CALIBRADO (t_rel_calibracion NaN)
    se excluyen salvo que ``incluir_sin_calibrar`` sea True.
    """
    mascara = mediciones["post_calibracion"].astype(bool)
    if incluir_sin_calibrar:
        mascara = mascara | mediciones["t_rel_calibracion"].isna()
    return mascara
