"""Carga de configuración: default.yaml del repositorio con fusión de un archivo opcional."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

RUTA_DEFECTO = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
VARIABLE_ENTORNO = "POSTURA_CONFIG"


class ErrorConfiguracion(Exception):
    """Configuración ausente o inválida."""


def fusionar(base: dict, encima: dict) -> dict:
    """Copia profunda de `base` con `encima` fusionado recursivamente."""
    resultado = copy.deepcopy(base)
    for clave, valor in encima.items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = fusionar(resultado[clave], valor)
        else:
            resultado[clave] = copy.deepcopy(valor)
    return resultado


def _leer_yaml(ruta: Path) -> dict:
    if not ruta.is_file():
        raise ErrorConfiguracion(f"No existe el archivo de configuración: {ruta}")
    with ruta.open(encoding="utf-8") as f:
        datos = yaml.safe_load(f) or {}
    if not isinstance(datos, dict):
        raise ErrorConfiguracion(f"La configuración debe ser un mapa YAML: {ruta}")
    return datos


def cargar_configuracion(ruta: str | Path | None = None, **sobrescrituras: Any) -> dict:
    """Devuelve la configuración por defecto fusionada con `ruta` (o $POSTURA_CONFIG).

    `sobrescrituras` permite fusionar un dict adicional (útil en pruebas),
    p. ej. ``cargar_configuracion(sesiones={"incluir_sin_calibrar": True})``.
    """
    config = _leer_yaml(RUTA_DEFECTO)
    ruta = ruta or os.environ.get(VARIABLE_ENTORNO)
    if ruta:
        config = fusionar(config, _leer_yaml(Path(ruta)))
    if sobrescrituras:
        config = fusionar(config, sobrescrituras)
    _validar(config)
    return config


def _validar(config: dict) -> None:
    u = config.get("umbrales", {})
    for clave in ("FRONTAL_MAX", "FRONTAL_MIN", "LATERAL_MAX", "LATERAL_MIN"):
        if not isinstance(u.get(clave), (int, float)):
            raise ErrorConfiguracion(f"Falta el umbral numérico umbrales.{clave}")
    if u["FRONTAL_MIN"] >= u["FRONTAL_MAX"] or u["LATERAL_MIN"] >= u["LATERAL_MAX"]:
        raise ErrorConfiguracion("Umbrales incoherentes: cada mínimo debe ser menor que su máximo")
    metodo = config["figura"]["metodo"]
    if metodo not in ("kde", "hex", "auto"):
        raise ErrorConfiguracion(f"figura.metodo debe ser kde, hex o auto; se recibió {metodo!r}")
    if config["sesiones"]["calibracion_multiple"] not in ("ultima", "primera"):
        raise ErrorConfiguracion("sesiones.calibracion_multiple debe ser 'ultima' o 'primera'")
