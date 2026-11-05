"""Orquestación: archivos de registro → DataFrames validados de mediciones y sesiones.

Funciona igual con un archivo que con muchos archivos de muchos usuarios. Cada
archivo pasa primero por ``validacion.revisar_archivo``: los que no son registros del
corrector se rechazan (con su motivo) y no se procesan. CLI y web comparten esta regla.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import modelo
from . import parser as p
from .sesiones import Contadores, Incidencia, construir
from .validacion import revisar_archivo

log = logging.getLogger("postura.ingesta")


@dataclass(frozen=True)
class Rechazo:
    archivo: str
    bytes: int
    motivo: str


@dataclass
class ResultadoIngesta:
    mediciones: pd.DataFrame
    sesiones: pd.DataFrame
    contadores: Contadores
    parseos: list[p.ResultadoParseo]      # solo archivos aceptados
    incidencias: list[Incidencia]
    rechazados: list[Rechazo]


def expandir_rutas(patrones: list[str]) -> list[Path]:
    """Expande comodines (también en Windows, donde la shell no lo hace) y quita duplicados."""
    rutas: list[Path] = []
    for patron in patrones:
        encontrados = sorted(glob.glob(patron)) if any(ch in patron for ch in "*?[") else [patron]
        rutas.extend(Path(x) for x in encontrados)
    vistos, unicas = set(), []
    for r in rutas:
        clave = r.resolve()
        if clave not in vistos:
            vistos.add(clave)
            unicas.append(r)
    return unicas


def ingerir_archivos(rutas: list[str | Path], config: dict) -> ResultadoIngesta:
    """Ingesta desde rutas en disco. Una ruta inexistente cuenta como rechazada."""
    contenidos, faltantes = [], []
    for ruta in map(Path, rutas):
        if ruta.is_file():
            contenidos.append((ruta.name, ruta.read_bytes()))
        else:
            faltantes.append(Rechazo(ruta.name, 0, f"No existe el archivo: {ruta}"))
    resultado = ingerir_contenidos(contenidos, config)
    resultado.rechazados[:0] = faltantes
    return resultado


def ingerir_contenidos(contenidos: list[tuple[str, bytes]], config: dict) -> ResultadoIngesta:
    """Ingesta desde pares (nombre de archivo, bytes); sirve para archivos subidos por web."""
    nombres = [Path(n).name for n, _ in contenidos]
    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
    if repetidos:
        log.warning("Hay archivos con el mismo nombre: %s. Sus sesion_id colisionarán; renómbralos.",
                    repetidos)

    filas_med, filas_ses, parseos, incidencias, rechazados = [], [], [], [], []
    total = Contadores()
    for nombre, crudo in contenidos:
        parseo, motivo = revisar_archivo(nombre, crudo, config)
        if motivo:
            rechazados.append(Rechazo(Path(nombre).name, len(crudo), motivo))
            log.warning("%s: rechazado, no se procesa: %s", nombre, motivo)
            continue
        parseos.append(parseo)
        rec = construir(parseo, config)
        filas_med.extend(rec.filas_mediciones)
        filas_ses.extend(rec.filas_sesiones)
        incidencias.extend(rec.incidencias)
        total.sumar(rec.contadores)
        log.info("%s: %d líneas, %d mediciones, %d sesiones", parseo.archivo, rec.contadores.n_lineas,
                 rec.contadores.n_mediciones, len(rec.filas_sesiones))

    mediciones = _a_mediciones(filas_med, config)
    sesiones = (modelo.coaccionar_sesiones(pd.DataFrame(filas_ses)) if filas_ses
                else modelo.sesiones_vacias())
    modelo.validar_mediciones(mediciones)
    modelo.validar_sesiones(sesiones)   # incluye sesion_id duplicado entre archivos
    return ResultadoIngesta(mediciones, sesiones, total, parseos, incidencias, rechazados)


def _a_mediciones(filas: list[dict], config: dict) -> pd.DataFrame:
    if not filas:
        return modelo.mediciones_vacias()
    df = pd.DataFrame(filas)
    df["estado_derivado"] = modelo.derivar_estado(df["frontal"], df["lateral"], config["umbrales"])
    return modelo.coaccionar_mediciones(df)


def ruta_sesiones(ruta_mediciones: str | Path) -> Path:
    """salidas/mediciones.parquet → salidas/mediciones.sesiones.parquet"""
    r = Path(ruta_mediciones)
    return r.with_name(r.stem + ".sesiones.parquet")


def guardar(resultado: ResultadoIngesta, ruta_mediciones: str | Path) -> tuple[Path, Path]:
    r = Path(ruta_mediciones)
    r.parent.mkdir(parents=True, exist_ok=True)
    resultado.mediciones.to_parquet(r, index=False)
    rs = ruta_sesiones(r)
    resultado.sesiones.to_parquet(rs, index=False)
    return r, rs


def cargar(ruta_mediciones: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee mediciones y sesiones desde parquet y vuelve a validar el esquema."""
    r = Path(ruta_mediciones)
    if not r.is_file():
        raise FileNotFoundError(f"No existe el archivo de mediciones: {r}")
    rs = ruta_sesiones(r)
    if not rs.is_file():
        raise FileNotFoundError(f"No existe el archivo de sesiones asociado: {rs}")
    med = modelo.coaccionar_mediciones(pd.read_parquet(r))
    ses = modelo.coaccionar_sesiones(pd.read_parquet(rs))
    return modelo.validar_mediciones(med), modelo.validar_sesiones(ses)
