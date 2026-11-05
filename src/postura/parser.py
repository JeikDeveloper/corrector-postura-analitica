"""Texto crudo → lista de eventos.

El parser es tolerante: nunca aborta por una línea rara. Lo que no reconoce lo
devuelve como evento ``desconocido`` (o ``invalida`` si ni siquiera trae marca de
tiempo) y lo registra en el log de ingesta con su texto literal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

log = logging.getLogger("postura.ingesta")

# Tipos de evento
MEDICION = "medicion"
CONECTANDO = "conectando"
CONECTADO = "conectado"
CALIBRADO = "calibrado"
APAGADO = "apagado"
DESCONECTADO = "desconectado"
CONEXION_PERDIDA = "conexion_perdida"
CONEXION_FALLIDA = "conexion_fallida"
DESCONOCIDO = "desconocido"
INVALIDA = "invalida"

EVENTOS_CIERRE = frozenset({APAGADO, DESCONECTADO, CONEXION_PERDIDA})

# Caracteres de grado aceptados: º (U+00BA, el oficial) y ° (U+00B0, por tolerancia).
GRADO_OFICIAL = "º"
GRADO_ALTERNATIVO = "°"
# Signos menos aceptados: guion ASCII y, por tolerancia, U+2212.
_MENOS = "-−"

_RE_MARCA = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3}) (?P<resto>.*)$")

# El signo puede venir pegado ("-14º") o separado por espacios ("- 2º").
_ANGULO = rf"(?P<{{n}}>[{_MENOS}]?\s*\d+)\s*[{GRADO_OFICIAL}{GRADO_ALTERNATIVO}]"
_RE_MEDICION = re.compile(
    r"^F\s*=\s*" + _ANGULO.format(n="f")
    + r"\s*,\s*L\s*=\s*" + _ANGULO.format(n="l")
    + r"\s*,\s*(?P<estado>BIEN|MAL)\s*$"
)
_RE_CONECTANDO = re.compile(r"^Connecting to (?P<dispositivo>.+?)\s*\.\.\.\s*$")
_RE_FALLIDA = re.compile(r"^Connection failed: gatt status (?P<gatt>\d+)\s*$")
_EVENTOS_FIJOS = {
    "Connected": CONECTADO,
    "CALIBRADO": CALIBRADO,
    "APAGADO": APAGADO,
    "Disconnected": DESCONECTADO,
    "Connection lost": CONEXION_PERDIDA,
}


@dataclass(frozen=True)
class Evento:
    linea: int                  # número de línea, base 1
    hora: time | None           # None solo para líneas inválidas
    tipo: str
    texto: str                  # texto literal de la línea (sin fin de línea)
    frontal: int | None = None
    lateral: int | None = None
    estado: str | None = None
    dispositivo: str | None = None
    gatt_status: int | None = None
    grado_alternativo: bool = False


@dataclass
class ResultadoParseo:
    archivo: str
    eventos: list[Evento] = field(default_factory=list)
    n_lineas: int = 0
    n_bytes: int = 0

    def de_tipo(self, *tipos: str) -> list[Evento]:
        return [e for e in self.eventos if e.tipo in tipos]


def parsear_angulo(texto: str) -> int:
    """Convierte "-14", "- 2", "−3" o "7" en entero. "- 2" y "-2" valen lo mismo."""
    limpio = re.sub(r"\s+", "", texto).replace("−", "-")
    return int(limpio)


def parsear_linea(texto: str, linea: int = 0) -> Evento:
    """Parsea una línea (sin fin de línea) a un Evento. Nunca lanza excepción."""
    m = _RE_MARCA.match(texto)
    if not m:
        return Evento(linea, None, INVALIDA, texto)
    try:
        hora = time(int(m["h"]), int(m["m"]), int(m["s"]), int(m["ms"]) * 1000)
    except ValueError:
        return Evento(linea, None, INVALIDA, texto)
    resto = m["resto"]

    if (mm := _RE_MEDICION.match(resto)):
        return Evento(
            linea, hora, MEDICION, texto,
            frontal=parsear_angulo(mm["f"]),
            lateral=parsear_angulo(mm["l"]),
            estado=mm["estado"],
            grado_alternativo=GRADO_ALTERNATIVO in resto,
        )
    if (mc := _RE_CONECTANDO.match(resto)):
        return Evento(linea, hora, CONECTANDO, texto, dispositivo=mc["dispositivo"])
    if (mf := _RE_FALLIDA.match(resto)):
        return Evento(linea, hora, CONEXION_FALLIDA, texto, gatt_status=int(mf["gatt"]))
    tipo = _EVENTOS_FIJOS.get(resto.strip())
    if tipo:
        return Evento(linea, hora, tipo, texto)
    return Evento(linea, hora, DESCONOCIDO, texto)


def parsear_texto(contenido: str, archivo: str = "<texto>") -> ResultadoParseo:
    """Parsea el contenido completo de un registro."""
    resultado = ResultadoParseo(archivo=archivo)
    lineas = contenido.splitlines()
    resultado.n_lineas = len(lineas)
    n_grado_alt = 0
    for i, cruda in enumerate(lineas, start=1):
        if not cruda.strip():
            continue
        ev = parsear_linea(cruda, i)
        if ev.tipo == DESCONOCIDO:
            log.warning("%s:%d evento desconocido, se ignora: %r", archivo, i, cruda)
        elif ev.tipo == INVALIDA:
            log.warning("%s:%d línea sin marca de tiempo válida, se ignora: %r", archivo, i, cruda)
        elif ev.grado_alternativo:
            n_grado_alt += 1
        resultado.eventos.append(ev)
    if n_grado_alt:
        log.info("%s: %d mediciones usan '°' (U+00B0) en lugar de 'º' (U+00BA); se aceptan",
                 archivo, n_grado_alt)
    return resultado


def parsear_archivo(ruta: str | Path, codificacion: str = "utf-8-sig") -> ResultadoParseo:
    """Lee y parsea un archivo de registro (lanza UnicodeDecodeError si no es UTF-8).

    Para archivos de origen desconocido use ``validacion.revisar_archivo``, que
    además rechaza lo que no es un registro.
    """
    ruta = Path(ruta)
    return parsear_texto(ruta.read_bytes().decode(codificacion), archivo=ruta.name)
