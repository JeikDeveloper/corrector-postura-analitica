"""Eventos → sesiones y mediciones normalizadas.

Reglas de dominio:

- Una sesión arranca en ``Connected`` y termina en el primero de ``Disconnected``,
  ``Connection lost`` o ``APAGADO``. ``Connection failed`` NO abre sesión.
- Si el archivo termina (o llega otro ``Connected``) sin cierre, la sesión se cierra
  en su última medición y queda con ``cierre_implicito = True``.
- ``CALIBRADO`` reubica el cero del sensor. Las muestras anteriores reflejan el
  montaje, no la postura: quedan con ``post_calibracion = False`` y se excluyen de
  estadística y figuras.
- CONSECUENCIA: los ángulos son relativos a la calibración de su propia sesión.
  Dos sesiones no son comparables en términos absolutos. Cualquier agregación de
  varias sesiones debe declararlo.
- Sesión sin ``CALIBRADO`` → ``sin_calibrar = True``; se excluye por defecto
  (``sesiones.incluir_sin_calibrar`` en configuración).

Este módulo es el único que sigue el estado de sesión abierta/cerrada. Lo que no
encaja (mediciones o CALIBRADO fuera de sesión, horas fuera de orden) se devuelve
como ``Incidencia`` con su número de línea para el informe de estructura.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import parser as p

log = logging.getLogger("postura.ingesta")

DISPOSITIVO_DESCONOCIDO = "desconocido"
CIERRE_IMPLICITO = "implicito"


@dataclass
class _SesionEnCurso:
    indice: int
    dispositivo: str
    inicio: datetime
    linea_inicio: int
    calibraciones: list[tuple[int, datetime]] = field(default_factory=list)
    mediciones: list[tuple[p.Evento, datetime]] = field(default_factory=list)
    fin: datetime | None = None
    evento_cierre: str | None = None


@dataclass
class Contadores:
    n_lineas: int = 0
    n_mediciones: int = 0
    n_mediciones_fuera_sesion: int = 0
    n_desconocidos: int = 0
    n_invalidas: int = 0
    n_conexiones_fallidas: int = 0

    def sumar(self, otro: "Contadores") -> None:
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(otro, k))


@dataclass(frozen=True)
class Incidencia:
    """Línea que la reconstrucción de sesiones no pudo usar como se esperaba."""
    codigo: str          # medicion_fuera_sesion | calibrado_fuera_sesion | fuera_de_orden
    archivo: str
    linea: int
    texto: str


@dataclass
class Reconstruccion:
    filas_mediciones: list[dict]
    filas_sesiones: list[dict]
    contadores: Contadores
    incidencias: list[Incidencia]


def resolver_usuario(nombre_archivo: str, config: dict) -> str:
    cu = config["usuarios"]
    if nombre_archivo in (cu.get("mapa_archivos") or {}):
        return str(cu["mapa_archivos"][nombre_archivo])
    patron = cu.get("patron_archivo")
    if patron and (m := re.search(patron, nombre_archivo)):
        return m["usuario"]
    return cu["por_defecto"]


def resolver_fecha(nombre_archivo: str, config: dict) -> tuple[date, bool]:
    """Devuelve (fecha_base, fecha_desconocida)."""
    cf = config["fecha"]
    patron = cf.get("patron_archivo")
    if patron and (m := re.search(patron, nombre_archivo)):
        try:
            return date.fromisoformat(m["fecha"]), False
        except ValueError:
            log.warning("%s: la fecha del nombre de archivo no es válida: %r", nombre_archivo, m["fecha"])
    if cf.get("por_defecto"):
        return date.fromisoformat(str(cf["por_defecto"])), False
    return date.fromisoformat(str(cf["fecha_marcador"])), True


def id_sesion(archivo: str, indice: int, inicio: datetime) -> str:
    """Identificador estable y determinista: depende solo del archivo, el orden y la hora."""
    return f"{Path(archivo).stem}-s{indice:02d}-{inicio:%H%M%S}{inicio.microsecond // 1000:03d}"


def construir(resultado: p.ResultadoParseo, config: dict) -> Reconstruccion:
    """Reconstruye las sesiones de un archivo parseado.

    Solo las mediciones dentro de una sesión llegan a las filas; las de fuera se
    cuentan, se avisan en el log y quedan como incidencias con su número de línea.
    """
    archivo = resultado.archivo
    usuario = resolver_usuario(archivo, config)
    fecha_base, fecha_desconocida = resolver_fecha(archivo, config)
    if fecha_desconocida:
        log.warning("%s: sin fecha en nombre ni configuración; se usa la fecha marcador %s "
                    "(las horas son válidas, la fecha no)", archivo, fecha_base)
    retroceso = float(config["fecha"]["retroceso_medianoche_s"])
    politica_cal = config["sesiones"]["calibracion_multiple"]

    c = Contadores(n_lineas=resultado.n_lineas)
    incidencias: list[Incidencia] = []
    cerradas: list[_SesionEnCurso] = []
    abierta: _SesionEnCurso | None = None
    dispositivo_pendiente: str | None = None
    dias = 0
    previo_s: float | None = None

    def cerrar_implicita(s: _SesionEnCurso, motivo: str) -> None:
        s.fin = s.mediciones[-1][1] if s.mediciones else s.inicio
        s.evento_cierre = CIERRE_IMPLICITO
        log.warning("%s: sesión iniciada en línea %d sin evento de cierre (%s); "
                    "se cierra en su última medición", archivo, s.linea_inicio, motivo)
        cerradas.append(s)

    for ev in resultado.eventos:
        if ev.tipo == p.DESCONOCIDO:
            c.n_desconocidos += 1
            continue
        if ev.tipo == p.INVALIDA:
            c.n_invalidas += 1
            continue

        segundos = ev.hora.hour * 3600 + ev.hora.minute * 60 + ev.hora.second + ev.hora.microsecond / 1e6
        if previo_s is not None and segundos < previo_s:
            if previo_s - segundos > retroceso:
                dias += 1
                log.info("%s:%d la hora retrocede de forma brusca; se asume cruce de medianoche",
                         archivo, ev.linea)
            else:
                incidencias.append(Incidencia("fuera_de_orden", archivo, ev.linea, ev.texto))
        previo_s = segundos
        t = datetime.combine(fecha_base, ev.hora) + timedelta(days=dias)

        if ev.tipo == p.MEDICION:
            c.n_mediciones += 1
            if abierta is None:
                c.n_mediciones_fuera_sesion += 1
                incidencias.append(Incidencia("medicion_fuera_sesion", archivo, ev.linea, ev.texto))
                log.warning("%s:%d medición fuera de sesión, se descarta: %r", archivo, ev.linea, ev.texto)
            else:
                abierta.mediciones.append((ev, t))
        elif ev.tipo == p.CONECTANDO:
            dispositivo_pendiente = ev.dispositivo
        elif ev.tipo == p.CONEXION_FALLIDA:
            c.n_conexiones_fallidas += 1
            log.info("%s:%d conexión fallida (gatt status %s); no abre sesión",
                     archivo, ev.linea, ev.gatt_status)
        elif ev.tipo == p.CONECTADO:
            if abierta is not None:
                cerrar_implicita(abierta, f"nuevo Connected en línea {ev.linea}")
            if dispositivo_pendiente is None:
                log.warning("%s:%d Connected sin 'Connecting to ...' previo; dispositivo desconocido",
                            archivo, ev.linea)
            abierta = _SesionEnCurso(
                indice=len(cerradas) + 1,
                dispositivo=dispositivo_pendiente or DISPOSITIVO_DESCONOCIDO,
                inicio=t, linea_inicio=ev.linea,
            )
        elif ev.tipo == p.CALIBRADO:
            if abierta is None:
                incidencias.append(Incidencia("calibrado_fuera_sesion", archivo, ev.linea, ev.texto))
                log.warning("%s:%d CALIBRADO fuera de sesión, se ignora", archivo, ev.linea)
            else:
                abierta.calibraciones.append((ev.linea, t))
        elif ev.tipo in p.EVENTOS_CIERRE:
            if abierta is None:
                log.info("%s:%d %r sin sesión abierta (normal tras un cierre previo)",
                         archivo, ev.linea, ev.texto)
            else:
                abierta.fin = t
                abierta.evento_cierre = ev.tipo
                cerradas.append(abierta)
                abierta = None

    if abierta is not None:
        cerrar_implicita(abierta, "fin de archivo")

    filas_med: list[dict] = []
    filas_ses: list[dict] = []
    for s in cerradas:
        sid = id_sesion(archivo, s.indice, s.inicio)
        if s.calibraciones:
            linea_cal, t_cal = s.calibraciones[-1] if politica_cal == "ultima" else s.calibraciones[0]
        else:
            linea_cal, t_cal = None, None
        if len(s.calibraciones) > 1:
            log.warning("%s: la sesión %s tiene %d CALIBRADO; se usa el %s como cero",
                        archivo, sid, len(s.calibraciones),
                        "último" if politica_cal == "ultima" else "primero")
        if t_cal is None:
            log.warning("%s: la sesión %s no tiene CALIBRADO (sin_calibrar)", archivo, sid)

        n_post = 0
        for ev, t in s.mediciones:
            post = linea_cal is not None and ev.linea > linea_cal
            n_post += post
            filas_med.append({
                "archivo": archivo,
                "dispositivo": s.dispositivo,
                "usuario_id": usuario,
                "sesion_id": sid,
                "t": t,
                "t_rel_sesion": (t - s.inicio).total_seconds(),
                "t_rel_calibracion": (t - t_cal).total_seconds() if t_cal else np.nan,
                "frontal": ev.frontal,
                "lateral": ev.lateral,
                "estado": ev.estado,
                "post_calibracion": post,
            })
        difs = np.diff([(t - s.inicio).total_seconds() for _, t in s.mediciones])
        filas_ses.append({
            "sesion_id": sid,
            "archivo": archivo,
            "dispositivo": s.dispositivo,
            "usuario_id": usuario,
            "indice": s.indice,
            "inicio": s.inicio,
            "calibracion": t_cal,
            "fin": s.fin,
            "evento_cierre": s.evento_cierre,
            "duracion_s": (s.fin - s.inicio).total_seconds(),
            "n_muestras": len(s.mediciones),
            # Sin CALIBRADO no hay "precalibración": todas sus muestras son sin_calibrar.
            "n_precalibracion": len(s.mediciones) - n_post if t_cal else 0,
            "n_post_calibracion": n_post,
            "n_calibraciones": len(s.calibraciones),
            "cadencia_mediana_s": float(np.median(difs)) if difs.size else np.nan,
            "sin_calibrar": t_cal is None,
            "cierre_implicito": s.evento_cierre == CIERRE_IMPLICITO,
            "calibracion_multiple": len(s.calibraciones) > 1,
            "fecha_desconocida": fecha_desconocida,
        })
    return Reconstruccion(filas_med, filas_ses, c, incidencias)
