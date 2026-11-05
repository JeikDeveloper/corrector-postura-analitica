"""Filtro de archivos y validación de estructura con hallazgos ubicados por archivo y línea.

1. ``revisar_archivo`` decide si un archivo es un registro del corrector. Lo que no lo
   es (otra extensión, vacío, binario, no UTF-8, sin marcas de tiempo, sin mediciones o
   sin ``Connected``) se rechaza y no se procesa.
2. ``validar_estructura`` revisa los archivos aceptados y da un veredicto:
   - ``con_advertencias``: hay líneas o sesiones que no cumplen el formato (eventos
     desconocidos, mediciones fuera de sesión, discrepancias de estado...).
   - ``correcta``: cumple la especificación del formato de entrada.
   Los hallazgos ``info`` (p. ej. fecha ausente) no cambian el veredicto.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import parser as p
from .modelo import derivar_estado

if TYPE_CHECKING:
    from .ingesta import ResultadoIngesta

ADVERTENCIA, INFO = "advertencia", "info"

DESCRIPCIONES = {
    "evento_desconocido": "Evento desconocido (se ignora)",
    "linea_invalida": "Línea sin marca de tiempo HH:MM:SS.mmm válida",
    "fuera_de_orden": "La hora retrocede respecto a la línea anterior",
    "medicion_fuera_sesion": "Medición fuera de una sesión (se descarta)",
    "calibrado_fuera_sesion": "CALIBRADO fuera de una sesión (se ignora)",
    "discrepancia_estado": "El estado reportado no coincide con el derivado de los umbrales",
    "sin_calibrar": "Sesión sin CALIBRADO (excluida del análisis por defecto)",
    "cierre_implicito": "Sesión sin evento de cierre (cerrada en su última medición)",
    "calibracion_multiple": "Sesión con varios CALIBRADO (vale el último)",
    "grado_alternativo": "Usa '°' (U+00B0) en lugar de 'º' (U+00BA); se acepta",
    "fecha_desconocida": "El nombre de archivo no trae fecha AAAA-MM-DD; se usa una fecha marcador",
}

SEVERIDAD = {
    "evento_desconocido": ADVERTENCIA, "linea_invalida": ADVERTENCIA, "fuera_de_orden": ADVERTENCIA,
    "medicion_fuera_sesion": ADVERTENCIA, "calibrado_fuera_sesion": ADVERTENCIA,
    "discrepancia_estado": ADVERTENCIA, "sin_calibrar": ADVERTENCIA, "cierre_implicito": ADVERTENCIA,
    "calibracion_multiple": INFO, "grado_alternativo": INFO, "fecha_desconocida": INFO,
}

# Firmas de formatos binarios frecuentes que la gente sube por error.
_FIRMAS_BINARIAS = {
    b"%PDF": "un PDF",
    b"PK\x03\x04": "un documento de Office o un ZIP",
    b"\x89PNG": "una imagen PNG",
    b"\xff\xd8\xff": "una imagen JPG",
    b"GIF8": "una imagen GIF",
    b"\xd0\xcf\x11\xe0": "un documento antiguo de Office",
}


@dataclass
class Hallazgo:
    codigo: str
    severidad: str
    archivo: str
    linea: int | None
    texto: str
    detalle: str = ""


@dataclass
class InformeEstructura:
    veredicto: str
    resumen: dict
    conteo: dict                       # codigo → número total de hallazgos
    hallazgos: list[Hallazgo] = field(default_factory=list)   # recortado a `max_por_codigo`

    def a_dict(self) -> dict:
        d = asdict(self)
        d["descripciones"] = {c: DESCRIPCIONES[c] for c in self.conteo}
        d["severidades"] = {c: SEVERIDAD[c] for c in self.conteo}
        return d


def revisar_archivo(nombre: str, crudo: bytes, config: dict) -> tuple[p.ResultadoParseo | None, str | None]:
    """Devuelve ``(parseo, None)`` si es un registro del corrector, o ``(None, motivo)``."""
    cv = config["validacion"]
    nombre = Path(nombre).name
    permitidas = [e.lower() for e in cv["extensiones_permitidas"]]
    extension = Path(nombre).suffix.lower()
    if extension not in permitidas:
        return None, (f"Solo se aceptan archivos {', '.join(permitidas)}; este tiene extensión "
                      f"{extension or '(ninguna)'}.")
    if not crudo.strip():
        return None, "El archivo está vacío."
    for firma, tipo in _FIRMAS_BINARIAS.items():
        if crudo.startswith(firma):
            return None, f"Es {tipo}, no un registro de texto, aunque su nombre termine en {extension}."
    if b"\x00" in crudo[: int(cv["bytes_revision_binaria"])]:
        return None, "Es un archivo binario, no un registro de texto."
    try:
        texto = crudo.decode(config["ingesta"]["codificacion"])
    except UnicodeDecodeError:
        return None, "No es texto UTF-8. Guarde el registro con codificación UTF-8."
    # Vistazo rápido antes de parsear todo: evita recorrer miles de líneas de otro tipo de texto.
    primeras = [l for l in texto.splitlines() if l.strip()][: int(cv["lineas_muestra_formato"])]
    if all(p.parsear_linea(l).tipo == p.INVALIDA for l in primeras):
        return None, (f"Ninguna de las primeras {len(primeras)} líneas empieza con una marca de tiempo "
                      "HH:MM:SS.mmm; no parece un registro del corrector.")
    r = p.parsear_texto(texto, archivo=nombre)
    r.n_bytes = len(crudo)
    if not r.de_tipo(p.MEDICION):
        return None, "No tiene ninguna línea de medición (por ejemplo «16:15:23.487 F = 2º, L = 2º, BIEN»)."
    if not r.de_tipo(p.CONECTADO):
        return None, "No tiene ninguna sesión: falta la línea «Connected»."
    return r, None


def _hallazgo(codigo, archivo, linea=None, texto="", detalle="") -> Hallazgo:
    return Hallazgo(codigo, SEVERIDAD[codigo], archivo, linea, texto, detalle)


def _hallazgos_de_lineas(r: p.ResultadoParseo, umbrales: dict) -> list[Hallazgo]:
    h: list[Hallazgo] = []
    mediciones = []
    for ev in r.eventos:
        if ev.tipo == p.DESCONOCIDO:
            h.append(_hallazgo("evento_desconocido", r.archivo, ev.linea, ev.texto))
        elif ev.tipo == p.INVALIDA:
            h.append(_hallazgo("linea_invalida", r.archivo, ev.linea, ev.texto))
        elif ev.tipo == p.MEDICION:
            mediciones.append(ev)
            if ev.grado_alternativo:
                h.append(_hallazgo("grado_alternativo", r.archivo, ev.linea, ev.texto))
    if mediciones:
        derivados = derivar_estado([e.frontal for e in mediciones], [e.lateral for e in mediciones], umbrales)
        for ev, derivado in zip(mediciones, derivados):
            if derivado != ev.estado:
                h.append(_hallazgo("discrepancia_estado", r.archivo, ev.linea, ev.texto,
                                   f"reportado {ev.estado}, derivado {derivado}"))
    return h


def validar_estructura(resultado: ResultadoIngesta, config: dict,
                       max_por_codigo: int | None = None) -> InformeEstructura:
    hallazgos: list[Hallazgo] = []
    for r in resultado.parseos:
        hallazgos.extend(_hallazgos_de_lineas(r, config["umbrales"]))
    hallazgos.extend(_hallazgo(i.codigo, i.archivo, i.linea, i.texto) for i in resultado.incidencias)
    sesiones = resultado.sesiones
    for _, s in sesiones.iterrows():
        for codigo in ("sin_calibrar", "cierre_implicito", "calibracion_multiple"):
            if bool(s[codigo]):
                hallazgos.append(_hallazgo(codigo, s["archivo"], detalle=s["sesion_id"]))
    for archivo in sesiones.loc[sesiones["fecha_desconocida"], "archivo"].unique():
        hallazgos.append(_hallazgo("fecha_desconocida", archivo))

    conteo = dict(Counter(x.codigo for x in hallazgos))
    veredicto = "con_advertencias" if any(x.severidad == ADVERTENCIA for x in hallazgos) else "correcta"

    if max_por_codigo is not None:
        vistos: Counter = Counter()
        recortados = []
        for x in hallazgos:
            vistos[x.codigo] += 1
            if vistos[x.codigo] <= max_por_codigo:
                recortados.append(x)
        hallazgos = recortados

    orden = {ADVERTENCIA: 0, INFO: 1}
    hallazgos.sort(key=lambda x: (orden[x.severidad], x.archivo, x.linea or 0))
    resumen = {
        "lineas": sum(r.n_lineas for r in resultado.parseos),
        "mediciones": sum(len(r.de_tipo(p.MEDICION)) for r in resultado.parseos),
        "sesiones": int(len(sesiones)),
        "dispositivos": sorted(sesiones["dispositivo"].astype(str).unique().tolist()),
    }
    return InformeEstructura(veredicto, resumen, conteo, hallazgos)
