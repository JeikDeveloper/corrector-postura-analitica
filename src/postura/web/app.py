"""Servidor Flask.

Rutas:
    GET  /                                  página
    POST /api/analizar                      sube uno o varios .txt → validación + sesiones
    GET  /api/figura/<id>.<formato>         figura (png, svg, pdf, jpg); ?descargar=1 la baja como archivo
    GET  /api/datos/<id>.csv                mediciones normalizadas

Filtros de la figura por query string: sesion, usuario, metodo (kde|hex|auto),
incluir_sin_calibrar (1/0). Los análisis se guardan en memoria (los más recientes).
"""

from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from .. import parser as p
from ..configuracion import cargar_configuracion, fusionar
from ..figuras.densidad import TIPOS_MIME, exportar, figura_densidad
from ..ingesta import ingerir_contenidos
from ..modelo import ErrorEsquema, discrepancias, seleccion_analizable
from ..sesiones import resolver_usuario
from ..validacion import validar_estructura

ESTATICOS = Path(__file__).parent / "estatico"
# Súbala cuando cambie la forma de la respuesta de /api/analizar; la página la compara
# para avisar si quedó abierto un servidor viejo.
VERSION_API = 4
_BLOQUEO_FIGURAS = threading.Lock()   # matplotlib no es seguro entre hilos


class _Almacen:
    """Análisis recientes en memoria, con tope de tamaño."""

    def __init__(self, maximo: int) -> None:
        self.maximo = maximo
        self._datos: OrderedDict[str, dict] = OrderedDict()
        self._bloqueo = threading.Lock()

    def guardar(self, valor: dict) -> str:
        clave = uuid.uuid4().hex
        with self._bloqueo:
            self._datos[clave] = valor
            while len(self._datos) > self.maximo:
                self._datos.popitem(last=False)
        return clave

    def obtener(self, clave: str) -> dict | None:
        with self._bloqueo:
            return self._datos.get(clave)


def _nombre_seguro(texto: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", texto).strip("_") or "registro"


def _persona_segura(texto: str, maximo: int = 60) -> str:
    """Solo letras (con acentos), dígitos, espacio, guion, guion bajo y punto.

    Evita, entre otros, '$', que matplotlib interpretaría como fórmula en el título.
    """
    limpio = re.sub(r"[^\w .-]+", "", texto or "", flags=re.UNICODE)
    return re.sub(r"\s+", " ", limpio).strip()[:maximo]


def crear_app(config: dict | None = None) -> Flask:
    config = config or cargar_configuracion()
    cw = config["web"]
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = int(cw["max_mb_subida"]) * 1024 * 1024
    almacen = _Almacen(int(cw["max_analisis_en_memoria"]))
    formatos = [f for f in cw["formatos_descarga"] if f in TIPOS_MIME]

    @app.get("/")
    def pagina():
        return send_from_directory(ESTATICOS, "index.html")

    @app.errorhandler(413)
    def demasiado_grande(_):
        return jsonify(error=f"El archivo supera el máximo de {cw['max_mb_subida']} MB."), 413

    @app.errorhandler(400)
    @app.errorhandler(404)
    def error_api(e):
        return jsonify(error=e.description), e.code

    @app.post("/api/analizar")
    def analizar():
        subidos = [f for f in request.files.getlist("archivos") if f and f.filename]
        if not subidos:
            return jsonify(error="No se recibió ningún archivo."), 400
        contenidos = [(Path(f.filename).name, f.read()) for f in subidos]
        # Persona opcional por archivo (mismo orden que los archivos). Vacía → se resuelve
        # con el patrón del nombre de archivo o queda sin identificar.
        personas = request.form.getlist("personas")
        mapa = {nombre: persona for (nombre, _), persona
                in zip(contenidos, map(_persona_segura, personas)) if persona}
        cfg = fusionar(config, {"usuarios": {"mapa_archivos": mapa}}) if mapa else config

        try:
            resultado = ingerir_contenidos(contenidos, cfg)
        except ErrorEsquema as error:
            return jsonify(api=VERSION_API, error=f"No se pudo normalizar el registro: {error}"), 422
        rechazados = [asdict(r) for r in resultado.rechazados]
        if not resultado.parseos:
            n = len(rechazados)
            return jsonify(
                api=VERSION_API,
                error=("El archivo no es un registro válido del corrector; no se procesó." if n == 1
                       else f"Ninguno de los {n} archivos es un registro válido del corrector; no se procesaron."),
                rechazados=rechazados,
            ), 422

        med, ses = resultado.mediciones, resultado.sesiones
        estructura = validar_estructura(resultado, cfg, max_por_codigo=int(cw["max_hallazgos_por_codigo"]))
        analizables = seleccion_analizable(med, bool(cfg["sesiones"]["incluir_sin_calibrar"]))
        sin_identificar = cfg["usuarios"]["por_defecto"]

        def persona(usuario_id: str) -> str | None:
            """Nombre visible de la persona; None si quedó sin identificar."""
            return None if usuario_id == sin_identificar else usuario_id

        nombres = [r.archivo for r in resultado.parseos]
        clave = almacen.guardar({
            "mediciones": med,
            "nombre": _nombre_seguro(Path(nombres[0]).stem if len(nombres) == 1 else f"{len(nombres)}_archivos"),
        })
        return jsonify(
            api=VERSION_API,
            id=clave,
            rechazados=rechazados,
            estructura=estructura.a_dict(),
            n_analizables=int(analizables.sum()),
            n_discrepancias_estado=int(discrepancias(med).sum()),
            sesiones=[{
                "sesion_id": s["sesion_id"],
                "archivo": s["archivo"],
                "persona": persona(s["usuario_id"]),
                "dispositivo": s["dispositivo"],
                "inicio": s["inicio"].strftime("%H:%M:%S"),
                "fin": s["fin"].strftime("%H:%M:%S"),
                "duracion_s": round(float(s["duracion_s"]), 1),
                "n_muestras": int(s["n_muestras"]),
                "n_precalibracion": int(s["n_precalibracion"]),
                "evento_cierre": s["evento_cierre"],
                "sin_calibrar": bool(s["sin_calibrar"]),
                "cierre_implicito": bool(s["cierre_implicito"]),
            } for _, s in ses.iterrows()],
            personas=[{
                "usuario_id": u,
                "persona": persona(u),
                "n_sesiones": int((ses["usuario_id"] == u).sum()),
                "n_analizables": int((analizables & (med["usuario_id"] == u)).sum()),
            } for u in sorted(ses["usuario_id"].astype(str).unique())],
            por_archivo=[{
                "archivo": r.archivo,
                "bytes": r.n_bytes,
                "lineas": r.n_lineas,
                "mediciones": len(r.de_tipo(p.MEDICION)),
                "sesiones": int((ses["archivo"] == r.archivo).sum()),
                "persona": persona(resolver_usuario(r.archivo, cfg)),
            } for r in resultado.parseos],
            formatos=formatos,
            graficable=bool(len(med)),
        )

    def _analisis(clave: str) -> dict:
        datos = almacen.obtener(clave)
        if datos is None:
            abort(404, description="Análisis no encontrado o expirado; vuelva a subir el archivo.")
        return datos

    @app.get("/api/figura/<clave>.<formato>")
    def figura(clave: str, formato: str):
        if formato not in formatos:
            abort(404, description=f"Formato no disponible: {formato}")
        datos = _analisis(clave)
        args = request.args
        metodo = args.get("metodo") or None
        if metodo not in (None, "kde", "hex", "auto"):
            abort(400, description="metodo debe ser kde, hex o auto")
        cfg = config
        if args.get("incluir_sin_calibrar") in ("1", "true"):
            cfg = fusionar(config, {"sesiones": {"incluir_sin_calibrar": True}})
        sesion = args.get("sesion") or None
        usuario = args.get("usuario") or None
        with _BLOQUEO_FIGURAS:
            r = figura_densidad(datos["mediciones"], cfg,
                                usuario=usuario, sesion=sesion, metodo=metodo)
            try:
                contenido = exportar(r.figura, formato, cfg)
            finally:
                plt.close(r.figura)
        respuesta = Response(contenido, mimetype=TIPOS_MIME[formato])
        respuesta.headers["Cache-Control"] = "no-store"
        if args.get("descargar") == "1":
            sufijo = _nombre_seguro(sesion or usuario or "completo")
            nombre = f"densidad_{datos['nombre']}_{sufijo}_{r.metodo}.{formato}"
            respuesta.headers["Content-Disposition"] = f'attachment; filename="{nombre}"'
        return respuesta

    @app.get("/api/datos/<clave>.csv")
    def datos_csv(clave: str):
        datos = _analisis(clave)
        csv = datos["mediciones"].to_csv(index=False, date_format="%H:%M:%S.%f")
        return Response(
            csv.encode("utf-8-sig"),  # BOM para que Excel lea bien los acentos
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="mediciones_{datos["nombre"]}.csv"'},
        )

    return app


def ejecutar(config: dict, host: str | None = None, puerto: int | None = None) -> None:
    cw = config["web"]
    host = host or cw["host"]
    puerto = int(puerto or cw["puerto"])
    print(f"Abra en el navegador: http://{host}:{puerto}   (Ctrl+C para detener)")
    crear_app(config).run(host=host, port=puerto, debug=False, threaded=True)
