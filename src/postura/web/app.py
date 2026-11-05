"""Servidor Flask.

Rutas:
    GET  /                          página
    POST /api/analizar              sube uno o varios .txt → validación + sesiones
    POST /api/figura.<formato>      figura (png, svg, pdf, jpg); descargar=1 la baja como archivo
    POST /api/datos.csv             mediciones normalizadas

Las tres rutas POST reciben los mismos campos: ``archivos`` y ``personas`` (una por
archivo). La de figura acepta además los filtros sesion, usuario, metodo
(kde|hex|auto) e incluir_sin_calibrar (1/0).

El servidor no guarda nada entre peticiones: la página reenvía los archivos con cada
figura o descarga y se vuelven a analizar. Así funciona igual en un servidor sin
memoria compartida entre peticiones, como Vercel.
"""

from __future__ import annotations

import re
import threading
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
VERSION_API = 5
_BLOQUEO_FIGURAS = threading.Lock()   # matplotlib no es seguro entre hilos


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
    formatos = [f for f in cw["formatos_descarga"] if f in TIPOS_MIME]

    @app.get("/")
    def pagina():
        return send_from_directory(ESTATICOS, "index.html")

    @app.errorhandler(413)
    def demasiado_grande(_):
        return jsonify(error=f"La subida supera el máximo de {cw['max_mb_subida']} MB "
                             "(sumando todos los archivos)."), 413

    @app.errorhandler(400)
    @app.errorhandler(404)
    def error_api(e):
        return jsonify(error=e.description), e.code

    def _ingerir_subida():
        """Lee archivos y personas del formulario y los ingiere. → (resultado, cfg)."""
        subidos = [f for f in request.files.getlist("archivos") if f and f.filename]
        if not subidos:
            abort(400, description="No se recibió ningún archivo.")
        contenidos = [(Path(f.filename).name, f.read()) for f in subidos]
        # Persona opcional por archivo (mismo orden que los archivos). Vacía → se resuelve
        # con el patrón del nombre de archivo o queda sin identificar.
        personas = request.form.getlist("personas")
        mapa = {nombre: persona for (nombre, _), persona
                in zip(contenidos, map(_persona_segura, personas)) if persona}
        cfg = fusionar(config, {"usuarios": {"mapa_archivos": mapa}}) if mapa else config
        return ingerir_contenidos(contenidos, cfg), cfg

    def _mediciones_subidas():
        """Figura y CSV: mediciones de los archivos reenviados y nombre base de la descarga."""
        try:
            resultado, _ = _ingerir_subida()
        except ErrorEsquema as error:
            abort(400, description=f"No se pudo normalizar el registro: {error}")
        if not resultado.parseos:
            abort(400, description="Ninguno de los archivos es un registro válido del corrector.")
        nombres = [r.archivo for r in resultado.parseos]
        nombre = _nombre_seguro(Path(nombres[0]).stem if len(nombres) == 1 else f"{len(nombres)}_archivos")
        return resultado.mediciones, nombre

    @app.post("/api/analizar")
    def analizar():
        try:
            resultado, cfg = _ingerir_subida()
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

        return jsonify(
            api=VERSION_API,
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

    @app.post("/api/figura.<formato>")
    def figura(formato: str):
        if formato not in formatos:
            abort(404, description=f"Formato no disponible: {formato}")
        args = request.form
        metodo = args.get("metodo") or None
        if metodo not in (None, "kde", "hex", "auto"):
            abort(400, description="metodo debe ser kde, hex o auto")
        cfg = config
        if args.get("incluir_sin_calibrar") in ("1", "true"):
            cfg = fusionar(config, {"sesiones": {"incluir_sin_calibrar": True}})
        sesion = args.get("sesion") or None
        usuario = args.get("usuario") or None
        mediciones, nombre_base = _mediciones_subidas()
        with _BLOQUEO_FIGURAS:
            r = figura_densidad(mediciones, cfg,
                                usuario=usuario, sesion=sesion, metodo=metodo)
            try:
                contenido = exportar(r.figura, formato, cfg)
            finally:
                plt.close(r.figura)
        respuesta = Response(contenido, mimetype=TIPOS_MIME[formato])
        respuesta.headers["Cache-Control"] = "no-store"
        if args.get("descargar") == "1":
            sufijo = _nombre_seguro(sesion or usuario or "completo")
            nombre = f"densidad_{nombre_base}_{sufijo}_{r.metodo}.{formato}"
            respuesta.headers["Content-Disposition"] = f'attachment; filename="{nombre}"'
        return respuesta

    @app.post("/api/datos.csv")
    def datos_csv():
        mediciones, nombre_base = _mediciones_subidas()
        csv = mediciones.to_csv(index=False, date_format="%H:%M:%S.%f")
        return Response(
            csv.encode("utf-8-sig"),  # BOM para que Excel lea bien los acentos
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="mediciones_{nombre_base}.csv"'},
        )

    return app


def ejecutar(config: dict, host: str | None = None, puerto: int | None = None) -> None:
    cw = config["web"]
    host = host or cw["host"]
    puerto = int(puerto or cw["puerto"])
    print(f"Abra en el navegador: http://{host}:{puerto}   (Ctrl+C para detener)")
    crear_app(config).run(host=host, port=puerto, debug=False, threaded=True)
