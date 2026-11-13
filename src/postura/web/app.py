"""Servidor Flask.

Rutas:
    GET  /                          página
    POST /api/analizar              sube uno o varios .txt → validación + sesiones
    POST /api/figura.<formato>      figura (png, svg, pdf, jpg); descargar=1 la baja como archivo
    POST /api/datos.csv             mediciones normalizadas
    POST /api/analizar/estadistica  análisis descriptivo de F y L, BIEN vs MAL (JSON)
    POST /api/analizar/estadistica.<formato>   figura descriptiva (png, svg, pdf; campo figura:
                                    histogramas, cajas, violin o series)
    POST /api/analizar/estadistica/reporte.pdf  reporte completo de varias páginas
    POST /api/analizar/estadistica.csv          tabla de estadísticos descriptivos
    POST /api/diagnostica           indicadores descriptivos y narración de datos (JSON)
    POST /api/diagnostica/figura.<formato>     figura (campo figura: narrativa o indicadores)
    POST /api/diagnostica/informe.pdf          narración, figuras e indicadores en PDF
    POST /api/diagnostica/indicadores.csv      indicadores por sesión
    POST /api/confiabilidad/resumen            todo el análisis de confiabilidad (campo modo: dispositivo
                                               o postura)
    POST /api/confiabilidad/metricas           MTBF, MTTR, disponibilidad, tasa de fallos y lista de fallos
    POST /api/confiabilidad/weibull            ajuste de Weibull y proyección
    POST /api/confiabilidad/anomalias          ventanas con puntaje y severidad (Isolation Forest)
    POST /api/confiabilidad/prediccion         riesgo de fallo inminente (regresión logística)
    POST /api/confiabilidad/alertas            alertas ordenadas por severidad
    POST /api/confiabilidad/figura.<formato>   figura (fallos, weibull, anomalias o prediccion)
    POST /api/confiabilidad/informe.pdf        informe
    POST /api/confiabilidad/fallos.csv         lista de fallos
    POST /api/densidad3d            superficie 3D del mapa de densidad (JSON, vista interactiva)
    POST /api/figura3d.<formato>    la misma superficie como imagen fija
    POST /api/ergonomia.<formato>   figura ergonómica ISO 11226 (campo figura: zonas,
                                    momentos, sujetos o histogramas)
    POST /api/ergonomia/metricas    métricas |σ|, |R|, σG, RG, % por zona y ANOVA (JSON)
    POST /api/ergonomia/metricas.csv  la tabla de métricas en CSV

Las tres rutas POST reciben los mismos campos: ``archivos`` y ``personas`` (una por
archivo). La de figura acepta además los filtros sesion, usuario, metodo
(kde|hex|auto) e incluir_sin_calibrar (1/0).

El servidor no guarda nada entre peticiones: la página reenvía los archivos con cada
figura o descarga y se vuelven a analizar. Así funciona igual en un servidor sin
memoria compartida entre peticiones, como Vercel.
"""

from __future__ import annotations

import csv
import io
import re
import threading
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from .. import ergonomia as erg
from .. import estadistica as est
from .. import confiabilidad as conf
from .. import diagnostica as diag
from .. import indicadores as ind
from .. import parser as p
from ..configuracion import cargar_configuracion, fusionar
from ..figuras.densidad import TIPOS_MIME, _alcance, exportar, figura_densidad, seleccion
from ..figuras.densidad3d import figura_densidad_3d, malla_3d
from ..figuras.ergonomia import FIGURAS, figura_ergonomia
from ..figuras.estadistica import FIGURAS as FIGURAS_EST
from ..figuras.estadistica import figura_estadistica, reporte_pdf
from ..figuras.confiabilidad import FIGURAS as FIGURAS_CONF
from ..figuras.confiabilidad import figura_confiabilidad
from ..figuras.confiabilidad import informe_pdf as informe_confiabilidad
from ..figuras.diagnostica import FIGURAS as FIGURAS_DIAG
from ..figuras.diagnostica import figura_diagnostica, informe_pdf
from ..ingesta import ingerir_contenidos
from ..modelo import ErrorEsquema, discrepancias, seleccion_analizable
from ..sesiones import resolver_usuario
from ..validacion import validar_estructura

ESTATICOS = Path(__file__).parent / "estatico"
# Súbala cuando cambie la forma de la respuesta de /api/analizar; la página la compara
# para avisar si quedó abierto un servidor viejo.
VERSION_API = 11
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

    def _filtros():
        """(config con incluir_sin_calibrar aplicado, sesión, persona) desde el formulario."""
        args = request.form
        cfg = config
        if args.get("incluir_sin_calibrar") in ("1", "true"):
            cfg = fusionar(config, {"sesiones": {"incluir_sin_calibrar": True}})
        return cfg, args.get("sesion") or None, args.get("usuario") or None

    def _imagen(figura, formato: str, cfg: dict, nombre: str) -> Response:
        try:
            contenido = exportar(figura, formato, cfg)
        finally:
            plt.close(figura)
        respuesta = Response(contenido, mimetype=TIPOS_MIME[formato])
        respuesta.headers["Cache-Control"] = "no-store"
        if request.form.get("descargar") == "1":
            respuesta.headers["Content-Disposition"] = f'attachment; filename="{nombre}.{formato}"'
        return respuesta

    @app.post("/api/figura.<formato>")
    def figura(formato: str):
        if formato not in formatos:
            abort(404, description=f"Formato no disponible: {formato}")
        metodo = request.form.get("metodo") or None
        if metodo not in (None, "kde", "hex", "auto"):
            abort(400, description="metodo debe ser kde, hex o auto")
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        with _BLOQUEO_FIGURAS:
            r = figura_densidad(mediciones, cfg, usuario=usuario, sesion=sesion, metodo=metodo)
            sufijo = _nombre_seguro(sesion or usuario or "completo")
            return _imagen(r.figura, formato, cfg, f"densidad_{nombre_base}_{sufijo}_{r.metodo}")

    formatos_est = [f for f in config["estadistica"]["formatos_descarga"] if f in TIPOS_MIME]

    @app.post("/api/analizar/estadistica")
    def estadistica():
        cfg, sesion, usuario = _filtros()
        mediciones, _ = _mediciones_subidas()
        datos, _, _ = seleccion(mediciones, cfg, [usuario] if usuario else None, [sesion] if sesion else None)
        return jsonify({**est.analisis(datos, cfg), "formatos": formatos_est})

    @app.post("/api/analizar/estadistica.csv")
    def estadistica_csv():
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        datos, _, _ = seleccion(mediciones, cfg, [usuario] if usuario else None, [sesion] if sesion else None)
        filas = est.descriptivos(datos, cfg)
        claves_p = list(filas[0]["percentiles"]) if filas else []
        salida = io.StringIO()
        w = csv.writer(salida)
        w.writerow(["angulo", "grupo", "n", "media", "sd", "min", *[f"p{k}" for k in claves_p], "max", "iqr",
                    "asimetria", "curtosis"])
        for f in filas:
            w.writerow([f["angulo"], f["grupo"], f["n"], f["media"], f["sd"], f["min"],
                        *[f["percentiles"][k] for k in claves_p], f["max"], f["iqr"], f["asimetria"], f["curtosis"]])
        sufijo = _nombre_seguro(sesion or usuario or "completo")
        return Response(salida.getvalue().encode("utf-8-sig"), mimetype="text/csv", headers={
            "Content-Disposition": f'attachment; filename="estadistica_{nombre_base}_{sufijo}.csv"'})

    @app.post("/api/analizar/estadistica/reporte.pdf")
    def estadistica_reporte():
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        with _BLOQUEO_FIGURAS:
            contenido = reporte_pdf(mediciones, cfg, usuario=usuario, sesion=sesion)
        sufijo = _nombre_seguro(sesion or usuario or "completo")
        return Response(contenido, mimetype=TIPOS_MIME["pdf"], headers={
            "Content-Disposition": f'attachment; filename="reporte_estadistico_{nombre_base}_{sufijo}.pdf"',
            "Cache-Control": "no-store"})

    @app.post("/api/analizar/estadistica.<formato>")
    def estadistica_figura(formato: str):
        if formato not in formatos_est:
            abort(404, description=f"Formato no disponible: {formato} (use {', '.join(formatos_est)})")
        tipo = request.form.get("figura") or "histogramas"
        if tipo not in FIGURAS_EST:
            abort(400, description=f"figura debe ser {', '.join(FIGURAS_EST)}")
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        with _BLOQUEO_FIGURAS:
            r = figura_estadistica(mediciones, cfg, tipo, usuario=usuario, sesion=sesion)
            sufijo = _nombre_seguro(sesion or usuario or "completo")
            return _imagen(r.figura, formato, cfg, f"estadistica_{tipo}_{nombre_base}_{sufijo}")

    def _diagnostica():
        """(datos analizables del filtro, config, análisis, alcance, nombre base de descarga)."""
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        usuarios, sesiones = ([usuario] if usuario else None), ([sesion] if sesion else None)
        datos, _, _ = seleccion(mediciones, cfg, usuarios, sesiones)
        alcance = _alcance(usuarios, sesiones)
        sufijo = _nombre_seguro(sesion or usuario or "completo")
        return datos, cfg, diag.analisis(datos, cfg, alcance), alcance, f"{nombre_base}_{sufijo}"

    @app.post("/api/diagnostica")
    def diagnostica():
        _, _, a, _, _ = _diagnostica()
        return jsonify({**a, "formatos": formatos_est})

    @app.post("/api/diagnostica/figura.<formato>")
    def diagnostica_figura(formato: str):
        if formato not in formatos_est:
            abort(404, description=f"Formato no disponible: {formato} (use {', '.join(formatos_est)})")
        tipo = request.form.get("figura") or "narrativa"
        if tipo not in FIGURAS_DIAG:
            abort(400, description=f"figura debe ser {', '.join(FIGURAS_DIAG)}")
        datos, cfg, a, alcance, nombre = _diagnostica()
        with _BLOQUEO_FIGURAS:
            figura = figura_diagnostica(datos, cfg, a, tipo, alcance, con_titulo=request.form.get("titulo") != "0")
            return _imagen(figura, formato, cfg, f"diagnostica_{tipo}_{nombre}")

    @app.post("/api/diagnostica/informe.pdf")
    def diagnostica_informe():
        datos, cfg, a, alcance, nombre = _diagnostica()
        with _BLOQUEO_FIGURAS:
            contenido = informe_pdf(datos, cfg, a, alcance)
        return Response(contenido, mimetype=TIPOS_MIME["pdf"], headers={
            "Content-Disposition": f'attachment; filename="analitica_diagnostica_{nombre}.pdf"', "Cache-Control": "no-store"})

    @app.post("/api/diagnostica/indicadores.csv")
    def diagnostica_csv():
        _, _, a, _, nombre = _diagnostica()
        claves = [i["clave"] for i in ind.INDICADORES]
        salida = io.StringIO()
        w = csv.writer(salida)
        w.writerow(["sesion_id", "usuario_id", "n", *claves])
        w.writerow(["(selección completa)", "", a["total"]["n"], *[a["total"][k] for k in claves]])
        for fila in a["por_sesion"]:
            w.writerow([fila["sesion_id"], fila["usuario_id"], fila["n"], *[fila[k] for k in claves]])
        return Response(salida.getvalue().encode("utf-8-sig"), mimetype="text/csv", headers={
            "Content-Disposition": f'attachment; filename="indicadores_{nombre}.csv"'})

    def _confiabilidad():
        """(análisis, config, alcance, nombre base) con los archivos, filtros y modo del formulario."""
        modo = request.form.get("modo") or "dispositivo"
        if modo not in conf.MODOS:
            abort(400, description=f"modo debe ser {' o '.join(conf.MODOS)}")
        try:
            resultado, cfg = _ingerir_subida()
        except ErrorEsquema as error:
            abort(400, description=f"No se pudo normalizar el registro: {error}")
        if not resultado.parseos:
            abort(400, description="Ninguno de los archivos es un registro válido del corrector.")
        if request.form.get("incluir_sin_calibrar") in ("1", "true"):
            cfg = fusionar(cfg, {"sesiones": {"incluir_sin_calibrar": True}})
        sesion, usuario = request.form.get("sesion") or None, request.form.get("usuario") or None
        nombres = [r.archivo for r in resultado.parseos]
        nombre = _nombre_seguro(Path(nombres[0]).stem if len(nombres) == 1 else f"{len(nombres)}_archivos")
        alcance = _alcance([usuario] if usuario else None, [sesion] if sesion else None)
        sufijo = _nombre_seguro(sesion or usuario or "completo")
        return conf.analisis(resultado, cfg, modo, usuario, sesion), cfg, alcance, f"{modo}_{nombre}_{sufijo}"

    def _publico(a: dict) -> dict:
        return {k: v for k, v in a.items() if not k.startswith("_")}

    @app.post("/api/confiabilidad/resumen")
    def confiabilidad_resumen():
        a, _, alcance, _ = _confiabilidad()
        return jsonify({**_publico(a), "alcance": alcance, "formatos": formatos_est})

    @app.post("/api/confiabilidad/<parte>")
    def confiabilidad_parte(parte: str):
        partes = {"metricas": ("metricas", "por_sesion", "fallos"), "weibull": ("weibull",),
                  "anomalias": ("anomalias",), "prediccion": ("prediccion",), "alertas": ("alertas",)}
        if parte not in partes:
            abort(404, description=f"Parte desconocida: {parte} (use resumen, {', '.join(partes)})")
        a, _, alcance, _ = _confiabilidad()
        return jsonify({"modo": a["modo"], "alcance": alcance, **{k: a[k] for k in partes[parte]}})

    @app.post("/api/confiabilidad/figura.<formato>")
    def confiabilidad_figura(formato: str):
        if formato not in formatos_est:
            abort(404, description=f"Formato no disponible: {formato} (use {', '.join(formatos_est)})")
        tipo = request.form.get("figura") or "fallos"
        if tipo not in FIGURAS_CONF:
            abort(400, description=f"figura debe ser {', '.join(FIGURAS_CONF)}")
        a, cfg, _, nombre = _confiabilidad()
        with _BLOQUEO_FIGURAS:
            return _imagen(figura_confiabilidad(a, tipo, cfg), formato, cfg, f"confiabilidad_{tipo}_{nombre}")

    @app.post("/api/confiabilidad/informe.pdf")
    def confiabilidad_informe():
        a, cfg, alcance, nombre = _confiabilidad()
        with _BLOQUEO_FIGURAS:
            contenido = informe_confiabilidad(a, cfg, alcance)
        return Response(contenido, mimetype=TIPOS_MIME["pdf"], headers={
            "Content-Disposition": f'attachment; filename="confiabilidad_{nombre}.pdf"', "Cache-Control": "no-store"})

    @app.post("/api/confiabilidad/fallos.csv")
    def confiabilidad_csv():
        a, _, _, nombre = _confiabilidad()
        columnas = ["archivo", "sesion_id", "tipo", "reloj_min", "reparacion_s", "linea", "detalle"]
        salida = io.StringIO()
        w = csv.writer(salida)
        w.writerow(columnas)
        for f in a["fallos"]:
            w.writerow([f[c] for c in columnas])
        return Response(salida.getvalue().encode("utf-8-sig"), mimetype="text/csv", headers={
            "Content-Disposition": f'attachment; filename="fallos_{nombre}.csv"'})

    @app.post("/api/densidad3d")
    def densidad_3d():
        cfg, sesion, usuario = _filtros()
        mediciones, _ = _mediciones_subidas()
        return jsonify(malla_3d(mediciones, cfg, usuario=usuario, sesion=sesion))

    @app.post("/api/figura3d.<formato>")
    def figura_3d(formato: str):
        if formato not in formatos:
            abort(404, description=f"Formato no disponible: {formato}")
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        with _BLOQUEO_FIGURAS:
            fig, _ = figura_densidad_3d(mediciones, cfg, usuario=usuario, sesion=sesion)
            sufijo = _nombre_seguro(sesion or usuario or "completo")
            return _imagen(fig, formato, cfg, f"densidad3d_{nombre_base}_{sufijo}")

    @app.post("/api/ergonomia.<formato>")
    def figura_ergo(formato: str):
        if formato not in formatos:
            abort(404, description=f"Formato no disponible: {formato}")
        tipo = request.form.get("figura") or "zonas"
        if tipo not in FIGURAS:
            abort(400, description=f"figura debe ser {', '.join(FIGURAS)}")
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        with _BLOQUEO_FIGURAS:
            r = figura_ergonomia(mediciones, cfg, tipo, usuario=usuario, sesion=sesion)
            sufijo = "todos" if tipo == "sujetos" else _nombre_seguro(sesion or usuario or "completo")
            return _imagen(r.figura, formato, cfg, f"ergonomia_{tipo}_{nombre_base}_{sufijo}")

    def _metricas_ergo():
        cfg, sesion, usuario = _filtros()
        mediciones, nombre_base = _mediciones_subidas()
        datos, global_ = erg.seleccionar(mediciones, cfg, usuario, sesion)
        sel = erg.metricas(datos, cfg)
        g = erg.metricas(global_, cfg)
        return {
            "seleccion": {**sel.a_dict(),
                          "sigma_g": erg._cociente(sel.dispersion, g.dispersion),
                          "r_g": erg._cociente(sel.desplazamiento, g.desplazamiento)},
            "filas": erg.tabla_metricas(global_, cfg, cfg["usuarios"]["por_defecto"]),
            "momentos": erg.comparar_momentos(global_, datos, cfg),
            "nombres_zona": cfg["ergonomia"]["nombres_zona"],
            "fuente": cfg["ergonomia"]["fuente"],
        }, nombre_base

    @app.post("/api/ergonomia/metricas")
    def metricas_ergo():
        return jsonify(_metricas_ergo()[0])

    @app.post("/api/ergonomia/metricas.csv")
    def metricas_ergo_csv():
        datos, nombre_base = _metricas_ergo()
        salida = io.StringIO()
        columnas = ["tipo", "etiqueta", "n", "media_frontal", "media_lateral", "sd_frontal", "sd_lateral",
                    "dispersion", "desplazamiento", "sigma_g", "r_g"]
        w = csv.writer(salida)
        w.writerow(columnas + [f"pct_{z}" for z in erg.NIVELES])
        for f in datos["filas"]:
            w.writerow([f[c] for c in columnas] + [f["pct_zona"][z] for z in erg.NIVELES])
        return Response(
            salida.getvalue().encode("utf-8-sig"), mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="metricas_ergonomia_{nombre_base}.csv"'},
        )

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
