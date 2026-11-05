"""Interfaz de línea de comandos.

    postura ingerir datos/*.txt --salida salidas/mediciones.parquet
    postura calidad salidas/mediciones.parquet
    postura figura densidad salidas/mediciones.parquet --usuario U01 --salida salidas/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import calidad, ingesta
from .configuracion import ErrorConfiguracion, cargar_configuracion
from .modelo import ErrorEsquema, discrepancias


def _utf8() -> None:
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _cmd_ingerir(args, config) -> int:
    rutas = ingesta.expandir_rutas(args.archivos)
    if not rutas:
        print("Error: ningún archivo coincide con los patrones indicados.", file=sys.stderr)
        return 1
    salida = Path(args.salida or Path(config["rutas"]["salidas"]) / "mediciones.parquet")
    salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_log = salida.parent / "ingesta.log"
    manejador = logging.FileHandler(ruta_log, mode="w", encoding="utf-8")
    manejador.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    registro = logging.getLogger("postura.ingesta")
    registro.setLevel(logging.INFO)
    registro.addHandler(manejador)
    try:
        resultado = ingesta.ingerir_archivos(rutas, config)
    finally:
        registro.removeHandler(manejador)
        manejador.close()
    for r in resultado.rechazados:
        print(f"RECHAZADO {r.archivo}: {r.motivo}", file=sys.stderr)
    if not resultado.parseos:
        print(f"Error: ningún archivo es un registro válido del corrector. Detalle en {ruta_log}",
              file=sys.stderr)
        return 1
    rm, rs = ingesta.guardar(resultado, salida)
    c = resultado.contadores
    med = resultado.mediciones
    n_disc = int(discrepancias(med).sum())
    print(f"Archivos: {len(resultado.parseos)} procesados, {len(resultado.rechazados)} rechazados  "
          f"líneas: {c.n_lineas}  mediciones: {c.n_mediciones}  sesiones: {len(resultado.sesiones)}")
    print(f"Fuera de sesión: {c.n_mediciones_fuera_sesion}  eventos desconocidos: {c.n_desconocidos}  "
          f"líneas inválidas: {c.n_invalidas}  conexiones fallidas: {c.n_conexiones_fallidas}")
    print(f"Discrepancias estado reportado vs derivado: {n_disc} de {len(med)}")
    if n_disc:
        print("AVISO: hay discrepancias de estado; posible firmware distinto o dato corrupto. "
              "Revise `postura calidad`.", file=sys.stderr)
    print(f"Mediciones → {rm}\nSesiones   → {rs}\nLog        → {ruta_log}")
    return 0


def _cmd_calidad(args, config) -> int:
    med, ses = ingesta.cargar(args.mediciones)
    informe = calidad.evaluar(med, ses, config)
    print(calidad.formatear(informe, config))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(informe.a_json(), encoding="utf-8")
        print(f"\nInforme JSON → {args.json}")
    if informe.n_discrepancias_estado:
        print(f"AVISO: {informe.n_discrepancias_estado} discrepancias de estado.", file=sys.stderr)
        return 3 if args.estricto else 0
    return 0


def _cmd_figura_densidad(args, config) -> int:
    from .figuras.densidad import figura_densidad

    med, _ = ingesta.cargar(args.mediciones)
    if args.usuario and not med["usuario_id"].isin(args.usuario).any():
        print(f"AVISO: no hay mediciones del usuario {', '.join(args.usuario)}. "
              f"Usuarios presentes: {', '.join(sorted(med['usuario_id'].unique()))}", file=sys.stderr)
    if args.sesion and not med["sesion_id"].isin(args.sesion).any():
        print(f"AVISO: no hay mediciones de la sesión {', '.join(args.sesion)}.", file=sys.stderr)
    salida = Path(args.salida or config["rutas"]["salidas"])
    r = figura_densidad(med, config, usuario=args.usuario, sesion=args.sesion,
                        metodo=args.metodo, salida=salida, nombre=args.nombre)
    print(r.titulo)
    print(r.pie)
    for ruta in r.rutas:
        print(f"→ {ruta}")
    return 0


def _cmd_web(args, config) -> int:
    from .web.app import ejecutar

    if not args.no_abrir:
        import threading
        import webbrowser

        host = args.host or config["web"]["host"]
        puerto = args.puerto or config["web"]["puerto"]
        threading.Timer(1.0, webbrowser.open, args=(f"http://{host}:{puerto}",)).start()
    ejecutar(config, host=args.host, puerto=args.puerto)
    return 0


def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="postura", description="Analítica de registros del corrector de postura.")
    ap.add_argument("--config", help="YAML que se fusiona sobre config/default.yaml")
    sub = ap.add_subparsers(dest="comando", required=True)

    pi = sub.add_parser("ingerir", help="Texto crudo → parquet de mediciones y sesiones")
    pi.add_argument("archivos", nargs="+", help="Archivos o patrones (admite comodines)")
    pi.add_argument("--salida", help="Ruta del parquet de mediciones (por defecto salidas/mediciones.parquet)")
    pi.set_defaults(funcion=_cmd_ingerir)

    pc = sub.add_parser("calidad", help="Informe de calidad sobre un parquet de mediciones")
    pc.add_argument("mediciones")
    pc.add_argument("--json", help="Guarda además el informe en JSON")
    pc.add_argument("--estricto", action="store_true", help="Código de salida 3 si hay discrepancias de estado")
    pc.set_defaults(funcion=_cmd_calidad)

    pf = sub.add_parser("figura", help="Genera figuras")
    subf = pf.add_subparsers(dest="figura", required=True)
    pd_ = subf.add_parser("densidad", help="Mapa de densidad conjunta con histogramas marginales")
    pd_.add_argument("mediciones")
    pd_.add_argument("--usuario", action="append", help="Filtra por usuario (repetible)")
    pd_.add_argument("--sesion", action="append", help="Filtra por sesion_id (repetible)")
    pd_.add_argument("--metodo", choices=["kde", "hex", "auto"], help="Sobrescribe figura.metodo")
    pd_.add_argument("--salida", help="Directorio de salida")
    pd_.add_argument("--nombre", help="Nombre base de los archivos generados")
    pd_.add_argument("--incluir-sin-calibrar", action="store_true",
                     help="Incluye sesiones sin CALIBRADO (por defecto se excluyen)")
    pd_.set_defaults(funcion=_cmd_figura_densidad)

    pw = sub.add_parser("web", help="Abre la página web para subir registros y ver la figura")
    pw.add_argument("--host", help="Por defecto web.host de la configuración (127.0.0.1)")
    pw.add_argument("--puerto", type=int, help="Por defecto web.puerto de la configuración (8000)")
    pw.add_argument("--no-abrir", action="store_true", help="No abrir el navegador automáticamente")
    pw.set_defaults(funcion=_cmd_web)
    return ap


def principal(argv: list[str] | None = None) -> int:
    _utf8()
    args = construir_parser().parse_args(argv)
    try:
        extra = {}
        if getattr(args, "incluir_sin_calibrar", False):
            extra["sesiones"] = {"incluir_sin_calibrar": True}
        config = cargar_configuracion(args.config, **extra)
        return args.funcion(args, config)
    except (ErrorConfiguracion, ErrorEsquema, FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(principal())
