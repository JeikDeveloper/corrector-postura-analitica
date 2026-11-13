"""Analítica diagnóstica: indicadores descriptivos y narración de datos.

La narración convierte los indicadores en un relato corto en lenguaje sencillo:
titular, contexto, hallazgos, qué significa, qué revisar y limitaciones. Es
determinista y basada en reglas: cada frase sale de un número calculado y los
umbrales que eligen las palabras ("frecuentes", "breves"...) están en
``diagnostica.narracion`` de la configuración. Si los datos no alcanzan para
afirmar algo, la narración lo dice en lugar de afirmarlo.
"""

from __future__ import annotations

from . import estadistica as est
from . import indicadores as ind

RECOMENDACION_CAUSA = {   # solo la acción: la causa ya está en «Otros hallazgos»
    "frontal_adelante": "Por la causa principal: revisar la altura y la distancia de la pantalla y del teclado.",
    "frontal_atras": "Por la causa principal: revisar el respaldo de la silla y la distancia a la pantalla.",
    "lateral_derecha": "Por la causa principal: revisar el apoyo de los brazos y lo que queda a la derecha (mouse, objetos).",
    "lateral_izquierda": "Por la causa principal: revisar el apoyo de los brazos y lo que queda a la izquierda.",
}


FRASE_CAUSA = {
    "frontal_adelante": ("inclinarse hacia adelante", "F > {FRONTAL_MAX}°"),
    "frontal_atras": ("inclinarse hacia atrás", "F < {FRONTAL_MIN}°"),
    "lateral_derecha": ("inclinarse hacia la derecha", "L > {LATERAL_MAX}°"),
    "lateral_izquierda": ("inclinarse hacia la izquierda", "L < {LATERAL_MIN}°"),
}


def _n(x, d=1) -> str:
    """Número con coma decimal y sin ceros sobrantes."""
    if x is None:
        return "—"
    texto = f"{x:.{d}f}".rstrip("0").rstrip(".") if d else f"{x:.0f}"
    return texto.replace(".", ",")


def _mins(segundos: float) -> str:
    return f"{_n(segundos / 60)} min" if segundos >= 90 else f"{_n(segundos, 0)} s"


def narracion(datos, config: dict, alcance: str) -> dict:
    cn = config["diagnostica"]["narracion"]
    cad = float(config["calidad"]["cadencia_nominal_s"])
    total = ind.calcular(datos, config)
    n = total["n"]
    if not n:
        return {"titular": "No hay mediciones analizables con este filtro.", "contexto": f"Selección: {alcance}.",
                "lecturas": {}, "hallazgos": [], "significado": None, "recomendaciones": [], "limitaciones":
                ["No se narra nada sin datos: elige otro filtro o sube un registro con mediciones posteriores al CALIBRADO."],
                "destacada": None}
    ep = est.episodios(datos, config)["MAL"]
    causas = [c for c in est.causas_mal(datos, config) if c["clave"] != "sin_umbral" and c["n"]]
    causa = max(causas, key=lambda c: c["n"]) if causas else None
    sesiones = ind.por_sesion(datos, config)
    tc, ah, dur = total["tiempo_correcto"], total["alertas_hora"], total["duracion_mal"]
    segundos = n * cad
    personas = datos.loc[datos["usuario_id"] != config["usuarios"]["por_defecto"], "usuario_id"].nunique()

    # ---------------- titular: el mensaje principal, sin cifras (las cifras están en los indicadores)
    if tc >= cn["tiempo_correcto_alto"]:
        base = "La postura se mantuvo correcta la mayor parte del tiempo"
    elif tc >= cn["tiempo_correcto_medio"]:
        base = "La postura fue correcta en buena parte del tiempo, con margen de mejora"
    else:
        base = "La postura estuvo fuera de los umbrales una parte importante del tiempo"
    breves = dur is not None and dur <= cn["duracion_corta_s"]
    if not ep["n_episodios"]:
        titular = base + " y no hubo alertas."
    elif ah >= cn["alertas_hora_altas"]:
        titular = base + ", pero las alertas fueron frecuentes" + (", aunque breves." if breves else ".")
    else:
        titular = base + ("; las alertas fueron pocas y breves." if breves else "; las alertas fueron pocas.")

    # ---------------- contexto
    contexto = (f"Se analizaron {n:,} mediciones".replace(",", ".") + f" (unos {_mins(segundos)} de registro) de "
                f"{total['sesiones']} sesi{'ón' if total['sesiones'] == 1 else 'ones'}"
                + (f" de {personas} persona{'s' if personas != 1 else ''}" if personas else "")
                + f" · selección: {alcance}. Solo cuentan las mediciones posteriores al CALIBRADO de cada sesión; "
                  "BIEN y MAL son los que reportó el dispositivo.")

    # ---------------- lectura de cada indicador: qué significa su valor, sin repetirlo
    lecturas = {}
    if tc >= cn["tiempo_correcto_alto"]:
        lecturas["tiempo_correcto"] = "Alto: casi todo el tiempo dentro de los umbrales del dispositivo."
    elif tc >= cn["tiempo_correcto_medio"]:
        lecturas["tiempo_correcto"] = "Medio: hay margen de mejora."
    else:
        lecturas["tiempo_correcto"] = "Bajo: una parte importante del tiempo fuera de los umbrales."
    if ep["n_episodios"]:
        frecuentes = "Frecuentes" if ah >= cn["alertas_hora_altas"] else "Pocas"
        lecturas["alertas_hora"] = f"{frecuentes} (se consideran frecuentes desde {_n(cn['alertas_hora_altas'])} por hora)."
        lecturas["duracion_mal"] = ("Breves: la postura se corrige en segundos." if breves else
                                    "Largas: cuando se pierde la postura, tarda en recuperarse.")
    else:
        lecturas["alertas_hora"] = "Sin alertas en la selección."
        lecturas["duracion_mal"] = "No aplica: no hubo episodios MAL."
    if total["primera_alerta"] is None:
        lecturas["primera_alerta"] = "No aplica: ninguna sesión tuvo alertas."
    elif total["primera_alerta"] < cn["primera_alerta_temprana_min"]:
        lecturas["primera_alerta"] = "Temprana: llega al poco tiempo de calibrar."
    else:
        lecturas["primera_alerta"] = "La buena postura se sostiene un buen rato al empezar."
    s = total["estabilidad"]
    if s < cn["estabilidad_baja"]:
        lecturas["estabilidad"] = "Estable: la postura se mueve poco alrededor de su promedio."
    elif s > cn["estabilidad_alta"]:
        lecturas["estabilidad"] = "Variable: la postura cambia mucho."
    else:
        lecturas["estabilidad"] = "Moderadamente variable."
    if total["desplazamiento"] < cn["desplazamiento_alto"]:
        lecturas["desplazamiento"] = "La postura habitual está cerca del cero calibrado."
    else:
        lecturas["desplazamiento"] = "La postura habitual se aleja del cero calibrado."
    if total["zona_optima"] is not None and tc - total["zona_optima"] >= cn["margen_iso_pct"]:
        lecturas["zona_optima"] = (f"{_n(tc - total['zona_optima'])} puntos por debajo del tiempo en postura correcta: "
                                   "la norma es más estricta que los umbrales del dispositivo (por ejemplo, cuenta como "
                                   "extrema cualquier inclinación hacia atrás).")
    else:
        lecturas["zona_optima"] = "Coincide en lo esencial con el tiempo en postura correcta."
    mov = total["movilidad"]
    if mov is None:
        lecturas["movilidad"] = "No aplica."
    elif mov < cn["movilidad_baja"]:
        lecturas["movilidad"] = "Baja: postura estática por periodos largos."
    elif mov > cn["movilidad_alta"]:
        lecturas["movilidad"] = "Alta: movimientos bruscos frecuentes."
    else:
        lecturas["movilidad"] = "Normal: ni estática ni con movimientos bruscos."

    # ---------------- otros hallazgos: solo lo que no está en ningún indicador
    hallazgos = []
    if ep["n_episodios"]:
        hallazgos.append({"tema": "Episodios",
                          "texto": f"{ep['n_episodios']} episodio{'s' if ep['n_episodios'] != 1 else ''} MAL en total; "
                                   f"el más largo duró {_mins(ep['duracion_max_s'])}."})
    if causa:
        frase, regla = FRASE_CAUSA[causa["clave"]]
        hallazgos.append({"tema": "Causa principal",
                          "texto": f"La causa más común de MAL fue {frase} ({regla.format(**config['umbrales'])}): "
                                   f"explica el {_n(causa['pct'])} % de las muestras MAL."})
    destacada = None
    validas = [x for x in sesiones if x["tiempo_correcto"] is not None]
    if len(validas) >= 2:
        peor = min(validas, key=lambda x: x["tiempo_correcto"])
        mejor = max(validas, key=lambda x: x["tiempo_correcto"])
        if mejor["tiempo_correcto"] - peor["tiempo_correcto"] >= 1:
            destacada = peor["sesion_id"]
            hallazgos.append({"tema": "Diferencias entre sesiones",
                              "texto": f"La sesión con más tiempo fuera de postura fue {peor['sesion_id']}: "
                                       f"{_n(mejor['tiempo_correcto'] - peor['tiempo_correcto'])} puntos menos de tiempo "
                                       f"correcto que la mejor ({mejor['sesion_id']}). El gráfico muestra el valor de cada "
                                       "sesión."})

    # ---------------- qué significa
    if not ep["n_episodios"]:
        significado = "No hubo pérdidas de postura en el periodo analizado."
    elif tc >= cn["tiempo_correcto_alto"] and dur is not None and dur <= cn["duracion_corta_s"]:
        significado = ("La postura habitual es buena y los desvíos se corrigen en segundos: el reto no es sostener la "
                       "postura, sino evitar las recaídas puntuales.")
    elif tc >= cn["tiempo_correcto_alto"]:
        significado = "La postura habitual es buena, pero cuando se pierde tarda en recuperarse."
    elif tc >= cn["tiempo_correcto_medio"]:
        significado = "La postura alterna entre periodos correctos y desvíos sostenidos; hay espacio claro de mejora."
    else:
        significado = ("Una parte relevante del tiempo transcurre fuera de los umbrales: el problema es sostenido, no "
                       "puntual.")

    # ---------------- qué revisar (acciones, no diagnósticos médicos)
    recomendaciones = []
    if causa and causa["clave"] in RECOMENDACION_CAUSA:
        recomendaciones.append(RECOMENDACION_CAUSA[causa["clave"]])
    if total["primera_alerta"] is not None and total["primera_alerta"] < cn["primera_alerta_temprana_min"]:
        recomendaciones.append("Revisar cómo queda el puesto al sentarse y repetir la calibración en la postura de "
                               "trabajo real.")
    if ep["n_episodios"] and ah >= cn["alertas_hora_altas"]:
        recomendaciones.append("Ver si las alertas coinciden con tareas concretas (por ejemplo, buscar documentos o "
                               "usar el teléfono).")
    if destacada:
        recomendaciones.append(f"Revisar en detalle la sesión {destacada}.")

    # ---------------- limitaciones: cuánto confiar en el relato
    limitaciones = []
    if ep["n_episodios"] < cn["min_episodios_confiable"]:
        limitaciones.append(f"Hay pocos episodios MAL (se sugieren al menos {cn['min_episodios_confiable']}): los hallazgos "
                            "sobre alertas y causas son poco representativos.")
    if segundos / 60 < cn["min_minutos_confiable"]:
        limitaciones.append(f"Es poco tiempo de registro para generalizar a una jornada de trabajo (se sugieren al menos "
                            f"{cn['min_minutos_confiable']} min).")
    if not personas:
        limitaciones.append("Los archivos no tienen persona asignada: no se puede comparar entre personas.")
    if total["sesiones"] > 1:
        limitaciones.append("Cada sesión tiene su propio cero de calibración: los ángulos de sesiones distintas no son "
                            "comparables en términos absolutos.")
    return {"titular": titular, "contexto": contexto, "lecturas": lecturas, "hallazgos": hallazgos,
            "significado": significado, "recomendaciones": recomendaciones, "limitaciones": limitaciones,
            "destacada": destacada}


def analisis(datos, config: dict, alcance: str) -> dict:
    """Indicadores (con definición, fórmula e interpretación), por sesión, y narración."""
    total = ind.calcular(datos, config)
    nar = narracion(datos, config, alcance)
    return {
        "indicadores": [{**i, "valor": total[i["clave"]], "lectura": nar["lecturas"].get(i["clave"])}
                        for i in ind.INDICADORES],
        "por_sesion": ind.por_sesion(datos, config),
        "total": total,
        "narracion": nar,
    }
