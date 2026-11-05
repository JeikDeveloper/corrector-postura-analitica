"""Informe de calidad: cadencia, huecos, duplicados, orden temporal y discrepancias de estado.

Las discrepancias entre el estado reportado y el derivado desde los umbrales
nunca se silencian: si aparecen, son señal de firmware distinto o dato corrupto.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .modelo import discrepancias, seleccion_analizable


@dataclass
class CalidadSesion:
    sesion_id: str
    usuario_id: str
    dispositivo: str
    n_muestras: int
    n_analizables: int
    n_precalibracion: int
    cadencia_mediana_s: float | None
    cadencia_p95_s: float | None
    n_huecos: int
    hueco_max_s: float | None
    n_rafagas: int
    n_duplicados_tiempo: int
    n_retrocesos_tiempo: int
    n_discrepancias_estado: int
    banderas: list[str] = field(default_factory=list)


@dataclass
class InformeCalidad:
    n_mediciones: int
    n_sesiones: int
    n_usuarios: int
    n_archivos: int
    n_analizables: int
    n_precalibracion: int
    n_sin_calibrar_excluidas: int
    n_discrepancias_estado: int
    alertas: list[str]
    sesiones: list[CalidadSesion]

    def a_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)


def _redondear(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def evaluar(mediciones: pd.DataFrame, sesiones: pd.DataFrame, config: dict) -> InformeCalidad:
    cq = config["calidad"]
    umbral_hueco = float(cq["umbral_hueco_s"])
    umbral_rafaga = float(cq["umbral_rafaga_s"])
    incluir_sc = bool(config["sesiones"]["incluir_sin_calibrar"])

    analizable = seleccion_analizable(mediciones, incluir_sc)
    discrepa = discrepancias(mediciones)
    alertas: list[str] = []
    por_sesion: list[CalidadSesion] = []

    for _, s in sesiones.sort_values(["archivo", "indice"]).iterrows():
        sel = mediciones["sesion_id"] == s["sesion_id"]
        m = mediciones.loc[sel]
        difs = m["t"].diff().dt.total_seconds().dropna().to_numpy()
        huecos = difs[difs > umbral_hueco]
        banderas = [b for b in ("sin_calibrar", "cierre_implicito", "calibracion_multiple", "fecha_desconocida")
                    if bool(s[b])]
        if len(m) == 0:
            banderas.append("sin_muestras")
        q = CalidadSesion(
            sesion_id=s["sesion_id"],
            usuario_id=s["usuario_id"],
            dispositivo=s["dispositivo"],
            n_muestras=int(len(m)),
            n_analizables=int(analizable[sel].sum()),
            n_precalibracion=int(s["n_precalibracion"]),
            cadencia_mediana_s=_redondear(np.median(difs)) if difs.size else None,
            cadencia_p95_s=_redondear(np.percentile(difs, 95)) if difs.size else None,
            n_huecos=int(huecos.size),
            hueco_max_s=_redondear(huecos.max()) if huecos.size else None,
            n_rafagas=int(((difs >= 0) & (difs < umbral_rafaga)).sum()),
            n_duplicados_tiempo=int(m["t"].duplicated().sum()),
            n_retrocesos_tiempo=int((difs < 0).sum()),
            n_discrepancias_estado=int(discrepa[sel].sum()),
            banderas=banderas,
        )
        por_sesion.append(q)
        if q.n_discrepancias_estado:
            alertas.append(f"Sesión {q.sesion_id}: {q.n_discrepancias_estado} discrepancias entre estado "
                           "reportado y derivado (¿firmware distinto o dato corrupto?)")
        if q.n_retrocesos_tiempo:
            alertas.append(f"Sesión {q.sesion_id}: {q.n_retrocesos_tiempo} marcas de tiempo fuera de orden")
        if q.n_duplicados_tiempo:
            alertas.append(f"Sesión {q.sesion_id}: {q.n_duplicados_tiempo} marcas de tiempo duplicadas")
        if "sin_calibrar" in banderas:
            alertas.append(f"Sesión {q.sesion_id}: sin CALIBRADO; "
                           + ("incluida por configuración" if incluir_sc else "excluida del análisis"))
        if "cierre_implicito" in banderas:
            alertas.append(f"Sesión {q.sesion_id}: sin evento de cierre; cerrada en su última medición")

    n_sc = 0 if incluir_sc else int(mediciones["t_rel_calibracion"].isna().sum())
    return InformeCalidad(
        n_mediciones=int(len(mediciones)),
        n_sesiones=int(len(sesiones)),
        n_usuarios=int(sesiones["usuario_id"].nunique()),
        n_archivos=int(sesiones["archivo"].nunique()),
        n_analizables=int(analizable.sum()),
        n_precalibracion=int(sesiones["n_precalibracion"].sum()),
        n_sin_calibrar_excluidas=n_sc,
        n_discrepancias_estado=int(discrepa.sum()),
        alertas=alertas,
        sesiones=por_sesion,
    )


def formatear(informe: InformeCalidad, config: dict) -> str:
    cq = config["calidad"]
    lin = [
        "INFORME DE CALIDAD",
        "=" * 60,
        f"Archivos: {informe.n_archivos}   Usuarios: {informe.n_usuarios}   Sesiones: {informe.n_sesiones}",
        f"Mediciones en sesión: {informe.n_mediciones}",
        f"  analizables (post-calibración): {informe.n_analizables}",
        f"  descartadas por precalibración: {informe.n_precalibracion}",
        f"  excluidas por sesión sin calibrar: {informe.n_sin_calibrar_excluidas}",
        f"Discrepancias estado reportado vs derivado: {informe.n_discrepancias_estado} "
        f"de {informe.n_mediciones}",
        "",
        f"Por sesión (cadencia nominal {cq['cadencia_nominal_s']} s; hueco > {cq['umbral_hueco_s']} s; "
        f"ráfaga < {cq['umbral_rafaga_s']} s):",
    ]
    for q in informe.sesiones:
        lin.append(f"- {q.sesion_id}  [usuario {q.usuario_id}, {q.dispositivo}]")
        lin.append(f"    muestras {q.n_muestras}  analizables {q.n_analizables}  precalibración {q.n_precalibracion}")
        lin.append(f"    cadencia mediana {q.cadencia_mediana_s} s  p95 {q.cadencia_p95_s} s  "
                   f"huecos {q.n_huecos} (máx {q.hueco_max_s} s)  ráfagas {q.n_rafagas}")
        lin.append(f"    duplicados {q.n_duplicados_tiempo}  fuera de orden {q.n_retrocesos_tiempo}  "
                   f"discrepancias {q.n_discrepancias_estado}")
        if q.banderas:
            lin.append(f"    banderas: {', '.join(q.banderas)}")
    lin.append("")
    if informe.alertas:
        lin.append("ALERTAS:")
        lin.extend(f"  ! {a}" for a in informe.alertas)
    else:
        lin.append("Sin alertas.")
    lin.append("")
    lin.append("Nota: los ángulos son relativos a la calibración de cada sesión; "
               "dos sesiones no son comparables en términos absolutos.")
    return "\n".join(lin)
