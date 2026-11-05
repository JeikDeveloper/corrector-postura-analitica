# corrector-postura-analitica

Ingesta, validación y normalización de registros de un dispositivo corrector de postura,
con informe de calidad y mapa de densidad conjunta de los dos ángulos posturales.

Funciona con un solo archivo de registro y escala sin cambios a muchos archivos de muchos
usuarios.

> **Advertencia de dominio.** `CALIBRADO` fija el cero del sensor en cada sesión. Los ángulos
> son relativos a la calibración de su propia sesión: **dos sesiones no son comparables en
> términos absolutos.** Toda figura que agrega varias sesiones o usuarios lo dice en su pie.

## Instalación

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
pytest
```

## Página web

```bash
postura web            # abre http://127.0.0.1:8000 en el navegador
```

En Windows también sirve hacer doble clic en `abrir_web.bat`.

En la página se sube uno o varios `.txt` (arrastrando o eligiendo). A cada archivo se le
puede asignar una **persona** (opcional): el formato del registro no la trae, y es lo que
permite filtrar el diagrama por persona. Al presionar *Analizar* se abre un modal que
confirma qué archivos se subieron (tamaño, líneas, mediciones, sesiones y persona) y el
veredicto.

Antes de procesar, cada archivo pasa un filtro (`validacion.revisar_archivo`), el mismo en la web y en `postura ingerir`. Se **rechaza
sin procesarlo** si no es `.txt`, está vacío, es binario (PDF, imagen, Office…), no es UTF-8,
ninguna de sus primeras líneas tiene marca `HH:MM:SS.mmm`, o no tiene mediciones o
`Connected`. Si se suben varios, se procesan solo los aceptados y el modal lista los
rechazados con su motivo. Colores: verde = todo subido y estructura correcta; amarillo =
advertencias o parte rechazada; rojo = nada aceptado.

Luego la página muestra:

1. **Veredicto de estructura**: correcta, con advertencias o incorrecta, con la lista de
   problemas ubicados por archivo y número de línea (eventos desconocidos, líneas sin marca
   de tiempo, horas fuera de orden, mediciones fuera de sesión, estados que no coinciden con
   los umbrales, sesiones sin calibrar o sin cierre).
2. **Sesiones detectadas** con inicio, fin, duración, muestras y tipo de cierre.
3. **Mapa de densidad**, con *Mostrar*: conjunto completo, por persona o por sesión; en
   variante kernel o hexágonos.
4. **Descargas**: PNG (300 ppp), SVG, PDF, JPG y los datos normalizados en CSV.

El servidor escucha solo en este equipo (`web.host: 127.0.0.1`); los archivos subidos se
procesan en memoria y no se guardan en disco.

## Línea de comandos

```bash
postura ingerir datos/*.txt --salida salidas/mediciones.parquet
postura calidad salidas/mediciones.parquet [--json salidas/calidad.json] [--estricto]
postura figura densidad salidas/mediciones.parquet --usuario U01 --salida salidas/
```

- `ingerir` escribe `mediciones.parquet`, `mediciones.sesiones.parquet` e `ingesta.log`
  (eventos desconocidos con su texto literal, mediciones fuera de sesión, sesiones sin
  calibrar, etc.). Los comodines se expanden también en Windows.
- `calidad` imprime cadencia, huecos, ráfagas, duplicados, marcas fuera de orden y
  discrepancias entre estado reportado y derivado. Con `--estricto` sale con código 3 si
  hay discrepancias.
- `figura densidad` acepta `--usuario` y `--sesion` (repetibles; sin filtros, el conjunto
  completo), `--metodo kde|hex|auto` e `--incluir-sin-calibrar`. Exporta PNG a 300 ppp, SVG
  y el pie de figura en `.pie.txt`.
- `--config otra.yaml` fusiona un YAML sobre [config/default.yaml](config/default.yaml)
  (también vale la variable de entorno `POSTURA_CONFIG`).

## Especificación del formato de entrada

Archivos de texto plano, UTF-8, una línea por evento, orden cronológico ascendente.
Toda línea empieza con una marca de tiempo `HH:MM:SS.mmm` seguida de un espacio.

### Líneas de medición

```
16:15:23.487 F = 2º, L = 2º, BIEN
16:18:06.290 F = -14º, L = 9º, MAL
16:17:05.820 F = - 2º, L = 0º, BIEN
```

Reglas:

- `F` es el ángulo frontal, `L` el lateral. Ambos en grados enteros, con el carácter `º`
  (U+00BA MASCULINE ORDINAL INDICATOR, **no** el símbolo de grado U+00B0).
- El signo negativo puede venir pegado (`-14º`) o **separado por un espacio** (`- 2º`).
  Ambas formas son válidas y significan lo mismo.
- El tercer campo es el estado, con dos valores posibles: `BIEN` o `MAL`.
- Resolución de un grado entero. No hay decimales.
- Cadencia nominal de una muestra por segundo.

### Líneas de evento del sistema

Siete tipos observados, sin la estructura de medición:

| Línea | Significado |
|---|---|
| `Connecting to <NOMBRE> ...` | Intento de conexión. `<NOMBRE>` es el nombre BLE del dispositivo y debe capturarse. |
| `Connected` | Inicio efectivo de sesión. |
| `CALIBRADO` | Se fija el cero de referencia. |
| `APAGADO` | Apagado voluntario del dispositivo. |
| `Disconnected` | Fin de sesión. |
| `Connection lost` | Fin de sesión por pérdida de enlace. |
| `Connection failed: gatt status <N>` | Intento de conexión fallido, NO abre sesión. |

El parser es tolerante: un evento desconocido se registra en el log de ingesta con su texto
literal y la ingesta continúa. Nunca aborta por una línea rara.

Tolerancias adicionales de esta implementación (no forman parte del formato oficial):
`°` (U+00B0) en lugar de `º`, el signo menos `−` (U+2212), BOM UTF-8 inicial y finales de
línea CRLF.

### Lo que el formato no trae

- **Fecha.** Las líneas solo traen la hora. La fecha se toma del nombre de archivo
  (`AAAA-MM-DD`) o de `fecha.por_defecto`. Si no hay ninguna, se usa una fecha marcador
  (`1900-01-01`) y la sesión queda con `fecha_desconocida = True`. Un retroceso brusco de la
  hora se interpreta como cruce de medianoche.
- **Usuario.** Se resuelve desde `usuarios.mapa_archivos`, luego desde el patrón del nombre
  de archivo (`U01_...txt`), y si no, `desconocido`. `usuario_id` ya es columna de primer
  nivel para cuando el formato lo traiga.

## Semántica

- **Sesión:** de `Connected` al primero de `Disconnected`, `Connection lost` o `APAGADO`.
  Sin cierre (fin de archivo u otro `Connected`) → se cierra en su última medición con
  `cierre_implicito`. Identificador determinista: `<archivo>-sNN-<HHMMSSmmm de Connected>`.
- **Calibración:** las muestras anteriores al `CALIBRADO` de su sesión tienen
  `post_calibracion = False` y se excluyen de estadística y figuras. Sesión sin `CALIBRADO`
  → `sin_calibrar`, excluida salvo `sesiones.incluir_sin_calibrar: true`. Con varios
  `CALIBRADO` en una sesión vale el último (`sesiones.calibracion_multiple`) y se marca
  `calibracion_multiple`.
- **Umbrales** (configurables): `MAL` si `F > 25` o `F < -10` o `L > 15` o `L < -15`.
  `estado_derivado` se recalcula y se compara con el reportado; las discrepancias se avisan
  en la ingesta y en el informe de calidad.
- Las mediciones fuera de una sesión se descartan, se cuentan y se avisan en el log.

## Modelo de datos

`mediciones.parquet`: `archivo`, `dispositivo`, `usuario_id`, `sesion_id` (str), `t`
(datetime), `t_rel_sesion`, `t_rel_calibracion` (float, s; NaN si sin calibrar), `frontal`,
`lateral` (int16, grados), `estado`, `estado_derivado` (category BIEN/MAL),
`post_calibracion` (bool).

`mediciones.sesiones.parquet`: una fila por sesión con identificadores, `inicio`,
`calibracion`, `fin`, `evento_cierre`, `duracion_s`, `n_muestras`, `n_precalibracion`,
`n_post_calibracion`, `n_calibraciones`, `cadencia_mediana_s` y las banderas `sin_calibrar`,
`cierre_implicito`, `calibracion_multiple`, `fecha_desconocida`.

Ver [src/postura/modelo.py](src/postura/modelo.py).

## La figura

Mapa de densidad conjunta: lateral en X, frontal en Y; densidad kernel gaussiana escalada a
muestras por grado² (tiempo de permanencia) con curvas de nivel rellenas; histogramas
marginales de 1°; rectángulo punteado de umbrales; paleta viridis. Las muestras en zonas
por debajo del nivel mínimo de relleno se dibujan como puntos, para que el suavizado no
esconda excursiones aisladas. Si la densidad kernel no es estimable (menos de
`figura.kde.min_muestras` o varianza nula en un eje) se muestra el conteo por celda de 1°
y el pie lo dice. Variante en hexágonos para conjuntos grandes (`figura.metodo: hex`, o
`auto` a partir de `figura.auto_umbral_muestras`).

Nada se rellena, interpola ni simula: si un filtro no tiene datos, la figura lo dice.

## Estructura

```
config/default.yaml        umbrales, rutas y todos los parámetros numéricos
src/postura/parser.py      texto crudo → eventos
src/postura/sesiones.py    eventos → sesiones + calibración
src/postura/ingesta.py     orquestación de varios archivos, parquet
src/postura/modelo.py      esquema, estado derivado, validación de tipos
src/postura/calidad.py     cadencia, huecos, duplicados, discrepancias
src/postura/figuras/       densidad.py (figura), estilo.py (paleta y tipografía)
src/postura/cli.py         comando `postura`
tests/                     pruebas y fixtures (referencia.txt: 363 líneas, 347 mediciones, 3 sesiones)
```

Fuera de alcance en esta iteración: modelado y predicción.
