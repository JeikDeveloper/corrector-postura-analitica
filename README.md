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

1. **Veredicto de estructura**: correcta o con advertencias, con la lista de
   problemas ubicados por archivo y número de línea (eventos desconocidos, líneas sin marca
   de tiempo, horas fuera de orden, mediciones fuera de sesión, estados que no coinciden con
   los umbrales, sesiones sin calibrar o sin cierre).
2. **Sesiones detectadas** con inicio, fin, duración, muestras y tipo de cierre.
3. **Análisis postural**, con *Mostrar*: conjunto completo, por persona o por sesión, y dos
   pestañas:
   - *4.1 Mapa de densidad*, con *Método*: densidad kernel, hexágonos o tridimensional.
     Tridimensional muestra la misma densidad kernel como superficie (altura = tiempo de
     permanencia), con el rectángulo de umbrales en el piso. En la página se gira
     y se acerca con el mouse (Plotly, cargado desde jsDelivr al abrir la vista, así que
     necesita internet); la imagen fija se descarga en PNG, SVG, PDF o JPG con la cámara
     de `figura.tres_d`.
   - *4.3 Análisis descriptivo*: estadística descriptiva de F y L separada por el estado
     BIEN/MAL que reporta el dispositivo (módulo `postura.estadistica`): n, media, desviación
     estándar, mínimo, percentiles (`estadistica.percentiles`), máximo, rango intercuartílico,
     asimetría y curtosis; episodios (rachas consecutivas BIEN o MAL y su duración); y qué
     umbral excede cada muestra MAL. Figuras deterministas (`postura.figuras.estadistica`):
     histogramas, box plots, violin plots y series temporales por sesión, descargables en
     PNG (300 ppp), SVG (texto editable) y PDF, más un reporte completo en PDF de varias
     páginas y la tabla en CSV. API: `POST /api/analizar/estadistica` (JSON),
     `/api/analizar/estadistica.<png|svg|pdf>`, `/api/analizar/estadistica/reporte.pdf` y
     `/api/analizar/estadistica.csv`.
   - *4.4 Analítica diagnóstica* (`postura.diagnostica`, `postura.indicadores`):
     - *Narración de datos*: relato corto generado a partir de los datos de la selección
       (titular, contexto, hallazgos, qué significa, qué revisar y limitaciones) con un
       gráfico anotado que cuenta el mensaje principal. Es determinista: cada frase sale
       de un indicador y las palabras («frecuentes», «breves», «estable») dependen de los
       umbrales de `diagnostica.narracion`. Se descarga como informe PDF o se copia como texto.
     - *Indicadores descriptivos*: 8 indicadores con definición operativa, fórmula e
       interpretación, y su valor por sesión (figura y CSV).

     API: `POST /api/diagnostica` (JSON), `/api/diagnostica/figura.<png|svg|pdf>`
     (figura: narrativa o indicadores), `/api/diagnostica/informe.pdf` y
     `/api/diagnostica/indicadores.csv`.
   - *4.5 Confiabilidad y tasa de fallos* (`postura.confiabilidad`): con un selector de qué
     cuenta como fallo:
     - *Fallos del dispositivo*: pérdida de conexión, conexión fallida, hueco de datos y sesión
       sin cierre (un `Connection lost` justo después de un `APAGADO` no cuenta: es la
       desconexión esperada al apagar). Operación = tiempo con datos.
     - *Fallos posturales*: cada episodio MAL. Operación = tiempo en BIEN.

     Calcula MTBF, MTTR, disponibilidad y tasa de fallos; ajusta Weibull por máxima
     verosimilitud con censura (β, η, B10, tiempo medio y probabilidad de fallar en el
     horizonte); detecta ventanas anómalas con Isolation Forest (puntaje y severidad media, alta
     o crítica); y predice fallos inminentes con regresión logística (validación cruzada, AUC,
     coeficientes y riesgo por sesión). Las alertas de la página salen de esos tres. Si no hay
     datos suficientes para un modelo, no se entrena y se informa el motivo y el mínimo
     configurado en `confiabilidad`. API: `POST /api/confiabilidad/resumen`, `/metricas`,
     `/weibull`, `/anomalias`, `/prediccion`, `/alertas`, `/figura.<png|svg|pdf>`,
     `/informe.pdf` y `/fallos.csv` (campo `modo`: dispositivo o postura). Usa scikit-learn.
   - *4.2 Análisis ergonómico (ISO 11226)*: lo que plantea la presentación "Posture Analysis
     During Desk Work of Researchers" (Betancur, UMB). Zonas óptima, aceptable y extrema
     (BGIA-Report 2/2007, ISO 11226, EN 1005-4) con la dispersión |σ| = √(σα² + σβ²) y el
     desplazamiento |R| = √(α̅² + β̅²); los cocientes σG y RG respecto al conjunto; paneles
     por momento del día (mañana, tarde, noche) con ANOVA; un panel por sujeto; histogramas
     de cada ángulo coloreados por zona; y la tabla de métricas por persona y sesión. Zonas,
     franjas horarias y colores están en `ergonomia` de
     [config/default.yaml](config/default.yaml). La relación con el dolor percibido no se
     calcula, porque el registro no trae ese dato.
4. **Descargas**: PNG (300 ppp), SVG, PDF, JPG, los datos normalizados en CSV y las
   métricas ergonómicas en CSV.

El servidor escucha solo en este equipo (`web.host: 127.0.0.1`). No guarda nada: los
archivos subidos se procesan en memoria, y la página los reenvía con cada figura o
descarga. Cada subida admite hasta `web.max_mb_subida` (4 MB) sumando todos los archivos.

### Publicarla en Vercel

El repositorio ya trae lo necesario: [app.py](app.py) (punto de entrada que Vercel busca)
y [vercel.json](vercel.json). En [vercel.com](https://vercel.com): *Add New → Project*,
importar este repositorio de GitHub y *Deploy*, sin cambiar ninguna opción. Cada
`git push` a `main` vuelve a publicarla. El límite de 4 MB viene de Vercel, que no acepta
peticiones de más de 4,5 MB.

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

`mediciones.parquet`: `archivo`, `linea` (línea del archivo de origen), `dispositivo`, `usuario_id`, `sesion_id` (str), `t`
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
