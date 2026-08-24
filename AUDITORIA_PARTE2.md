# Auditoría Técnica Previa a la Parte 2
## Laboratorio 4 — Datos Geoespaciales, CC3084

> ## ⚠ Este documento describe el ESTADO PREVIO A LA CORRECCIÓN
>
> Se conserva como **registro histórico** del diagnóstico que motivó las correcciones. **No
> describe el estado actual del proyecto.** Los problemas que enumera ya fueron resueltos:
>
> | Hallazgo de esta auditoría | Estado actual |
> |---|---|
> | No existía ningún ráster real | **22/22 GeoTIFF Sentinel-2 L1C** descargados y validados |
> | Cobertura real 0/22 | **22/22** combinaciones lago-fecha |
> | Se descartaban CRS y transformación afín | Se conservan; dataset en **EPSG:32615** a 20 m |
> | Faltaban `shap` y `xgboost` | Pendientes de instalar antes del Ejercicio 4 |
> | Fuga vía NDVI sin analizar | **Analizada y documentada**; NDVI excluido del modelo principal |
> | Umbral de 50 µg/L sin respaldo | Respuesta principal ahora **8 µg/L** (OECD 1982) |
>
> Para el estado vigente consulte `outputs/parte2/reports/preparacion_dataset.md`,
> `outputs/parte2/reports/threshold_viability.md` y `README.md`.


**Alcance:** solo inspección. No se modificó ningún archivo, no se entrenó ningún modelo, no se generó ningún dato.
**Fecha de auditoría:** 2026-08-23
**Directorio auditado:** `C:\Users\mjyee\Downloads\DataScience\lab4data\`
**Archivo adicional auditado:** `C:\Users\mjyee\Downloads\lab4-2.ipynb` (idéntico byte a byte al `lab4-2.ipynb` ya versionado dentro de `lab4data/`, confirmado por `diff`)

---

## A. Resumen ejecutivo

1. **No existe ningún ráster real en el proyecto.** `outputs/rasters/` está vacía y una búsqueda recursiva de `.tif`, `.tiff`, `.nc`, `.geojson`, `.shp`, `.gpkg` en todo `lab4data/` no arroja resultados. Todo valor de NDVI, NDWI, clorofila-a/cianobacteria usado hasta ahora —en la Parte I y en el avance de Parte 2— proviene de generadores sintéticos en memoria, nunca de una descarga Sentinel-2.
2. **El avance de Parte 2 (`lab4-2.ipynb`) ya está commiteado**, pero no es trabajo tuyo: el mensaje del commit `90a8e22` dice textualmente *"Aunque lo suba yo, el avance es el trabajo de Pablo José Méndez, solo que no se subió al repositorio"*. Fue ejecutado en una máquina y entorno distintos a los que audito aquí (ver sección G).
3. **Cobertura real de las 22 combinaciones lago-fecha oficiales: 0/22.** La cobertura de 22/22 que sí existe es únicamente sobre datos sintéticos derivados (ver sección D).
4. **No hay manejo de CRS ni de transformación afín en ningún punto del código.** `load_bands_from_geotiffs()` en `mainlab4.py` lee el arreglo de píxeles con `rasterio` pero descarta `crs`, `transform`, `bounds` y `res`. El PDF de Parte 2 exige reproyectar a **EPSG:32615** antes de construir la cuadrícula espacial (§6.1); no existe ninguna función que lo haga.
5. **No hay polígono real de ningún lago**, solo bounding boxes rectangulares (`LAKE_BBOXES`). No se encontró GeoJSON, Shapefile ni GeoPackage.
6. **Dependencias faltantes para Parte 2:** `xgboost` y `shap` no están instalados en el entorno verificado. `requirements.txt` está desactualizado (ver sección G).
7. **Fuga de información:** el borrador de `lab4-2.ipynb` ya excluye correctamente `cyano_index`, `chlorophyll`, `NDCI`, `B04`, `B05` y `high_cyano` de los predictores, pero **no analiza la fuga indirecta de NDVI** (que usa B04) tal como pide el enunciado del PDF y tu pregunta 10.
8. ⚠️ **Hallazgo de fechas, verificable contra la fecha de hoy (2026-08-23):** el PDF de Parte 2 fija la entrega de "Avances: Ejercicios 1, 2 y 3" el **20 de agosto de 2026, 17:20** — **esa fecha ya pasó, hace 3 días**. La entrega de "Ejercicios Completos" está impresa como **17 de agosto de 2025, 23:59**, un año antes de la de avances; es casi con certeza una errata del PDF (probablemente debía decir 2026), pero incluso leída como 2026-08-17 también ya pasó. Reporto el dato tal como aparece en el documento; no lo interpreto ni lo corrijo por ti.

**Veredicto adelantado:** el proyecto **no está listo** para iniciar el modelado de Parte 2 con datos reales. Ver sección J para el detalle.

---

## B. Inventario de archivos

Árbol completo de `lab4data/` (excluyendo `.git/`):

```
lab4data/
├── .gitignore
├── AUDITORIA_PARTE2.md                                  <- este archivo (nuevo, no modifica nada existente)
├── INFORME_LAB4.md
├── INFORME_LAB4.pdf
├── Laboratorio 4. Datos Geoespaciales. 2026.pdf
├── Laboratorio 4. Parte 2. Datos Geoespaciales. 2026.pdf
├── README.md
├── __pycache__/
│   ├── lab4_analisis.cpython-311.pyc
│   └── mainlab4.cpython-311.pyc
├── lab4-2.ipynb            (413,474 bytes — commit 90a8e22, autor real: Pablo José Méndez)
├── lab4.ipynb               (207,818 bytes — Parte I + integración Actividades 5/6/8)
├── lab4_analisis.py           (7,680 bytes)
├── mainlab4.py                (23,331 bytes)
├── md2pdf.py                  (9,106 bytes — utilidad de conversión, no forma parte del análisis)
├── requirements.txt              (84 bytes)
├── resultado_ejemplo_cyanolakes.png
└── outputs/
    ├── Figure_1.png            (67,376 bytes)
    ├── Figure_2.png            (67,376 bytes — mismo tamaño que Figure_1; nombre genérico de matplotlib, sin función identificable que los genere en el código actual)
    ├── act5_comparativo_Amatitlan.png
    ├── act5_comparativo_Atitlan.png
    ├── act5_mapa_interactivo_Amatitlan.html
    ├── act5_mapa_interactivo_Atitlan.html
    ├── act6_correlacion_Amatitlan.png
    ├── act6_correlacion_Atitlan.png
    ├── act8_distribuciones_Amatitlan.png
    ├── act8_distribuciones_Atitlan.png
    ├── act8_extension_espacial.csv
    ├── actividad_4_evolucion_temporal_demo.png
    ├── actividad_4_interpretacion_demo.txt
    ├── actividad_4_picos_demo.csv
    ├── actividad_4_serie_temporal_demo.csv     <- única fuente real de las 22 combinaciones (sintética)
    ├── evidencia_indices_mapa_sintetico.png
    └── rasters/                                 <- VACÍA. No trackeada por git (directorio local, creado en una sesión anterior de ejecución de lab4.ipynb; nunca recibió archivos)
```

**Archivos geoespaciales de límites/vectores buscados y no encontrados:** `*.geojson`, `*.shp`, `*.shx`, `*.dbf`, `*.gpkg` — ninguno existe en el proyecto.

**Carpetas `data/`, `raw/`, `processed/` o similares:** no existen.

`Figure_1.png` / `Figure_2.png` fueron añadidos en el commit `f584e76` ("Actividades del 5 al 8"); no corresponden a ninguna función con nombre identificable en `mainlab4.py` ni `lab4_analisis.py` (esas producen archivos con prefijo `act5_`/`act6_`/`act8_`). Probablemente son guardados manuales (`plt.savefig` con nombre por defecto) durante una sesión interactiva anterior.

---

## C. Metadatos de los rásteres

**No aplica — no se encontró ningún archivo ráster (GeoTIFF, NetCDF u otro) en el proyecto.**

No hay ruta, tamaño, bandas, dimensiones, CRS, transformación afín, resolución, bounds, dtype, NoData ni estadísticas por banda que reportar, porque el archivo físico correspondiente no existe. Todo lo que el código produce bajo el nombre "ráster" (`chlorophyll`, `NDVI`, `NDWI`, `water_mask`, etc.) son arreglos NumPy en memoria, generados por `compute_cyan_index_for_date()` a partir de bandas simuladas por `synthetic_bands()`, y no se serializan a disco en ningún punto del flujo actual.

---

## D. Cobertura de las 22 combinaciones lago-fecha

Confirmé que las fechas y bounding boxes en `mainlab4.py` coinciden exactamente con las que indicaste (Atitlán: -91.326256 / -91.071510 / 14.594800 / 14.750979; Amatitlán: -90.638065 / -90.512924 / 14.412347 / 14.493799; 11 fechas por lago).

| Nivel de dato | Combinaciones cubiertas | Duplicadas | Faltantes |
|---|---|---|---|
| **Ráster Sentinel-2 real** (`.tif`/`.nc` en disco) | **0 / 22** | — | Las 22 |
| Resumen sintético por fecha (`outputs/actividad_4_serie_temporal_demo.csv`) | 22 / 22 | Ninguna | Ninguna |
| Tabla de píxeles sintética de Parte 2 (`lab4-2.ipynb`, generada a partir del CSV anterior) | 22 / 22 (2,226 filas válidas tras limpieza, de 3,960 sintéticas iniciales) | Ninguna | Ninguna |

Las fechas `2026-04-13` y `2026-04-28` aparecen en las listas oficiales de **ambos** lagos; esto no es una duplicación de datos, es coincidencia de calendario entre Atitlán y Amatitlán, y el código las trata correctamente como observaciones lago-fecha distintas.

**Conclusión de esta sección:** la cobertura "22/22" que muestran los outputs actuales es completamente sintética. No hay evidencia de que ninguna de las 22 escenas Sentinel-2 oficiales haya sido descargada.

---

## E. Datos reales frente a datos sintéticos

**Procedencia: 100% sintética. No se encontró evidencia de datos Sentinel-2 reales en ningún punto del proyecto.**

Evidencia concreta:

- `RUN_OPENEO` no está definida en el entorno (`echo $RUN_OPENEO` → vacío), y `mainlab4.py`/`lab4.ipynb` solo llaman a `connect_to_openeo_backend()` cuando esa variable vale `"1"`.
- No hay variables `OPENEO_*` ni `COPERNICUS_*` definidas en el entorno auditado (no se muestran valores, solo se confirmó ausencia).
- `outputs/rasters/` —el directorio donde `lab4.ipynb` guardaría los `.tif` descargados vía `cube.download()`— está vacía.
- La función `synthetic_bands()` (definida tanto en `lab4.ipynb` como recreada de forma independiente en `lab4-2.ipynb`) genera reflectancias con `numpy.random.default_rng`, sembrado por un hash SHA-256 de `f"{lake}-{date}"`. Es reproducible, pero es una simulación, no una observación satelital.
- `lab4-2.ipynb` es explícito y honesto al respecto: su propia celda introductoria dice *"En este repositorio no hay rasters de entrada... El modo demostrativo conserva las fechas y lagos oficiales, pero sus píxeles son sintéticos y no deben interpretarse como observaciones satelitales reales."* Y la salida real de ejecución confirma: `Modo de datos: demostrativo parametrizado por la serie temporal de la Parte I` — es decir, un sintético construido a partir de **otro** sintético (el CSV resumen de la Parte I), no a partir de rásteres.
- `INFORME_LAB4.md` (entregable de Parte I) ya documenta esta misma limitación en un aviso al inicio del documento.

No hay mezcla: no encontré ningún caso en que una fecha/lago tenga datos reales y otra sintéticos. Es sintético de punta a punta.

---

## F. Límites, máscaras y georreferenciación

**Polígonos de lago:** no existen. `LAKE_BBOXES` en `mainlab4.py` define únicamente un rectángulo (west/east/south/north) por lago — es un *bounding box*, no la geometría real de la costa. No se encontró ningún GeoJSON/Shapefile/GeoPackage con el contorno verdadero de Atitlán ni de Amatitlán en el proyecto ni en `Downloads/`.

**Máscara de agua:** `wbi_vectorized()` en `mainlab4.py` implementa un índice compuesto (MNDWI/NDWI/AWEI/NDVI con filtro urbano) que sí es una lógica de máscara de agua real y reutilizable, pero nunca se ha ejecutado sobre píxeles reales — solo sobre las bandas sintéticas.

**dataMask:** el evalscript `CYANOLAKES_NUMERIC_EVALSCRIPT` (para Sentinel Hub) sí solicita la banda `dataMask` y la multiplica en la salida, pero esa ruta de código (Sentinel Hub, distinta de la ruta openEO) nunca se ha invocado contra el servicio real.

**SCL / QA60 / máscara de nubes:** no existe ningún manejo de `SCL` ni `QA60` en `mainlab4.py`, `lab4_analisis.py` ni en ninguno de los notebooks. En `lab4-2.ipynb`, la columna `cloud_free` existe en el esquema de datos pero está **codificada como `True` para el 100% de las filas sintéticas** (`"cloud_free": True` en el generador) — es un campo de esquema correcto conceptualmente, pero sin ninguna fuente real de nubosidad detrás todavía.

**CRS y transformación afín — pérdida confirmada:**
`load_bands_from_geotiffs()` (`mainlab4.py`) abre cada GeoTIFF con `rasterio.open(path)` y ejecuta `src.read(1)`, devolviendo solo el arreglo NumPy normalizado. **No se conserva `src.crs`, `src.transform`, `src.bounds` ni `src.res` en ningún punto posterior del pipeline.** Toda la cadena de cálculo (`NDVI`, `NDWI`, `wbi_vectorized`, `FAI`, `NDCI`, `compute_cyan_index_for_date`) opera sobre arreglos NumPy puros, sin ninguna noción de coordenada geográfica por píxel.

Esto es un problema directo para el Ejercicio 6 del PDF de Parte 2, que exige:
1. Reproyectar a **EPSG:32615** (WGS 84 / UTM zona 15N) antes de construir la cuadrícula de 1×1 km.
2. Asignar cada observación a un bloque espacial y visualizarlo en un mapa.

Ninguna de las dos cosas es posible con el código actual sin antes escribir una función que preserve `crs`/`transform` en la carga y derive las coordenadas `x, y` de cada píxel a partir de ellos (`rasterio.transform.xy`, por ejemplo).

**Nota sobre resolución/área de píxel:** en la integración de Actividad 5-8 (`lab4.ipynb`) se fijó `AREA_PIXEL_HA = 0.01` asumiendo píxeles Sentinel-2 reales de 10 m, pero se aplicó sobre una grilla sintética de `shape=(96, 96)` sin resolución geoespacial real. Es un supuesto incorrecto si se reutiliza tal cual sobre datos reales: la grilla real de un lago de ~13,000 ha (Atitlán) a 10 m de resolución tendría del orden de cientos de miles de píxeles, no 9,216.

---

## G. Dependencias disponibles

⚠️ **Aviso de entorno importante:** detecté que existen **dos entornos distintos** involucrados en este proyecto, y solo pude verificar uno directamente.

1. **Entorno de esta auditoría** (el que uso para ejecutar comandos ahora): Python 3.11.9, sin entorno virtual activo, intérprete en `C:\Users\mjyee\AppData\Local\Programs\Python\Python311\python.exe`. Las versiones de abajo están verificadas *en vivo* contra este entorno.
2. **Entorno en el que realmente se ejecutó `lab4-2.ipynb`**: según sus propios metadatos (`kernelspec`) y su salida impresa (`Directorio de trabajo: d:\lab4data`), corrió con **Python 3.13.5** dentro de un entorno virtual llamado **`.venv-1`**, en la unidad `D:\`. Esa unidad y ese entorno **no existen en esta máquina/sesión de auditoría** (`D:/` no es accesible), así que **no puedo verificar qué paquetes ni qué versiones tiene ese entorno real** — solo puedo reportar lo que sus propios outputs implican que sí corrió sin error ahí (numpy, pandas, matplotlib, seaborn).

Versiones verificadas en el entorno (1), con `import` directo:

| Paquete | Versión instalada | ¿Requerido por Parte 2? |
|---|---|---|
| numpy | 2.4.3 | Sí |
| pandas | 2.3.3 | Sí |
| rasterio | 1.4.4 | Sí |
| scikit-learn | 1.9.0 | Sí (Regresión Logística, Random Forest, `GroupKFold`) |
| geopandas | 1.1.4 | Sí (bloques espaciales, reproyección) |
| **shap** | **no instalado** | **Sí — obligatorio, Ejercicio 8.2** |
| **xgboost** | **no instalado** | **Sí, si se elige XGBoost sobre Gradient Boosting genérico (Ejercicio 4.1)** |
| matplotlib | 3.11.1 | Sí |
| seaborn | 0.13.2 | Sí |
| pyproj | 3.7.2 | Sí (reproyección a EPSG:32615) |
| joblib | 1.5.3 | Sí (opcional, persistencia de modelos) |
| openeo | 0.51.0 | Sí (si se busca descargar datos reales) |
| folium / branca | 0.20.0 / 0.8.2 | Ya usados en Actividad 5 de Parte I |

**`requirements.txt` está desactualizado.** Contiene solo: `openeo, numpy, matplotlib, rasterio, xarray, pandas, nbformat, nbclient, ipykernel`. No incluye `scikit-learn`, `seaborn`, `geopandas`, `pyproj`, `folium`, `branca`, ni (los que faltan) `shap`/`xgboost`, pese a que varios de esos ya se usan en el código versionado. Si otro miembro del equipo clona el repo y corre `pip install -r requirements.txt`, no podrá ejecutar `lab4_analisis.py` ni `lab4-2.ipynb` tal cual están.

No se inspeccionó ni se muestra ningún valor de variable de entorno sensible, token o contraseña.

---

## H. Riesgos de fuga de información

Cadena de dependencia confirmada leyendo `mainlab4.py`:

```
B04, B05  →  NDCI = (B05 − B04) / (B05 + B04)
NDCI      →  chl_from_ndci(NDCI)  →  chlorophyll / cyano_index
cyano_index  →  high_cyano (variable respuesta binaria, umbral 50 µg/L en lab4-2.ipynb)
```

**Deben excluirse de los predictores (fuga directa):**
- `cyano_index`, `chlorophyll` — son la base literal de la respuesta.
- `NDCI` — es el insumo directo de la calibración de clorofila.
- `B04`, `B05` — son los únicos insumos espectrales del NDCI; incluirlos permite reconstruir el NDCI (y por tanto la respuesta) casi exactamente.

`lab4-2.ipynb` ya excluye correctamente estas cinco variables (`leakage_columns = ["cyano_index", "chlorophyll", "NDCI", "B04", "B05", "high_cyano"]`) y lo verifica con un `assert`.

**Punto no resuelto — NDVI y fuga indirecta (tu pregunta explícita):**
`NDVI = (B08 − B04) / (B08 + B04)` comparte el término `B04` con el numerador y denominador del NDCI. Esto significa que NDVI **no es independiente** de la variable con la que se construyó la respuesta: parte de su varianza está correlacionada con B04, que es exactamente el canal que también mueve al NDCI hacia arriba o abajo. No es una fuga tan directa como incluir B04 crudo (el NDVI combina B04 con B08, que no interviene en el NDCI), pero **es una fuga parcial, no cero**, y el borrador actual de `lab4-2.ipynb` **mantiene NDVI en el conjunto final de predictores sin discutir este acoplamiento** — su celda 8 justifica NDVI solo por su valor ecológico (contraste vegetación/agua), sin mencionar la superposición con B04.

Recomendación técnica (no es una decisión que yo deba tomar por ti, es para que la definas con criterio): antes de aceptar NDVI como predictor, correr una comparación de desempeño con y sin NDVI, o revisar su importancia/SHAP con escepticismo adicional si aparece como variable dominante — un peso desproporcionado de NDVI en el modelo final sería una señal de alerta de fuga residual, no solo de señal ecológica genuina.

**Otro caso relacionado que vale la pena señalar:** `FAI = B07 − B04 − (B8A − B04) × (783−665)/(865−665)` también contiene `B04` y, en el script original de CyanoLakes, se usa junto al NDCI para decidir si un píxel es "vegetación flotante" antes de aplicar la paleta de clorofila — es decir, FAI y NDCI comparten rol funcional en el algoritmo de origen. `lab4-2.ipynb` ya excluye FAI del conjunto final de predictores (no aparece en `predictor_columns`), lo cual es consistente con evitar esta fuga, aunque el notebook no lo explica como decisión de fuga sino que simplemente no lo selecciona.

**Variables correctamente no usadas como predictoras por buen criterio (no por fuga, sino por riesgo de sobreajuste geográfico):** `lake`, `x`, `y` — el notebook ya documenta explícitamente que conservarlas como predictoras "podría hacer que el modelo memorice la geografía de los lagos en vez de aprender una señal espectral generalizable". Es el razonamiento correcto y es consistente con lo que pide la Parte 2 para la validación espacial y de generalización entre lagos (Ejercicios 6 y 7).

---

## I. Recursos faltantes

Para poder ejecutar honestamente los Ejercicios 1–9 del PDF de Parte 2 sobre datos reales, falta:

1. **Rásteres Sentinel-2 reales** para las 22 combinaciones lago-fecha (o, como mínimo, una tabla de píxeles exportada desde rásteres reales) — actualmente no existe ninguno.
2. **Credenciales/conexión Copernicus activa** (`RUN_OPENEO=1` + autenticación) para poder descargarlos, o una fuente alterna (Sentinel Hub con el evalscript numérico ya preparado en `CYANOLAKES_NUMERIC_EVALSCRIPT`).
3. **Máscara de nubes real** (SCL o QA60) — no implementada; `cloud_free` es un campo de esquema vacío de contenido real.
4. **Polígono real de cada lago** (GeoJSON/Shapefile) para distinguir "dentro del lago" de "dentro del bounding box" — actualmente todo el recorte se hace por rectángulo.
5. **Una función de carga que preserve CRS y transformación afín**, y que derive `x, y` reales por píxel (hoy se descartan en `load_bands_from_geotiffs`).
6. **Función de reproyección a EPSG:32615** — no existe ninguna llamada a `pyproj`/`geopandas.to_crs` en el proyecto todavía, pese a que ambas librerías ya están disponibles.
7. **Función de cuadrícula espacial de ~1×1 km y asignación de bloques** (Ejercicio 6.1–6.2) — no implementada.
8. **Estrategia de validación espacial** (`GroupKFold` o equivalente) y **estrategia de validación temporal** — no implementadas ni referenciadas en ningún archivo.
9. **`xgboost` y `shap`** — no instalados en el entorno verificado; se requieren para los Ejercicios 4.1 y 8.2.
10. **`requirements.txt` actualizado** con todas las dependencias realmente usadas (scikit-learn, geopandas, pyproj, shap, xgboost, seaborn, folium, branca).
11. **Un entorno de ejecución único y documentado para el equipo** — hoy hay evidencia de al menos dos entornos distintos (esta auditoría en Python 3.11.9 sin venv, y `.venv-1` con Python 3.13.5 en `D:\lab4data` usado por tu compañero), lo cual es un riesgo de reproducibilidad para un laboratorio que el PDF pide versionar y evaluar por contribución individual.
12. **Código de reconstrucción espacial de predicciones** (Ejercicio 9) y de comparación de mapas — no implementado, y depende de que primero exista (5).

---

## J. Veredicto final: **NO LISTO PARA IMPLEMENTAR**

Razones concretas, en orden de bloqueo:

1. **No hay datos reales que modelar.** Toda observación disponible hoy —tanto en la Parte I como en el borrador de Parte 2— es sintética. Entrenar Regresión Logística, Random Forest y Gradient Boosting/XGBoost ahora mismo produciría modelos que aprenden la lógica de un generador aleatorio sembrado por hash, no cianobacteria real. El propio PDF exige explícitamente partir "de los rasters obtenidos en la Parte I" (§1.1), y esos rasters no existen.
2. **Falta infraestructura geoespacial obligatoria y explícitamente pedida:** reproyección a EPSG:32615 y cuadrícula de 1×1 km (Ejercicio 6) no se pueden construir porque el pipeline actual descarta CRS y transformación afín al cargar cada banda.
3. **Faltan dos dependencias obligatorias** (`shap`, `xgboost`) en el entorno verificado.
4. **La fuga de información no está completamente resuelta**: la exclusión de B04/B05/NDCI/chlorophyll/cyano_index es correcta, pero la fuga parcial vía NDVI (comparte B04) no se ha analizado ni documentado todavía, tal como pide expresamente el enunciado ("directa o indirectamente").
5. **Hay un riesgo de plazo activo y verificable hoy:** la fecha de avances del PDF (20 de agosto de 2026) ya pasó al momento de esta auditoría (23 de agosto de 2026).

**Qué sí está listo:** el diseño conceptual de limpieza, la justificación del umbral de 50 µg/L con bibliografía real (OECD 1982, WHO 2021, Mishra et al. 2019), el razonamiento sobre desbalance de clases, y la exclusión de `lake`/coordenadas como predictores en `lab4-2.ipynb` son metodológicamente sólidos y reutilizables **en cuanto existan datos reales que alimenten esa misma lógica**.
