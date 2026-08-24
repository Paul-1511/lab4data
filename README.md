# Laboratorio 4 — Análisis de datos geoespaciales

Detección de cianobacterias en los lagos de **Atitlán** y **Amatitlán** (Guatemala) con
imágenes **Sentinel-2 L1C** del programa Copernicus.
Universidad del Valle de Guatemala — CC3084 Data Science.

> **Estado.** La Parte 1 está regenerada con **22 escenas Sentinel-2 reales**. Los
> artefactos de la primera versión, que usaban datos simulados, se conservan marcados como
> obsoletos y **no deben citarse como resultados ecológicos**.

---

## Estructura del proyecto

```
lab4data/
├── mainlab4.py                       Funciones base: índices, máscara WBI, clorofila, openEO
├── lab4_analisis.py                  Utilidades de visualización de la primera versión
├── descargar_rasters.py              Adquisición y validación de los 22 GeoTIFF (openEO)
├── regenerar_parte1_real.py          Pipeline de la Parte 1 con datos reales (act. 1-8)
├── preparar_dataset_ml.py            Parte 2: dataset por píxel (ejercicios 1-3)
├── modelos_parte2.py                 Parte 2: modelos y validación (ejercicios 4-7)
├── explicabilidad_mapas_parte2.py    Parte 2: interpretabilidad y mapas (ejercicios 8-10)
├── md2pdf.py                         Conversión de Markdown a PDF (NO usado en Parte 2)
├── lab4.ipynb                        Cuaderno de la Parte 1 (flujo real)
├── lab4-2.ipynb                      Cuaderno de la Parte 2 (ejercicios 1-10, 79 celdas)
├── INFORME_LAB4.md / .pdf            Informe de la Parte 1 (público no técnico)
├── INFORME_LAB4_PARTE2.md            Informe académico de la Parte 2 (SOLO Markdown)
├── ARTICULO_TECNICO_LAB4.md          Artículo técnico-pedagógico extendido (~6,600 palabras)
├── AUDITORIA_PARTE2.md               Auditoría previa a la Parte 2 (estado histórico)
├── Lago_Atitlan.geojson              Rectángulo del área de estudio (NO es el contorno real)
├── Lago_Amatitlan.geojson            Rectángulo del área de estudio (NO es el contorno real)
├── requirements.txt
└── outputs/
    ├── rasters/                      22 GeoTIFF reales (NO versionados, ~244 MB)
    ├── manifests/                    Manifiesto y reporte de validación de los rásteres
    ├── parte1_real/                  RESULTADOS VÁLIDOS de la Parte 1
    │   ├── tables/   figures/   maps/   reports/   validation/
    ├── parte2/                       Resultados VÁLIDOS de la Parte 2 (perfil standard)
    │   ├── data/                     Dataset por píxel (Parquet NO versionado)
    │   ├── metrics/ figures/ tuning/ splits/   Ejercicios 4-7
    │   ├── interpretability/         Ejercicio 8 (importancia nativa, permutation, SHAP)
    │   ├── maps/probability/         Ejercicio 9: 22 GeoTIFF (NO versionados)
    │   ├── maps/errors_oof/          Ejercicio 9: errores out-of-fold (Parquet NO versionado)
    │   ├── maps/figures/             Ejercicio 9: PNG versionables
    │   ├── conclusions/              Ejercicio 10: conclusiones calculadas
    │   ├── reports/                  Los 4+ reportes Markdown/TXT de la Parte 2
    │   └── smoke/                    Perfil de prueba, NUNCA entregable (NO versionado)
    └── *_demo.*, act5_*, act6_*, act8_*   OBSOLETOS (versión simulada de la Parte 1)
```

### Resultados reales frente a resultados demostrativos/smoke

| Ubicación | Naturaleza | ¿Usar? |
|---|---|---|
| `outputs/parte1_real/` | 22 escenas Sentinel-2 reales | **Sí** |
| `outputs/manifests/` | Metadatos reales de los rásteres | **Sí** |
| `outputs/parte2/` (perfil **standard**) | Dataset y modelos sobre los 3,756,510 píxeles reales | **Sí** |
| `outputs/parte2/smoke/` | Muestra reducida para probar que el pipeline corre | **No.** Nunca es conclusión del laboratorio |
| `outputs/*_demo.*`, `act5_*`, `act6_*`, `act8_*`, `evidencia_indices_mapa_sintetico.png` | Generados con `synthetic_bands()` | **No.** Solo registro histórico |

---

## Instalación

```bash
python -m pip install -r requirements.txt
```

Probado con **Python 3.11.9**.

---

## Reproducir el trabajo

### 1. Descarga de los rásteres (ya ejecutada)

Requiere una cuenta de Copernicus. La autenticación es por OIDC: el script muestra un
enlace y un código para autorizar en el navegador. **No se guardan credenciales ni tokens
en el repositorio.**

```bash
python descargar_rasters.py --dry-run     # verifica las 22 solicitudes, sin descargar
python descargar_rasters.py --download    # descarga (reanudable, no sobrescribe válidos)
python descargar_rasters.py --validate    # audita los GeoTIFF y genera los manifiestos
```

Este paso **ya se completó**: los 22 GeoTIFF están en `outputs/rasters/`. No es necesario
repetirlo salvo que se borren.

### 2. Regenerar la Parte 1 (actividades 1 a 8)

No se conecta a Copernicus: reutiliza los GeoTIFF ya descargados.

```bash
python regenerar_parte1_real.py --dry-run    # verifica rásteres, escala y evalscript
python regenerar_parte1_real.py --build      # tablas, figuras, mapas e informe técnico
python regenerar_parte1_real.py --validate   # validación completa
```

### 3. Informe

```bash
python md2pdf.py INFORME_LAB4.md INFORME_LAB4.pdf "Informe Lab 4 - Cianobacterias Atitlan y Amatitlan" "Laboratorio 4 - Cianobacterias en Atitlan y Amatitlan - UVG"
```

### 4. Parte 2 (dataset de Machine Learning)

```bash
python preparar_dataset_ml.py --dry-run
python preparar_dataset_ml.py --build
python preparar_dataset_ml.py --validate
```

### 5. Modelos (Ejercicios 4-7: aleatorio, espacial, temporal, entre lagos)

```bash
python modelos_parte2.py --dry-run --profile standard
python modelos_parte2.py --all --profile standard --n-jobs 8   # ~80 min
python modelos_parte2.py --validate --profile standard
```

Predictores: `B02, B03, B07, B08, B8A, B11, B12, NDWI`. Respuesta: `high_cyano_8`
(clorofila-a ≥ 8 µg/L). El código incluye *asserts* que bloquean cualquier variable con
fuga de información (B04, B05, NDCI, clorofila, FAI, NDVI, coordenadas, lago, fecha).

### 6. Interpretabilidad y mapas (Ejercicios 8-10)

```bash
python explicabilidad_mapas_parte2.py --dry-run --profile standard
python explicabilidad_mapas_parte2.py --all --profile standard --n-jobs 8   # ~44 min
python explicabilidad_mapas_parte2.py --report-only --profile standard      # genera el .md
python explicabilidad_mapas_parte2.py --validate --profile standard
```

Genera 22 GeoTIFF de probabilidad (no versionados), mapas de error out-of-fold, SHAP,
importancia nativa/permutation, y `INFORME_LAB4_PARTE2.md`. **No genera PDF.**

### 7. Cuadernos

`lab4.ipynb` (Parte 1) y `lab4-2.ipynb` (Parte 2, ejercicios 1-10) usan **rutas relativas**
y deben ejecutarse desde la raíz del repositorio. Ambos cargan resultados ya calculados;
no reentrenan modelos ni recalculan SHAP/OOF al ejecutarse, y no contienen credenciales ni
resultados sintéticos o de perfil `smoke` presentados como reales.

### 8. Informes y artículo

- `INFORME_LAB4_PARTE2.md` — informe académico con cifras reales y figuras incrustadas
  (generado por `--report-only`, **solo Markdown, nunca PDF**).
- `ARTICULO_TECNICO_LAB4.md` — artículo técnico-pedagógico independiente, redactado a mano,
  pensado para que alguien sin experiencia previa en el proyecto pueda entenderlo y
  reproducirlo.

---

## Decisiones técnicas y correcciones aplicadas

| Aspecto | Decisión |
|---|---|
| **Escala radiométrica** | Los GeoTIFF guardan números digitales en float32. Reflectancia = **DN × 0.0001** (`scale` oficial de la colección, `offset` 0), aplicada **una sola vez**. `normalize_band()` se corrigió: antes solo dividía si el tipo era entero, así que dejaba los float32 sin escalar. |
| **Área por píxel** | Se **deriva** de la transformación afín del GeoTIFF: 20 × 20 m = **0.04 ha**. La primera versión asumía 0.01 ha (10 m) y subestimaba todas las superficies en un factor de 4. |
| **CRS y resolución** | Todo en **EPSG:32615** (UTM 15N) a **20 m**, la resolución nativa de las bandas *red-edge* y SWIR. Deja los datos listos para la cuadrícula espacial de la Parte 2 sin reproyectar. |
| **Separación agua/tierra** | Los GeoJSON son **rectángulos**, no contornos de lago. La separación efectiva la hace la **máscara WBI**. Áreas detectadas: Atitlán 122.1 km², Amatitlán 14.5 km². |
| **Verificación del algoritmo** | La implementación local se compara **píxel a píxel** contra el evalscript literal de CyanoLakes: diferencia máxima 0.0. |
| **Mapas interactivos** | Se **reproyecta** el ráster de EPSG:32615 a EPSG:4326 antes de superponerlo en Folium, y los píxeles que no son agua quedan transparentes. |
| **Fechas críticas** | **Calculadas** (media + 1 desviación estándar por lago), no escritas a mano. |

## Limitaciones importantes

- **Sin máscara de nubes por píxel.** La colección `SENTINEL2_L1C` de openEO **no expone**
  CLM, CLP, dataMask, SCL ni QA60. No se fabricó ninguna máscara sustituta. El control
  disponible es la selección oficial de escenas con baja nubosidad más los filtros de
  validez espectral.
- **Sin validación in situ.** Los valores son estimaciones satelitales, no mediciones de
  laboratorio; el modelo predictivo es un apoyo de cribado, no un diagnóstico.
- **Incertidumbre del algoritmo.** MAPE 42.3 % y RMSE relativo 95.8 %, calibrado para
  *Microcystis aeruginosa* sobre datos simulados, con dominio 1–60 µg/L. En Atitlán el
  45.08 % de los píxeles queda por debajo de ese dominio y 16.68 % da clorofila negativa.
- **Nivel L1C**: reflectancia de tope de atmósfera, sin corrección atmosférica.
- **Desbalance severo** (1:61) y fuerte concentración espacial/temporal de los positivos.
- **Inestabilidad temporal**: PR-AUC medio 0.7416 ± 0.2510 en validación por ventana
  expansiva, frente a 0.9758 ± 0.0128 en validación espacial.
- **La transferencia entre lagos no es satisfactoria** con umbral fijo: Recall operacional
  de 4–13 % al entrenar en un lago y evaluar en el otro (prevalencia ~227× distinta).

---

## Archivos excluidos del control de versiones

Se excluyen por tamaño; todos se regeneran con los comandos anteriores:

- `outputs/rasters/` — 22 GeoTIFF de entrada, ~244 MB
- `outputs/parte2/data/pixels/` — Parquet particionado del dataset, ~80–130 MB
- `outputs/parte2/maps/probability/**/*.tif` — 22 GeoTIFF de probabilidad
- `outputs/parte2/maps/errors_oof/pixels/` — Parquet con predicciones out-of-fold por píxel
- `outputs/parte2/**/*.joblib` — modelos serializados
- `outputs/parte2/smoke/` — todo el perfil de prueba
- `outputs/logs/`, `outputs/parte2/logs/`, cachés y temporales

**Sí** se versionan: código, cuadernos, informes (`.md`), el artículo técnico, manifiestos,
tablas CSV/JSON pequeñas, figuras PNG finales y reportes. **No existe ningún PDF de la
Parte 2** por restricción explícita del laboratorio.
