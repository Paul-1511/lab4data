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
├── mainlab4.py                  Funciones base: índices, máscara WBI, clorofila, openEO
├── lab4_analisis.py             Utilidades de visualización de la primera versión
├── descargar_rasters.py         Adquisición y validación de los 22 GeoTIFF (openEO)
├── regenerar_parte1_real.py     Pipeline de la Parte 1 con datos reales (act. 1-8)
├── preparar_dataset_ml.py       Parte 2: dataset por píxel (ejercicios 1-3)
├── md2pdf.py                    Conversión del informe Markdown a PDF
├── lab4.ipynb                   Cuaderno de la Parte 1 (flujo real)
├── lab4-2.ipynb                 Cuaderno de la Parte 2
├── INFORME_LAB4.md / .pdf       Informe para público no técnico
├── AUDITORIA_PARTE2.md          Auditoría previa a la Parte 2
├── Lago_Atitlan.geojson         Rectángulo del área de estudio (NO es el contorno real)
├── Lago_Amatitlan.geojson       Rectángulo del área de estudio (NO es el contorno real)
├── requirements.txt
└── outputs/
    ├── rasters/                 22 GeoTIFF reales (NO versionados, ~244 MB)
    ├── manifests/               Manifiesto y reporte de validación de los rásteres
    ├── parte1_real/             RESULTADOS VÁLIDOS de la Parte 1
    │   ├── tables/   figures/   maps/   reports/   validation/
    ├── parte2/                  Dataset de Machine Learning (Parte 2)
    └── *_demo.*, act5_*, act6_*, act8_*   OBSOLETOS (versión simulada)
```

### Resultados reales frente a resultados demostrativos

| Ubicación | Naturaleza | ¿Usar? |
|---|---|---|
| `outputs/parte1_real/` | 22 escenas Sentinel-2 reales | **Sí** |
| `outputs/manifests/` | Metadatos reales de los rásteres | **Sí** |
| `outputs/parte2/` | Dataset derivado de los rásteres reales | **Sí** |
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

### 5. Cuaderno

`lab4.ipynb` usa **rutas relativas** y debe ejecutarse desde la raíz del repositorio.
Invoca el pipeline real y muestra resultados resumidos; no contiene credenciales ni
resultados sintéticos presentados como reales.

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
  laboratorio.
- **Incertidumbre del algoritmo.** MAPE 42.3 % y RMSE relativo 95.8 %, calibrado para
  *Microcystis aeruginosa* sobre datos simulados, con dominio 1–60 µg/L. En Atitlán el
  45 % de los píxeles queda por debajo de ese dominio.
- **Nivel L1C**: reflectancia de tope de atmósfera, sin corrección atmosférica.

---

## Archivos excluidos del control de versiones

Se excluyen por tamaño; todos se regeneran con los comandos anteriores:

- `outputs/rasters/` — 22 GeoTIFF, ~244 MB
- `outputs/parte2/data/pixels/` — Parquet particionado, ~80–130 MB
- `outputs/logs/`, `outputs/parte2/logs/`

**Sí** se versionan: código, cuadernos, informe y PDF, manifiestos, tablas CSV, figuras y
reportes.
