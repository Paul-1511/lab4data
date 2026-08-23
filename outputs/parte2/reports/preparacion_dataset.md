# Preparacion del dataset de Machine Learning — Laboratorio 4, Parte 2

Generado: 2026-08-23T16:22:39  
Fuente: 22 GeoTIFF Sentinel-2 L1C reales (SENTINEL2_L1C), EPSG:32615, 20 m.  
Semilla fija: 42

> **Origen de los datos.** Todas las cifras provienen de imagenes Sentinel-2 descargadas de Copernicus Data Space. No se utilizo ningun dato sintetico en ninguna etapa.

---

## 1. Construccion y limpieza

- Observaciones validas de agua: **3,756,510**
  - Amatitlan: 399,646 (10.6 %)
  - Atitlan: 3,356,864 (89.4 %)

### Descartes por criterio (suma sobre los 22 rasteres)

| Criterio | Pixeles descartados |
|---|---|
| NoData / -9999 / fuera de datos validos | 340,929 |
| NaN o infinitos | 0 |
| Reflectancia TOA <= 0 (no fisica) | 1,220 |
| No es agua segun la mascara WBI | 12,361,361 |
| Indice espectral no finito | 0 |
| Filas duplicadas (lake,date,row,col) | 0 |

Pixeles brutos totales: 16,423,470

Diagnostico (NO se descartan): 14,662 pixeles conservan alguna banda con reflectancia > 1, compatible con nubes o reflexion especular. No se impuso ningun limite superior arbitrario.

### Escala radiometrica

- Deteccion automatica: numeros digitales (mediana |v|=2318.0); se aplica escala 0.0001 declarada por la coleccion (offset 0)
- La coleccion `SENTINEL2_L1C` declara `scale = 0.0001, offset = 0`. Los GeoTIFF conservan numeros digitales, por lo que se multiplican por 0.0001 una unica vez.
- `mainlab4.normalize_band()` **no** se utiliza aqui: solo divide cuando el dtype es entero, y estos rasteres son float32, por lo que habria dejado los DN sin escalar.

### Limitaciones documentadas

- **Separacion agua/tierra:** los GeoJSON disponibles son rectangulos identicos al bounding box oficial, no contornos reales de lago. La unica separacion efectiva es la mascara **WBI** de `mainlab4.py`. El area detectada es coherente con la realidad (Atitlan ~123 km2, Amatitlan ~15 km2), lo que respalda la mascara.
- **Nubes:** `SENTINEL2_L1C` no expone CLM, CLP, dataMask, SCL ni QA60. No se fabrico ninguna mascara sustituta. El control disponible es la seleccion oficial de fechas con baja nubosidad sobre el lago mas los filtros de validez espectral aplicados aqui.
- **Nivel L1C:** son reflectancias de tope de atmosfera, sin correccion atmosferica. El algoritmo NDCI fue disenado para L1C, pero esto anade incertidumbre a la magnitud absoluta de la clorofila.

---

## 2. Variable respuesta

### 2.1 Distribucion real de la clorofila-a

- Rango observado: -291.11 a 135.86 ug/L
- Mediana: 1.31 ug/L | p99: 11.31 ug/L
- Valores negativos: 560,214 (14.91 %)
- Fuera del dominio de calibracion [1.0, 60.0] ug/L: 1,513,792 (40.30 %)

Los valores negativos carecen de sentido fisico: proceden de evaluar el polinomio NDCI->clorofila con NDCI negativo, fuera de su dominio. **No se recortaron ni transformaron**, porque hacerlo alteraria artificialmente el balance de clases; se reportan y se marcan con `fuera_calibracion`.

### 2.2 Verificacion de la bibliografia citada en `lab4-2.ipynb`

| Referencia citada | Verificacion |
|---|---|
| OECD (1982), *Eutrophication of Waters*, DOI 10.1787/9789264077980-en | **Existe y es pertinente.** Sistema de fronteras fijas por clorofila-a media: ultraoligotrofico <1, oligotrofico <2.5, mesotrofico 2.5-8, **eutrofico 8-25**, **hipertrofico >25** ug/L. |
| WHO (2021), *Guidelines on recreational water quality, Vol. 1* | **Existe y es pertinente.** Con dominancia de cianobacterias: nivel de vigilancia 1-12 ug/L de clorofila-a; **Alerta 1: 12-24 ug/L**. La Alerta 2 se define por natas y transparencia, no por un valor de clorofila. |
| Mishra, S. et al. (2019), *Applicability of Sentinel-2...*, RSE 232, 111354, DOI 10.1016/j.rse.2019.111354 | **REFERENCIA INCORRECTA.** Ese DOI corresponde a Hurskainen, Adhikari, Siljander, Pellikka y Hemp (2019), *Auxiliary datasets improve accuracy of object-based land use/land cover classification in heterogeneous savanna landscapes*, RSE **233**, 111354: un articulo de cobertura del suelo en sabana, sin relacion con clorofila ni calidad de agua. |

**Sustitucion propuesta.** La referencia correcta para el indice empleado es Mishra, S. & Mishra, D. R. (2012), *Normalized difference chlorophyll index: a novel model for remote estimation of chlorophyll-a concentration in turbid productive waters*, Remote Sensing of Environment **117**, 394-406 (DOI 10.1016/j.rse.2011.10.016), que introduce el NDCI y calibra el modelo cuadratico en un rango de **1-60 mg/m3**.

El polinomio implementado en `mainlab4.chl_from_ndci` (`826.57*NDCI^3 - 176.43*NDCI^2 + 19*NDCI + 4.071`) proviene del script *Cyanobacteria Chlorophyll-a NDCI L1C* de Sentinel Hub, cuya documentacion indica calibracion para *Microcystis aeruginosa*, entrenamiento con clorofila < 500 ug/L y errores de **MAPE 42.3 %** y **RMSE relativo 95.8 %**. Esa incertidumbre debe acompanar cualquier conclusion.

### 2.3 Analisis de los umbrales candidatos

| Umbral | Respaldo bibliografico | % clase alta global | % alta Amatitlan | % alta Atitlan |
|---|---|---|---|---|
| 20 ug/L | Dentro de la banda eutrofica OECD (8-25) y del rango de Alerta 1 de la WHO. Es el umbral usado en la Parte 1, pero no es una frontera publicada. | 0.262 % | 2.444 % | 0.002 % |
| 25 ug/L | **Frontera eutrofico -> hipertrofico de OECD (1982)**, que coincide con el techo de la Alerta 1 de la WHO (24 ug/L). Dos fuentes independientes convergen aqui. | 0.138 % | 1.296 % | 0.000 % |
| 50 ug/L | **Sin respaldo en las fuentes citadas.** No aparece en OECD 1982 ni como valor de clorofila en WHO 2021. En `lab4-2.ipynb` se justifico como 'criterio mas conservador', lo que es una eleccion arbitraria. | 0.016 % | 0.151 % | 0.000 % |

### 2.4 Umbral recomendado: **25 ug/L**

Se recomienda como respuesta principal `high_cyano_25` porque es la unica de las tres candidatas que corresponde a una **frontera publicada**: separa el estado eutrofico del hipertrofico en OECD (1982) y coincide practicamente con el limite superior de la Alerta 1 de la WHO (24 ug/L). Ambientalmente marca el punto en que la biomasa algal deja de ser alta para pasar a ser caracteristica de un sistema degradado.

**Analisis de sensibilidad recomendado:** repetir el modelado con `high_cyano_20` (continuidad con la Parte 1 y con el rango de vigilancia de la WHO). Se descarta 50 ug/L como respuesta principal por carecer de respaldo bibliografico en las fuentes citadas.

La eleccion **no se hizo por balance de clases**: como muestra la tabla anterior, los tres umbrales producen un desbalance severo, y el recomendado no es el que mas favorece el entrenamiento.

### 2.5 Advertencia critica sobre el desbalance

Con el umbral recomendado la clase alta representa el **0.138 %** del dataset (5,195 de 3,756,510 observaciones), es decir una razon de desbalance de **1:722**.


Consecuencias para el modelado: *accuracy* sera enganosa; deben reportarse *recall*, *precision*, F1 y sobre todo **PR-AUC**, y considerar pesos de clase o remuestreo en el entrenamiento, nunca modificando la clorofila.

---

## 3. Variables predictoras

Cadena de construccion de la respuesta:

```
B04, B05  ->  NDCI  ->  chlorophyll  ->  high_cyano_*
```

### 3.1 Conjunto predictor principal (sin fuga)

| Variable | Tipo | Justificacion |
|---|---|---|
| `B02` | Banda espectral | Azul (492 nm); sensible a dispersion y turbidez. |
| `B03` | Banda espectral | Verde (560 nm); maximo de reflectancia de la biomasa algal, no interviene en el NDCI. |
| `B07` | Banda espectral | Borde rojo (783 nm); responde a biomasa sin usar B04/B05. |
| `B08` | Banda espectral | NIR (833 nm); separa agua de vegetacion y detecta natas. |
| `B8A` | Banda espectral | NIR estrecho (865 nm); complementa a B08. |
| `B11` | Banda espectral | SWIR1 (1610 nm); distingue agua de suelo y nube. |
| `B12` | Banda espectral | SWIR2 (2190 nm); refuerza la discriminacion de agua. |
| `NDWI` | Indice | (B03-B08)/(B03+B08); no usa B04 ni B05, por lo que es independiente de la cadena de la respuesta. |

### 3.2 Variables excluidas del modelo principal

| Variable | Motivo de exclusion |
|---|---|
| `B04` | insumo directo de NDCI, que genera la clorofila y la respuesta |
| `B05` | insumo directo de NDCI, que genera la clorofila y la respuesta |
| `NDCI` | indice del que se deriva la clorofila y por tanto la respuesta |
| `chlorophyll` | variable de la que se deriva directamente la respuesta |
| `FAI` | utiliza B04; comparte insumo con NDCI (fuga indirecta) |
| `NDVI` | utiliza B04; se conserva porque el enunciado lo exige, pero solo es admisible en un analisis de sensibilidad etiquetado como fuga indirecta |
| `high_cyano_20` | es la propia respuesta candidata |
| `high_cyano_25` | es la propia respuesta candidata |
| `high_cyano_50` | es la propia respuesta candidata |
| `water_mask` | es un filtro de construccion del dataset, no un predictor |
| `valid_data` | es un filtro de construccion del dataset, no un predictor |

**NDVI** se conserva en el dataset porque el enunciado lo exige explicitamente, pero **no entra en el modelo principal**: usa B04, el mismo canal que alimenta el NDCI, por lo que aporta fuga indirecta. Queda preparado como analisis de sensibilidad etiquetado.

### 3.3 Columnas de trazabilidad (no predictoras)

| Columna | Uso previsto |
|---|---|
| `lake` | agrupacion, validacion entre lagos y estratificacion |
| `date` | validacion temporal y estratificacion |
| `row` | reconstruccion espacial del raster |
| `col` | reconstruccion espacial del raster |
| `x_utm` | bloques espaciales de 1x1 km en EPSG:32615 |
| `y_utm` | bloques espaciales de 1x1 km en EPSG:32615 |
| `longitude` | mapas y trazabilidad geografica |
| `latitude` | mapas y trazabilidad geografica |

No se usan como predictoras para evitar que el modelo memorice la geografia en lugar de aprender una senal espectral generalizable.

---

## 4. Artefactos generados

- `outputs/parte2/data/pixels/` — Parquet particionado por lago y fecha (3,756,510 filas)
- `outputs/parte2/data/eda_sample.parquet` — muestra determinista (200,000 filas, semilla 42)
- `outputs/parte2/data/dataset_schema.csv`, `dataset_manifest.csv`, `limpieza_diagnostico.csv`
- `outputs/parte2/target/` — distribuciones global, por lago y por fecha
- `outputs/parte2/eda/` — figuras
