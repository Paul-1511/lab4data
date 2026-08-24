# Del píxel satelital al modelo predictivo: un recorrido completo por la detección de cianobacteria con teledetección y aprendizaje automático

## Resumen

Este artículo explica, paso a paso y desde cero, cómo se construyó un sistema de detección de
floraciones de cianobacteria en dos lagos de Guatemala (Atitlán y Amatitlán) usando imágenes del
satélite Sentinel-2 y tres modelos de aprendizaje automático (Regresión Logística, Random Forest
y XGBoost). No es un resumen de resultados —eso está en `INFORME_LAB4_PARTE2.md`— sino una guía
pedagógica pensada para que una estudiante que domina Python pero todavía está consolidando
conceptos de validación de modelos, series de tiempo y aprendizaje automático pueda entender
*por qué* se tomó cada decisión y pueda reproducir el proyecto completo. Se usan, en todo
momento, cifras reales extraídas de los artefactos generados por el propio proyecto: 3,756,510
observaciones de píxel, 22 escenas Sentinel-2, un Random Forest con PR-AUC 0.9758 ± 0.0128 en
validación espacial y 0.7416 ± 0.2510 en validación temporal, entre otras.

## Palabras clave

Teledetección, Sentinel-2, cianobacteria, clorofila-a, aprendizaje automático, Random Forest,
XGBoost, validación espacial, validación temporal, SHAP, desbalance de clases, PR-AUC,
autocorrelación espacial, *domain shift*.

---

## 1. El problema ambiental y la motivación

Las cianobacterias son microorganismos que viven de forma natural en lagos y ríos. Cuando reciben
demasiados nutrientes —por aguas residuales sin tratar, fertilizantes agrícolas o escorrentía
urbana— y encuentran temperatura y luz favorables, se multiplican de forma explosiva. Esto se
llama **floración algal** y puede reducir el oxígeno del agua, matar peces, producir toxinas
peligrosas y afectar el turismo y el abastecimiento de agua.

Medir esto con lanchas y análisis de laboratorio es caro y cubre solo unos pocos puntos del lago.
Los satélites, en cambio, fotografían el lago entero cada pocos días. La pregunta que motiva este
proyecto es: **¿podemos usar esas fotografías para estimar dónde hay más probabilidad de
floración, sin tener que ir físicamente a medir cada punto?**

> **Idea intuitiva.** Piensa en el satélite como una cámara que, en vez de ver solo rojo-verde-
> azul como tu celular, ve nueve "colores" distintos (bandas), algunos invisibles al ojo humano.
> La clorofila —el pigmento verde de las algas— refleja la luz de forma característica en
> ciertos colores, así que comparando esos colores se puede *estimar* cuánta clorofila hay.

## 2. La pregunta de investigación

Concretamente: **dado un píxel de 20×20 metros de la superficie de un lago, fotografiado por
Sentinel-2 en una fecha conocida, ¿podemos predecir si su concentración de clorofila-a supera
8 µg/L (el umbral que marca el inicio de condición eutrófica) usando únicamente ocho variables
espectrales que no participaron en calcular esa misma clorofila?**

Y una pregunta secundaria, igual de importante: **¿ese modelo generaliza a zonas nuevas del
mismo lago, a fechas futuras, y a otro lago distinto?** Como verás, la respuesta a cada una de
esas tres preguntas es diferente, y esa diferencia es el hallazgo central del proyecto.

## 3. Arquitectura general del pipeline

El proyecto está dividido en scripts que corresponden a fases sucesivas:

```
descargar_rasters.py          -> descarga las 22 escenas Sentinel-2 reales (openEO)
regenerar_parte1_real.py      -> calcula índices, series temporales, mapas de la Parte 1
preparar_dataset_ml.py        -> construye el dataset por píxel (3.75 millones de filas)
modelos_parte2.py             -> entrena y valida los tres modelos (Ejercicios 4-7)
explicabilidad_mapas_parte2.py -> interpretabilidad, mapas y conclusiones (Ejercicios 8-10)
```

Cada script tiene modos `--dry-run` (estima costo sin ejecutar), `--build`/`--all` (ejecuta de
verdad) y `--validate` (audita que los resultados sean correctos). Este patrón —planificar,
ejecutar, validar— se repite en todo el proyecto porque procesar 3.75 millones de filas y
entrenar tres modelos toma tiempo real (la corrida `standard` de `modelos_parte2.py` tomó
**80.2 minutos**; la de `explicabilidad_mapas_parte2.py`, **43.4 minutos**), y no queremos
descubrir un error después de esperar una hora.

> **Cómo interpretarlo.** Que el pipeline esté dividido en fases con checkpoints no es solo
> organización: significa que si algo falla a la mitad, no hay que repetir todo desde cero. Cada
> fase guarda un archivo de "checkpoint" con un hash de su configuración; si vuelves a ejecutar
> el mismo comando, el script detecta que ya existe un resultado válido y lo reutiliza.

## 4. Sentinel-2 y el significado de cada banda usada

Sentinel-2 es un satélite del programa europeo Copernicus que fotografía la Tierra en 13 bandas
espectrales, cada una sensible a un rango distinto de longitud de onda. Para este proyecto se
usan 9:

| Banda | Longitud de onda aprox. | Qué "ve" | Resolución nativa |
|---|---|---|---|
| B02 | 492 nm (azul) | Dispersión por partículas, turbidez | 10 m |
| B03 | 560 nm (verde) | Reflectancia de biomasa vegetal/algal | 10 m |
| B04 | 665 nm (rojo) | Absorción por clorofila (fuerte) | 10 m |
| B05 | 705 nm (borde rojo 1) | Transición rojo→infrarrojo, sensible a clorofila | 20 m |
| B07 | 783 nm (borde rojo 3) | Biomasa vegetal, sin la fuerte absorción de B04 | 20 m |
| B08 | 833 nm (infrarrojo cercano) | Separa agua de vegetación; natas flotantes | 10 m |
| B8A | 865 nm (NIR estrecho) | Similar a B08, menos afectado por vapor de agua | 20 m |
| B11 | 1610 nm (SWIR1) | El agua absorbe casi toda la radiación aquí | 20 m |
| B12 | 2190 nm (SWIR2) | Refuerza la distinción agua/suelo/nube | 20 m |

**B04 y B05 se usan para calcular el índice NDCI (ver sección 13), que a su vez calcula la
clorofila-a —la variable que queremos predecir—. Por eso ambas bandas están prohibidas como
predictoras** (sección 18): usarlas sería "hacer trampa", porque el modelo tendría acceso directo
a la fórmula de la respuesta.

## 5. De números digitales (DN) a reflectancia: por qué `× 0.0001`

Cuando Sentinel-2 mide la luz reflejada, no guarda directamente un número entre 0 y 1 (la
fracción de luz reflejada, llamada **reflectancia**). Por eficiencia de almacenamiento, guarda un
**número digital (DN)**, un entero, junto con un factor de escala. La colección
`SENTINEL2_L1C` de Copernicus Data Space declara oficialmente:

```
scale = 0.0001, offset = 0
```

Esto significa: `reflectancia = DN × 0.0001`. Un DN de 719, por ejemplo, corresponde a una
reflectancia de 0.0719 (7.19 % de la luz incidente se refleja).

> **Error frecuente.** En una versión anterior de este proyecto, la función que debía aplicar
> esta escala (`normalize_band()`) solo dividía cuando el tipo de dato era entero. Pero los
> GeoTIFF descargados se guardaban como `float32` conservando los DN (por ejemplo, `719.0` en vez
> de `719`), así que la función los dejaba **sin escalar**. El error se detectó verificando
> estadísticamente el rango resultante: una reflectancia "normal" debe estar entre 0 y ~1.5, y
> los valores sin escalar rondaban miles. Se corrigió detectando la magnitud típica de los datos
> (si la mediana supera 2.0, son DN) en vez de confiar solo en el tipo de dato declarado.

## 6. Resoluciones espaciales y la decisión de trabajar a 20 m

No todas las bandas de Sentinel-2 tienen la misma resolución nativa: B02, B03, B04 y B08 vienen a
10 m; B05, B07, B8A, B11 y B12 vienen a 20 m. Si se mezclaran directamente, unas bandas tendrían
más detalle que otras sin que eso significara nada real.

**Se eligió trabajar todo a 20 m** (la resolución más gruesa de las bandas necesarias), en vez de
10 m. La razón es honesta: procesar a 10 m atribuiría un detalle inexistente a las bandas de
red-edge y SWIR, que en realidad solo tienen 20 m de resolución real. Trabajar a 20 m es una
pérdida de detalle en las bandas de 10 m, pero es la decisión que no inventa información.

## 7. CRS EPSG:32615, transformación afín y cálculo de área

Un mapa necesita un **sistema de referencia de coordenadas (CRS)** para decir "este píxel está
exactamente aquí en la superficie terrestre". Se usó **EPSG:32615** (WGS 84 / UTM zona 15N), que
mide distancias en **metros** —a diferencia de las coordenadas geográficas (latitud/longitud), que
miden en grados y no son directamente comparables a distancias—.

Cada GeoTIFF trae una **transformación afín**: seis números que dicen cómo convertir un índice de
fila/columna del arreglo de píxeles a una coordenada real en metros. Concretamente:

```
x_utm = c + a·col
y_utm = f + e·fila
```

donde `a` y `e` son el tamaño del píxel (20 y −20 metros, el signo negativo porque las filas
crecen hacia el sur mientras las coordenadas y crecen hacia el norte) y `c`, `f` son el origen.

**El área de un píxel se deriva de esta transformación, no se asume:**

$$\\text{área}_{ha} = \\frac{|a \\times e|}{10{,}000}$$

Con `a = 20` y `e = -20`: área = `400 / 10,000 = 0.04 ha` por píxel. Este valor (**0.04 ha**) se
usa en todo el proyecto para convertir conteos de píxeles a hectáreas: los 13,386 falsos
negativos de la evaluación out-of-fold (sección 35) representan `13,386 × 0.04 = 535.4 hectáreas`
de floración que el modelo no detectó.

> **Error frecuente.** Una versión anterior asumía `AREA_PIXEL_HA = 0.01` (asumiendo 10 m de
> resolución) mientras el resto del pipeline ya trabajaba a 20 m. Eso subestimaba todas las
> superficies calculadas en un factor de 4. Se corrigió derivando el área directamente de la
> transformación afín real de cada GeoTIFF, en vez de fijar una constante.

## 8. La máscara de agua (WBI)

Antes de calcular cualquier índice de clorofila hay que saber qué píxeles son agua y cuáles son
tierra, porque la vegetación terrestre también contiene clorofila y contaminaría el análisis. El
proyecto usa un **índice compuesto de detección de agua (WBI)** que combina varios criterios
espectrales — MNDWI, NDWI, AWEI y NDVI — más un filtro adicional que descarta suelos desnudos y
superficies urbanas que a veces se confunden espectralmente con agua.

La validez de esta máscara se comprobó comparando el área de agua detectada contra la superficie
real conocida de cada lago: **123.2 km² para Atitlán** (superficie real ≈ 130 km²) y
**14.6 km² para Amatitlán** (superficie real ≈ 15 km²). La coincidencia es una señal fuerte de
que el método funciona correctamente.

## 9. Índices espectrales utilizados, con fórmulas

$$\\text{NDVI} = \\frac{B08 - B04}{B08 + B04} \\qquad \\text{NDWI} = \\frac{B03 - B08}{B03 + B08}$$

**NDVI** (Índice de Vegetación de Diferencia Normalizada) contrasta el infrarrojo cercano (que la
vegetación refleja mucho) con el rojo (que la clorofila absorbe). Aquí ayuda a detectar
acumulaciones de algas flotantes, pero **usa B04**, por lo que está prohibido como predictor
(sección 18).

**NDWI** (Índice de Agua de Diferencia Normalizada) contrasta el verde con el infrarrojo cercano.
Caracteriza el estado de la lámina de agua y **no usa B04 ni B05**, así que sí es un predictor
válido.

## 10. NDCI y la transformación CyanoLakes a clorofila-a

$$\\text{NDCI} = \\frac{B05 - B04}{B05 + B04}$$

El NDCI (Índice de Clorofila de Diferencia Normalizada, Mishra & Mishra 2012) explota el
contraste entre el rojo (B04, donde la clorofila absorbe fuerte) y el borde rojo (B05, donde
absorbe menos). El script *Cyanobacteria Chlorophyll-a NDCI L1C* de Sentinel Hub (CyanoLakes)
convierte el NDCI en una estimación de clorofila-a mediante un polinomio cúbico:

$$\\text{Chl-a} = 826.57 \\cdot \\text{NDCI}^3 - 176.43 \\cdot \\text{NDCI}^2 + 19 \\cdot \\text{NDCI} + 4.071$$

Esta fórmula, junto con la máscara de agua, es exactamente el algoritmo usado en este proyecto —
verificado píxel a píxel contra la implementación literal del evalscript en la Parte 1, con
diferencia numérica de 0.0.

## 11. Dominio de calibración e incertidumbre

El NDCI fue calibrado por Mishra & Mishra (2012) sobre datos simulados en un rango de
**1 a 60 mg/m³** (equivalente a µg/L). Fuera de ese rango, la estimación es una extrapolación, no
una medición confiable. Además, la documentación del script CyanoLakes reporta un **MAPE
(error porcentual medio absoluto) de 42.3 %** y un **RMSE relativo de 95.8 %** — errores grandes,
que hay que llevar presentes en cualquier interpretación de los resultados.

Al examinar el dataset real se encontró algo importante: en **Atitlán, el 45.08 % de los píxeles
de agua caen fuera del dominio de calibración** y **16.68 % dan clorofila estimada negativa** (sin
sentido físico). En Amatitlán, en cambio, solo 0.17 % y 0.04 % respectivamente. Esto **no se
corrigió forzando esos valores a cero**: se conservaron tal cual para trazabilidad, marcados con
una columna `fuera_calibracion`, porque alterar el dato original solo para "verse mejor" sería
falsificar la evidencia.

> **Cómo interpretarlo.** Que Atitlán tenga menos floraciones que Amatitlán es una conclusión
> robusta (la diferencia es de órdenes de magnitud), pero los valores *absolutos* de clorofila en
> Atitlán deben tomarse con mucha cautela, porque casi la mitad de sus estimaciones están fuera
> del rango en que el algoritmo fue calibrado.

## 12. Construcción del dataset por píxel

Cada fila del dataset final representa **un píxel de agua válido, en una fecha concreta**. El
proceso, por cada uno de los 22 GeoTIFF:

1. Leer las 9 bandas y convertirlas a reflectancia (sección 5).
2. Calcular NDVI, NDWI, NDCI, FAI y clorofila-a.
3. Aplicar la máscara WBI (sección 8) y descartar píxeles inválidos (NoData, no finitos,
   reflectancia ≤ 0 —físicamente imposible—, duplicados).
4. Derivar las coordenadas de cada píxel desde la transformación afín real (sección 7).
5. Añadir variables auxiliares: `year`, `month`, `season` (estacional, ver sección 28) y
   `spatial_block_1km` (el bloque espacial al que pertenece, ver sección 26).
6. Guardar en formato Parquet, particionado por lago y fecha.

El resultado son **3,756,510 observaciones**: 399,646 de Amatitlán y 3,356,864 de Atitlán (un
lago mucho más grande). No cargar los 3.75 millones de filas de golpe en memoria fue una decisión
deliberada: se usa **proyección de columnas** de PyArrow, leyendo solo las 8 columnas de
predictores más la respuesta y los identificadores de agrupación, en vez de las 33 columnas
totales del dataset.

## 13. Selección del umbral de 8 µg/L y análisis 20/25/50

La OECD (1982) clasifica los lagos por su clorofila-a media: oligotrófico < 2.5, mesotrófico
2.5–8, eutrófico 8–25, hipertrófico > 25 µg/L. Se eligió **8 µg/L** como umbral de "alta
presencia" porque marca **la transición a condición eutrófica**: es el punto en que un lago deja
de ser equilibrado y empieza a mostrar exceso de nutrientes.

Se analizaron también 20, 25 y 50 µg/L como sensibilidad, pero cada uno tiene un problema
práctico distinto:

| Umbral | Positivos en Atitlán | Problema |
|---|---|---|
| 8 µg/L | 2,150 | Ninguno: es el único con suficientes positivos en ambos lagos |
| 20 µg/L | 64 | Insuficiente para entrenar en Atitlán |
| 25 µg/L | 14 | Insuficiente incluso para evaluar de forma confiable |
| 50 µg/L | 4 | Prácticamente inexistente |

**El criterio de selección fue ambiental, no estadístico**: se buscó la frontera trófica
correcta, y resultó que esa frontera también es la que deja suficientes casos para modelar. Si se
hubiera elegido el umbral *porque* deja más casos, sería un razonamiento circular (ajustar la
pregunta para que la respuesta salga bien); aquí ocurrió al revés.

## 14. Desbalance de clases

Con el umbral de 8 µg/L, solo **60,146 de 3,756,510 observaciones (1.6011 %)** son positivas. Esto
se llama **desbalance de clases**: una clase (positiva) es mucho más rara que la otra (negativa).

> **Idea intuitiva.** Imagina que tienes que adivinar si mañana lloverá en el desierto de Atacama,
> donde llueve en promedio 1 día de cada 60. Si simplemente dices "no lloverá" todos los días,
> aciertas el 98.3 % de las veces — pero tu predicción es inútil, porque nunca avisas el único
> día que sí llueve. Eso es exactamente lo que pasaría si midiéramos nuestro modelo solo con
> **Accuracy** (porcentaje de aciertos): un modelo que dijera "nunca hay floración" tendría
> 98.4 % de Accuracy y sería completamente inútil.

Por eso este proyecto usa **PR-AUC** (ver sección 20) en vez de Accuracy como métrica principal, y
técnicas como `class_weight="balanced"` (Regresión Logística, Random Forest) o
`scale_pos_weight` (XGBoost, calculado únicamente con el conjunto de entrenamiento) para que el
modelo no ignore la clase minoritaria durante el ajuste.

## 15. Fuga de información: por qué se excluyeron ciertas variables

La respuesta se calcula así: `B04, B05 → NDCI → clorofila-a → high_cyano_8`. Cualquier variable
que participe en esa cadena, directa o indirectamente, filtraría información de la respuesta
hacia los predictores — un error llamado **fuga de información** (*data leakage*).

**Excluidas:** `B04`, `B05` (insumos directos del NDCI), `NDCI`, `clorofila-a` (la respuesta
misma), `FAI` y `NDVI` (ambos usan B04, fuga indirecta), y las variables de agrupación (`lake`,
`date`, coordenadas, `spatial_block_1km`) porque permitirían "memorizar" la geografía o la fecha
en vez de aprender la señal espectral real.

**Predictores permitidos:** `B02, B03, B07, B08, B8A, B11, B12, NDWI` — ocho variables que no
comparten ningún insumo con la cadena de la respuesta.

> **Error frecuente.** Es tentador incluir NDVI porque "es un índice espectral útil". Pero como
> comparte B04 con el NDCI, un modelo con NDVI como predictor podría parecer mejor sin realmente
> aportar señal ecológica nueva — solo estaría reconstruyendo parte de la fórmula de la respuesta.
> El código incluye *asserts* que detienen la ejecución si una variable prohibida entra al
> conjunto de entrenamiento, precisamente para blindarse contra este error.

## 16. Los tres modelos, explicados intuitivamente

**Regresión Logística.** Traza una "línea" (en 8 dimensiones) que separa lo mejor posible los
píxeles positivos de los negativos, y calcula la probabilidad según qué tan lejos está un punto
de esa línea. Es el modelo más simple e interpretable: cada variable tiene un coeficiente que
indica su peso.

**Random Forest.** Construye cientos de "árboles de decisión" (cada uno hace preguntas tipo "¿B07
es mayor que 0.05?") sobre subconjuntos aleatorios de datos y variables, y promedia sus votos.
Es más flexible que la Regresión Logística porque puede capturar relaciones no lineales e
interacciones entre variables sin que se lo digamos explícitamente.

**XGBoost.** También construye árboles, pero de forma secuencial: cada árbol nuevo se enfoca en
corregir los errores de los árboles anteriores (*gradient boosting*). Suele ser muy preciso, pero
también más propenso a ajustarse demasiado a los datos de entrenamiento si no se regula bien.

> **En un proyecto real.** No existe un modelo "mejor" en abstracto: el mejor depende de la
> pregunta. En este proyecto, XGBoost ganó en la división aleatoria (PR-AUC 0.9826) pero Random
> Forest resultó prácticamente empatado con XGBoost bajo validación espacial (0.9758 vs. 0.9757,
> diferencia de solo 0.0001) y fue elegido para las secciones siguientes por ser la validación más
> honesta.

## 17. Hiperparámetros y su efecto

Los **hiperparámetros** son las "perillas" que se ajustan antes de entrenar (a diferencia de los
parámetros del modelo, que se aprenden durante el entrenamiento). Se buscaron con
`RandomizedSearchCV` (12 configuraciones aleatorias × 3 validaciones cruzadas), optimizando
PR-AUC, **usando solo una muestra del entrenamiento**, nunca el conjunto de prueba.

Los seleccionados:

- **Regresión Logística:** `C=0.01` (regularización fuerte: penaliza coeficientes grandes para
  evitar sobreajuste), `l1_ratio=0.0` (regularización L2 pura).
- **Random Forest:** `min_samples_leaf=5` (cada hoja del árbol necesita al menos 5 ejemplos, para
  no memorizar casos aislados), `max_features='sqrt'` (cada árbol considera solo la raíz cuadrada
  del número de variables en cada división, para diversificar los árboles), `max_depth=None`
  (los árboles pueden crecer sin límite de profundidad).
- **XGBoost:** `max_depth=10`, `learning_rate=0.03` (aprendizaje lento y cuidadoso),
  `subsample=0.6`, `colsample_bytree=0.8`, `min_child_weight=10`.

> **Nota de reproducibilidad.** `penalty` de scikit-learn (para elegir L1/L2 en la Regresión
> Logística) fue deprecado en la versión 1.8 del paquete. Se migró a `l1_ratio` (0 = L2 pura,
> 1 = L1 pura) y se verificó numéricamente que produce coeficientes idénticos con la misma
> configuración: no es solo un cambio de nombre, es la misma matemática expresada con la API
> vigente.

## 18. Métricas: Accuracy, Precision, Recall, F1, ROC-AUC y PR-AUC, con ejemplos

Imagina 100 píxeles de prueba, de los cuales 10 son realmente positivos (floración alta). El
modelo predice 12 positivos, de los cuales 8 son correctos.

- **TP (verdaderos positivos)** = 8 · **FP (falsos positivos)** = 4 · **FN (falsos negativos)** = 2
  · **TN (verdaderos negativos)** = 86
- **Accuracy** = (TP+TN)/total = 94/100 = 0.94 — parece alto, pero ignora que se perdieron 2 de
  10 positivos.
- **Precision** = TP/(TP+FP) = 8/12 = 0.667 — de lo que el modelo marcó como positivo, 66.7 %
  era correcto.
- **Recall** = TP/(TP+FN) = 8/10 = 0.80 — de los 10 positivos reales, el modelo detectó 80 %.
- **F1** = media armónica de Precision y Recall = 0.727 — equilibra ambas.
- **ROC-AUC** mide qué tan bien el modelo ordena positivos por encima de negativos, usando la
  tasa de falsos positivos.
- **PR-AUC** (o *average precision*) hace lo mismo pero usando Precision en vez de la tasa de
  falsos positivos, lo que la hace mucho más informativa cuando la clase positiva es rara (como
  aquí, 1.6 %).

## 19. Por qué PR-AUC es prioritaria en este problema

Con 1.6 % de positivos, un modelo casi cualquiera obtiene un ROC-AUC alto simplemente porque hay
muchísimos verdaderos negativos "fáciles" que inflan esa métrica. **PR-AUC no se deja engañar por
eso**: si el modelo genera muchos falsos positivos, la Precision cae y PR-AUC lo refleja de
inmediato. Por eso el ajuste de hiperparámetros (sección 17) usó PR-AUC como criterio, nunca
Accuracy ni ROC-AUC.

## 20. Matriz de confusión con los resultados OOF reales

La evaluación **out-of-fold** (OOF, sección 35) sobre las 3,756,510 observaciones, con el umbral
operacional (0.9408) del Random Forest, dio esta matriz de confusión real:

| | Predicho negativo | Predicho positivo |
|---|---|---|
| **Real negativo** | TN = 3,695,567 | FP = 797 |
| **Real positivo** | FN = 13,386 | TP = 46,760 |

De aquí: Recall = 46,760/(46,760+13,386) = **0.7774**, Precision = 46,760/(46,760+797) =
**0.9832**, PR-AUC = **0.9779**. El modelo detecta el 77.7 % de las floraciones reales, y cuando
avisa, acierta el 98.3 % de las veces.

## 21. Diferencia entre split aleatorio y validación espacial

Un **split aleatorio 70/30** reparte las filas al azar entre entrenamiento y prueba. El problema:
dos píxeles vecinos —a 20 metros el uno del otro— son casi idénticos espectralmente. Si uno cae en
entrenamiento y el otro en prueba, el modelo "reconoce" al vecino en vez de generalizar realmente.
Esto se llama **autocorrelación espacial** (sección 22): valores geográficamente cercanos tienden
a parecerse.

Los resultados lo confirman: con split aleatorio, Random Forest obtiene PR-AUC 0.9814; con
validación espacial (bloques enteros separados), 0.9758 ± 0.0128. La caída es moderada, pero la
variabilidad entre folds (± 0.0128) es la huella de esa autocorrelación: algunos bloques son más
difíciles que otros.

## 22. Autocorrelación espacial

**Ley de Tobler** (primera ley de la geografía): "todo está relacionado con todo lo demás, pero
las cosas cercanas están más relacionadas que las distantes". En este proyecto, eso significa que
un píxel con floración alta probablemente tiene vecinos también con floración alta (las algas se
concentran en zonas de acumulación, no aparecen de forma aislada). Ignorar esto al validar
produce una sensación de que el modelo "sabe más" de lo que realmente sabe.

## 23. Validación por bloques de 1 km

Para controlar la autocorrelación espacial se dividió cada lago en una cuadrícula de bloques de
**1000 × 1000 metros**, derivada de las coordenadas UTM:

```
columna_bloque = floor(x_utm / 1000)
fila_bloque    = floor(y_utm / 1000)
```

Se usó `StratifiedGroupKFold` con 5 folds: **ningún bloque aparece a la vez en entrenamiento y
validación**, y se intenta mantener la proporción de positivos en cada fold. Se verificó
explícitamente (con un *assert* en el código) que el número de bloques compartidos entre
entrenamiento y validación fuera exactamente 0 en los 5 folds.

## 24. Validación temporal de ventana expansiva

Para simular "predecir el futuro", se ordenaron las 22 fechas (de ambos lagos combinados)
cronológicamente y se usó una estrategia de **ventana expansiva** (*rolling origin*): en cada
corte se entrena con **todas las fechas anteriores** y se evalúa con la siguiente fecha, nunca al
revés. Un *assert* garantiza que ninguna fecha de prueba aparezca también en el entrenamiento de
ese fold.

## 25. Estacionalidad, explicada desde cero

**Estacionalidad** es un patrón que se repite en función de la época del año. En Guatemala, el
año se divide en estación seca (noviembre a abril) y lluviosa (mayo a octubre). Las floraciones
de cianobacteria suelen relacionarse con la disponibilidad de nutrientes y la temperatura del
agua, que cambian con la estación. Por eso el dataset incluye una columna `season` (calculada
desde el mes de cada fecha), útil para análisis exploratorio — aunque **no se usa como predictor**
del modelo principal, para evitar que memorice la fecha en vez de la señal espectral.

## 26. Media, varianza y desviación estándar, con ejemplos del laboratorio

La **media** ($\\bar{x} = \\frac{1}{n}\\sum x_i$) es el promedio simple. La **varianza**
($\\sigma^2 = \\frac{1}{n}\\sum (x_i - \\bar{x})^2$) mide qué tan dispersos están los valores
respecto a la media. La **desviación estándar** ($\\sigma = \\sqrt{\\sigma^2}$) es la raíz de la
varianza, y tiene la ventaja de estar en las mismas unidades que los datos originales.

Ejemplo real: el PR-AUC de Random Forest en los 5 folds espaciales fue aproximadamente
[0.962, 0.994, 0.978, 0.965, 0.980] (media 0.9758, desviación 0.0128). Eso significa que, en el
peor fold, el desempeño bajó a 0.962 — todavía bueno, porque la desviación es pequeña respecto a
la media.

Compáralo con la validación temporal: media 0.7416, desviación **0.2510**. Una desviación tan
grande respecto a la media indica que **algunos folds funcionaron casi perfecto (0.99) y otros
casi fallaron por completo (0.13)** — una variabilidad enorme que una sola cifra promedio
esconde.

## 27. Por qué una desviación temporal cercana a 0.25 importa

Si te dijeran "el modelo tiene PR-AUC de 0.74 en promedio", sonaría razonable. Pero con
desviación 0.25, el rango real observado fue **[0.1307, 0.9884]**. Eso significa que **no puedes
confiar en que el modelo funcionará bien en cualquier fecha futura**: en algunas fechas
funcionará casi perfecto, y en otras será casi inútil, y no hay forma de saber de antemano cuál
será cuál sin evaluarlo. Por eso el informe insiste en que existe **inestabilidad temporal
marcada** y en que reportar solo el promedio sin la desviación sería engañoso.

## 28. Cambio de distribución (*domain shift*)

**Domain shift** ocurre cuando la relación que el modelo aprendió en unos datos deja de ser
válida en otros datos, porque las condiciones subyacentes cambiaron. Aquí ocurre de dos formas:

1. **En el tiempo:** la iluminación, la atmósfera, la temperatura del agua y la composición de la
   comunidad algal cambian de una fecha a otra, así que la relación reflectancia↔clorofila no es
   perfectamente estable.
2. **Entre lagos:** Amatitlán tiene 14.51 % de prevalencia positiva; Atitlán, 0.064 % — un factor
   de ~227 veces. Un modelo entrenado donde la floración es común y evaluado donde es rarísima (o
   viceversa) enfrenta una "probabilidad base" completamente distinta, y el umbral de decisión
   que funcionaba en un contexto deja de tener sentido en el otro.

## 29. Generalización entre Atitlán y Amatitlán

Se hicieron dos experimentos con el Random Forest (el modelo elegido por validación espacial):

- **Experimento A** (entrenar en Atitlán, evaluar en Amatitlán): PR-AUC 0.7851, pero con el
  umbral operacional el **Recall cae a 0.0401** — de 58,000 zonas positivas reales en Amatitlán,
  el modelo detecta solo 2,324 (TP) y deja pasar 55,672 (FN).
- **Experimento B** (entrenar en Amatitlán, evaluar en Atitlán): PR-AUC 0.5081, Recall
  operacional 0.1312 (282 TP, 1,868 FN).

> **Cómo interpretarlo.** Un PR-AUC de 0.785 suena razonable a primera vista, pero PR-AUC mide
> **capacidad de ordenar** correctamente los positivos por encima de los negativos —no dice nada
> sobre qué pasa con un umbral fijo de decisión—. Aquí el umbral aprendido en Atitlán (donde casi
> no hay positivos) resulta demasiado exigente para Amatitlán (donde son comunes), así que casi
> nada supera ese umbral y el Recall se desploma. **La conclusión correcta es que el modelo no
> generaliza satisfactoriamente entre lagos sin recalibrar el umbral con datos locales.**

## 30. Importancia nativa, permutation importance y SHAP

Tres formas distintas de preguntar "¿qué variables importan más?":

- **Importancia nativa (por impureza):** cuenta cuánto reduce cada variable la "impureza" (mezcla
  de clases) en los árboles del Random Forest. Resultado: B07 (0.330), B03 (0.221), B08 (0.149).
- **Permutation importance:** revuelve aleatoriamente los valores de una variable (rompiendo su
  relación con la respuesta) y mide cuánto empeora el PR-AUC. Se calculó **sobre observaciones
  que el modelo no vio durante su ajuste**, para que el resultado no esté sesgado por
  sobreajuste. Resultado: B07 domina con una caída de 0.876 en PR-AUC al permutarla — una
  diferencia enorme frente a las demás variables (0.02–0.07).
- **SHAP** (SHapley Additive exPlanations): para cada predicción individual, reparte de forma
  matemáticamente justa (basada en teoría de juegos cooperativos) cuánto contribuyó cada
  variable a esa predicción específica, no solo al modelo en general. Se calculó con
  `shap.TreeExplainer` sobre una muestra de **15,704 observaciones**, estratificada por lago y
  clase (para conservar suficientes positivos, que son escasos). Resultado: B07 (0.2108), B03
  (0.0930), B8A (0.0659).

## 31. Cómo interpretar los resultados SHAP sin afirmar causalidad

**SHAP no implica causalidad.** Que B07 sea la variable más influyente en las predicciones del
modelo no significa que "B07 causa la clorofila alta" en un sentido físico verificado — significa
que, dado cómo el modelo aprendió a combinar las variables, B07 es la que más mueve la
predicción. Además, las bandas espectrales están correlacionadas entre sí (B07, B08 y B8A miden
regiones cercanas del espectro), así que la importancia se reparte entre variables parcialmente
redundantes: un valor bajo en una banda no significa que sea prescindible, solo que otra banda
correlacionada ya está "cubriendo" esa información.

También hay que recordar: la muestra usada para SHAP tiene 36–39 % de positivos (frente al 1.6 %
poblacional), porque se sobremuestrearon deliberadamente para tener suficientes ejemplos
positivos que analizar. **Eso significa que las magnitudes SHAP no reflejan la frecuencia real de
floraciones en el lago**, son importancia relativa dentro de esa muestra.

## 32. Diferencia entre mapa descriptivo y mapa de error OOF

Se generaron dos tipos de mapa muy distintos y es crucial no confundirlos:

- **Mapa descriptivo (de despliegue):** el Random Forest se ajusta con el **100 % del dataset**
  y luego predice sobre esos mismos datos. Es útil para visualizar dónde el modelo *cree* que
  hay riesgo hoy, pero **no mide desempeño**, porque el modelo ya vio esas observaciones durante
  el entrenamiento — sería como evaluar un examen con las respuestas ya memorizadas.
- **Mapa de error out-of-fold (OOF):** cada observación se predice usando un modelo que **nunca
  la vio** durante su propio ajuste (los mismos 5 folds espaciales de la sección 23). Esta es la
  fuente correcta de cualquier estadística de desempeño espacial, y es la que respalda la matriz
  de confusión de la sección 20.

## 33. Interpretación ambiental de los resultados

**Falso negativo (FN):** una zona con floración alta que el modelo no detecta. Consecuencia: no
se emite aviso, hay riesgo de exposición recreativa o de consumo sin advertencia, y se pierde la
ventana de intervención temprana. **Falso positivo (FP):** se marca una zona sin floración real.
Consecuencia: una inspección de campo innecesaria.

**El falso negativo es el error ambientalmente más grave**, por lo que la métrica prioritaria es
el Recall (complementada con F2, que pondera el Recall por encima de la Precision), y el umbral
de decisión se eligió exigiendo un Recall mínimo de 0.80 sobre datos de validación —nunca sobre
el conjunto de prueba final—.

## 34. Limitaciones científicas

- La respuesta es un **proxy espectral** de clorofila-a, no una medición de cianobacterias ni de
  toxinas.
- **No hubo validación in situ** en ninguna fase del proyecto.
- El algoritmo CyanoLakes reporta MAPE 42.3 % y RMSE relativo 95.8 %, calibrado sobre datos
  simulados para una sola especie (*Microcystis aeruginosa*).
- 45.08 % de los píxeles de Atitlán caen fuera del dominio de calibración del NDCI.
- La colección Sentinel-2 L1C usada no expone máscara de nubes por píxel.
- Solo 22 escenas — insuficiente para tendencias interanuales.
- Desbalance 1:61 y fuerte concentración espacial/temporal de los positivos.

## 35. Qué datos in situ serían necesarios

Para validar y mejorar el sistema haría falta: (1) muestras de agua con medición de laboratorio
de clorofila-a y, si es posible, conteo de células de cianobacteria y análisis de toxinas
(microcistinas), tomadas el mismo día que pasa el satélite; (2) datos de nutrientes (nitrógeno,
fósforo) de la cuenca; (3) temperatura del agua in situ; (4) datos meteorológicos diarios (no solo
climatológicos); (5) idealmente, boyas de monitoreo continuo para capturar la variabilidad
intradiaria que una sola pasada satelital no puede ver.

## 36. Cómo convertir este prototipo en un sistema operacional

1. Automatizar la descarga de nuevas escenas Sentinel-2 apenas estén disponibles (el script
   `descargar_rasters.py` ya reintenta con espera exponencial ante errores transitorios).
2. Recalibrar el umbral de decisión periódicamente con datos de campo, en vez de fijarlo una
   sola vez.
3. Añadir una máscara de nubes real (por ejemplo, migrando a Sentinel-2 L2A con banda SCL).
4. Entrenar modelos separados por lago (dado que no generalizan bien entre sí) o incorporar
   variables de contexto que expliquen la diferencia de prevalencia.
5. Presentar los mapas de probabilidad junto con su incertidumbre, no como un número único.

## 37. Estructura de archivos y propósito de cada script

Ver sección 3. Adicionalmente: `mainlab4.py` contiene las funciones base (índices, máscara WBI,
polinomio de clorofila); `lab4_analisis.py`, utilidades de la Parte 1; `lab4.ipynb` y
`lab4-2.ipynb`, los cuadernos de la Parte 1 y la Parte 2 respectivamente.

## 38. Errores encontrados durante el proyecto y cómo se resolvieron

| Error | Causa | Corrección |
|---|---|---|
| `normalize_band()` no escalaba float32 | Solo revisaba el tipo de dato, no la magnitud | Detección por magnitud típica |
| Área de píxel incorrecta (0.01 vs 0.04 ha) | Constante fija asumiendo 10 m | Derivada de la transformación afín real |
| `penalty` deprecado en scikit-learn 1.8 | Cambio de API entre versiones | Migración a `l1_ratio`, verificada como equivalente |
| `FutureWarning` de SHAP sobre semilla global | `np.random.seed()` global interfería con el RNG interno de `shap.summary_plot` | Se pasa un generador explícito `rng=np.random.default_rng(SEED)` |
| Referencia bibliográfica incorrecta (DOI apuntaba a otro artículo) | Cita mal transcrita en un ciclo anterior | Verificada y sustituida por Mishra & Mishra (2012) |
| Filtro de nubosidad excluía tiles válidos en la descarga | Confundía nubosidad del AOI con la del tile completo | Filtro desactivado por omisión; se documentó la diferencia |

## 39. Conexión con *deep learning*

**¿Qué es una CNN (red neuronal convolucional)?** Es una arquitectura de red neuronal diseñada
para procesar datos con estructura espacial (como imágenes), aplicando filtros pequeños
("convoluciones") que se deslizan sobre la imagen detectando patrones locales (bordes, texturas)
y los combinan progresivamente en capas más profundas para reconocer patrones más complejos.

**¿Cómo se diferencia de Random Forest?** Random Forest en este proyecto trabaja **píxel por
píxel, de forma independiente**: cada fila del dataset es un punto en un espacio de 8 dimensiones,
sin ninguna noción de "vecindad". Una CNN, en cambio, procesaría un **parche de imagen**
(por ejemplo, 32×32 píxeles) de una sola vez, aprendiendo patrones espaciales —texturas,
gradientes, formas de las manchas de floración— que Random Forest no puede ver porque no conoce
la posición relativa de los píxeles entre sí.

**¿Cuándo tendría sentido usar parches de imagen?** Si el objetivo fuera detectar *patrones
espaciales* de floración (por ejemplo, distinguir una mancha compacta de ruido disperso, o
reconocer la forma característica de una nata de algas), una CNN podría aportar señal que
Random Forest ignora por construcción.

**¿Por qué no era necesariamente la mejor opción aquí?** Tres razones: (1) el problema, tal como
se planteó, es de clasificación por píxel usando información espectral, no de reconocimiento de
formas espaciales; (2) las CNN necesitan **muchos más datos etiquetados** para no sobreajustar —
aquí solo hay 22 escenas, un volumen pequeño para entrenar una red profunda desde cero; y (3) la
interpretabilidad es más difícil: Random Forest permite calcular SHAP de forma exacta y
relativamente barata (`TreeExplainer`), mientras que interpretar una CNN requiere técnicas más
costosas y menos exactas (como Grad-CAM o SHAP aproximado con `DeepExplainer`).

**¿Qué etiquetas y volumen de datos serían necesarios?** Para entrenar una CNN de forma
razonable haría falta al menos cientos de escenas etiquetadas (no 22), o técnicas de
*transfer learning* partiendo de una red preentrenada en otras imágenes satelitales, más
aumentación de datos cuidadosa (rotaciones, pero sin voltear horizontalmente si eso invierte
información geográfica relevante, análogamente a como el Laboratorio 3 evitó `horizontal_flip`
en el alfabeto de señas).

## 40. Aplicaciones profesionales reales

Este tipo de sistema, con las mejoras de la sección 36, se usa en el mundo real para: monitoreo
temprano de calidad de agua en embalses de abastecimiento público, priorización de rutas de
muestreo de campo para agencias ambientales con recursos limitados, alertas tempranas para
operadores turísticos y autoridades de salud pública, y seguimiento de tendencias de
eutrofización a largo plazo en cuerpos de agua bajo presión de desarrollo urbano o agrícola.

## 41. Glosario

- **Reflectancia:** fracción de luz incidente que una superficie refleja, entre 0 y 1.
- **DN (número digital):** valor entero almacenado en una imagen satelital antes de escalar a
  reflectancia.
- **CRS:** sistema de referencia de coordenadas.
- **Transformación afín:** conjunto de coeficientes que convierten índices de píxel en
  coordenadas geográficas reales.
- **Desbalance de clases:** situación en que una clase es mucho más frecuente que otra.
- **PR-AUC:** área bajo la curva Precision-Recall; métrica preferida con clases desbalanceadas.
- **Autocorrelación espacial:** tendencia de valores geográficamente cercanos a parecerse.
- **Fuga de información:** cuando un predictor contiene, directa o indirectamente, información
  usada para construir la respuesta.
- **SHAP:** método basado en teoría de juegos para repartir la contribución de cada variable a
  una predicción individual.
- **Domain shift:** cambio en la relación aprendida cuando se aplica el modelo a datos de
  condiciones distintas a las de entrenamiento.
- **Out-of-fold (OOF):** predicción hecha por un modelo que no vio esa observación durante su
  propio ajuste.

## 42. Bibliografía

- OECD (1982). *Eutrophication of Waters: Monitoring, Assessment and Control.* París.
  DOI: [10.1787/9789264077980-en](https://doi.org/10.1787/9789264077980-en)
- WHO (2021). *Guidelines on recreational water quality. Volume 1: coastal and fresh waters.*
  Organización Mundial de la Salud, Ginebra.
  [https://www.who.int/publications/i/item/9789240031302](https://www.who.int/publications/i/item/9789240031302)
- Mishra, S. & Mishra, D. R. (2012). *Normalized difference chlorophyll index: a novel model for
  remote estimation of chlorophyll-a concentration in turbid productive waters.* Remote Sensing
  of Environment, 117, 394–406. DOI: [10.1016/j.rse.2011.10.016](https://doi.org/10.1016/j.rse.2011.10.016)
- Sentinel Hub Custom Scripts. *Cyanobacteria Chlorophyll-a NDCI L1C.*
  [https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/cyanobacteria_chla_ndci_l1c/](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/cyanobacteria_chla_ndci_l1c/)
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.*
  Advances in Neural Information Processing Systems 30 (NeurIPS 2017). SHAP:
  [https://github.com/shap/shap](https://github.com/shap/shap)
- scikit-learn developers. *StratifiedGroupKFold.*
  [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the
  22nd ACM SIGKDD. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- Documentación oficial de Sentinel-2, Copernicus Data Space Ecosystem:
  [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
- Tobler, W. R. (1970). *A Computer Movie Simulating Urban Growth in the Detroit Region.*
  Economic Geography, 46, 234–240 (origen de la "primera ley de la geografía").

---

## Anexo — Guía completa de reproducción

```bash
# 1. Descargar los 22 GeoTIFF reales (requiere cuenta Copernicus)
python descargar_rasters.py --download

# 2. Regenerar la Parte 1 (índices, series temporales, mapas)
python regenerar_parte1_real.py --build

# 3. Construir el dataset de píxeles para ML
python preparar_dataset_ml.py --build

# 4. Entrenar y validar los tres modelos (Ejercicios 4-7)
python modelos_parte2.py --all --profile standard --n-jobs 8

# 5. Interpretabilidad, mapas y conclusiones (Ejercicios 8-10)
python explicabilidad_mapas_parte2.py --all --profile standard --n-jobs 8

# 6. Generar el informe académico (Markdown, sin PDF)
python explicabilidad_mapas_parte2.py --report-only --profile standard

# 7. Validar el resultado final
python explicabilidad_mapas_parte2.py --validate --profile standard
```

Cada script admite `--dry-run` para estimar tiempo, memoria y disco antes de ejecutar, y
`--validate` para auditar que los resultados sean correctos.
