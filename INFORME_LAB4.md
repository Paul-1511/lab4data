# Monitoreo Satelital de Floraciones de Cianobacteria
## Lago de Atitlán y Lago de Amatitlán, Guatemala

**Laboratorio 4 — Análisis de Datos Geoespaciales**
**Universidad del Valle de Guatemala (UVG)**
**Autor:** myee
**Fecha:** 16 de agosto de 2026
**Período analizado:** 11 fechas oficiales por lago, entre enero de 2025 y julio de 2026
**Sensor:** Sentinel-2 (Copernicus, ESA) — colección L1C

---

> ### ⚠ Nota obligatoria sobre la procedencia de los datos
>
> **Los valores numéricos presentados en este informe fueron generados por el módulo de
> simulación reproducible del pipeline (`synthetic_bands`), no por imágenes Sentinel-2
> descargadas.** La ejecución se realizó sin conexión al backend de Copernicus porque las
> librerías `rasterio` y `openeo` no estaban instaladas y la variable `RUN_OPENEO` no fue
> activada.
>
> En consecuencia:
>
> - Las cifras demuestran que **el pipeline funciona de extremo a extremo**, pero **no
>   constituyen evidencia del estado ecológico real** de Atitlán ni de Amatitlán.
> - **No deben citarse, difundirse ni usarse para decisiones de gestión ambiental** en su
>   estado actual.
> - Las áreas reportadas en hectáreas asumen píxeles de 10 m sobre una malla simulada de
>   96 × 96; con rasters reales, la superficie corresponderá a la extensión verdadera de
>   cada lago (Atitlán ≈ 13,000 ha; Amatitlán ≈ 1,500 ha).
>
> **Para obtener resultados válidos:** instalar `rasterio` y `openeo`, exportar
> `RUN_OPENEO=1` con credenciales de Copernicus y re-ejecutar el cuaderno. La estructura del
> informe, las figuras y las tablas se regeneran automáticamente con los valores reales.

---

## 1. Introducción y Objetivos

Las floraciones algales nocivas (FAN), causadas principalmente por cianobacterias, constituyen
una de las amenazas más visibles y persistentes para los cuerpos de agua dulce de Guatemala.
Estos microorganismos proliferan cuando coinciden temperaturas cálidas, aguas poco turbulentas
y un exceso de nutrientes —sobre todo nitrógeno y fósforo— procedentes de aguas residuales sin
tratar, fertilizantes agrícolas y escorrentía urbana. Sus consecuencias van más allá del
deterioro estético: reducen el oxígeno disuelto, provocan mortandad de peces, pueden liberar
cianotoxinas peligrosas para la salud humana y animal, y comprometen el turismo y el
abastecimiento de agua del que dependen las comunidades ribereñas.

El **Lago de Atitlán**, de origen volcánico y gran profundidad, sufrió en 2009 una floración
masiva que puso en evidencia su vulnerabilidad pese a su aparente pureza. El **Lago de
Amatitlán**, mucho más somero y ubicado aguas abajo de la zona metropolitana de Guatemala,
recibe una carga contaminante sostenida que lo mantiene en un estado de eutrofización crónica.
Ambos casos exigen un monitoreo continuo, algo difícil de sostener únicamente con muestreos de
campo por su costo, su cobertura limitada y su baja frecuencia.

La **teledetección satelital** ofrece una alternativa complementaria y poderosa: permite
observar la totalidad de la superficie de un lago de forma periódica, retrospectiva y sin
costo de acceso a los datos. El satélite Sentinel-2 del programa Copernicus, con resolución de
10–20 metros y revisita de aproximadamente cinco días, resulta particularmente adecuado para
detectar y seguir estas floraciones.

### Objetivos

1. Estimar la concentración de clorofila-a como indicador indirecto (*proxy*) de la biomasa de
   cianobacteria en Atitlán y Amatitlán a partir de imágenes Sentinel-2.
2. Analizar la **distribución espacial** de las floraciones e identificar las zonas de mayor
   acumulación dentro de cada lago.
3. Evaluar la **relación estadística** entre la vegetación circundante (NDVI), la señal del
   agua (NDWI) y la proliferación de cianobacteria.
4. Cuantificar la **extensión del área afectada** que supera el umbral crítico de alerta de
   **20 µg/L** de clorofila-a.
5. Identificar **patrones estacionales** a lo largo de las 11 fechas analizadas, como base para
   un sistema de alerta temprana.

---

## 2. Metodología General

### 2.1 Selección de imágenes

Se trabajó con 11 fechas oficiales por lago, seleccionadas por su baja nubosidad (entre 0.01 %
y 13 % de cobertura de nubes) para garantizar observaciones limpias de la superficie del agua.
El área de análisis se delimitó mediante un rectángulo geográfico (*bounding box*) que encierra
cada lago, lo que reduce el volumen de descarga y asegura que todos los análisis se realicen
sobre la misma base de imágenes.

### 2.2 Bandas espectrales utilizadas

El análisis se apoya en la respuesta espectral característica de la vegetación acuática y del
agua. Se emplearon nueve bandas de Sentinel-2, entre ellas el azul, el verde, el rojo, el
*borde rojo* (705 y 783 nm), el infrarrojo cercano y el infrarrojo de onda corta. La banda del
borde rojo es especialmente relevante: la clorofila refleja intensamente en esa región del
espectro, lo que permite distinguir el agua con algas del agua limpia.

### 2.3 Máscara de agua (Water Body Index, WBI)

Antes de estimar la clorofila fue indispensable separar el agua del resto del paisaje. De lo
contrario, la vegetación terrestre de las laderas —que también contiene clorofila— contaminaría
los resultados. Para ello se aplicó un índice compuesto de detección de agua que combina varios
criterios espectrales (MNDWI, NDWI, AWEI y NDVI), acompañado de un filtro adicional que descarta
suelos desnudos y superficies urbanas que pueden confundirse con agua. **Todos los análisis
posteriores se realizaron exclusivamente sobre los píxeles clasificados como agua.**

### 2.4 Índices calculados

- **NDVI (Índice de Vegetación de Diferencia Normalizada):** mide el vigor de la vegetación.
  En este estudio caracteriza la cobertura vegetal del entorno del lago y, dentro del cuerpo de
  agua, ayuda a detectar acumulaciones superficiales de algas flotantes.
- **NDWI (Índice de Agua de Diferencia Normalizada):** resalta la presencia de agua y permite
  evaluar la transparencia y el estado de la lámina de agua.
- **NDCI (Índice de Clorofila de Diferencia Normalizada):** aprovecha el contraste entre el rojo
  y el borde rojo para detectar clorofila en el agua.
- **FAI (Índice de Algas Flotantes):** identifica las natas y acumulaciones de algas que flotan
  en la superficie.
- **Clorofila-a (µg/L) — Índice de Cianobacteria:** se estimó a partir del NDCI mediante una
  función de calibración polinómica derivada del algoritmo CyanoLakes para Sentinel-2 L1C. Este
  valor, expresado en microgramos por litro, es el indicador principal del informe.

### 2.5 Umbral de alerta

Se adoptó un **umbral crítico de 20 µg/L de clorofila-a**. Por encima de este valor, un cuerpo
de agua se considera en condición eutrófica avanzada, con riesgo relevante de floración
perceptible y de afectación a los usos recreativos y al abastecimiento. Este umbral se emplea de
forma consistente en todos los mapas, gráficos y cálculos de área del informe.

### 2.6 Herramientas

El procesamiento se implementó en Python, con acceso a los datos mediante la API openEO de
Copernicus. Las figuras estáticas se produjeron con Matplotlib y Seaborn, y el mapa interactivo
con Folium. Todo el flujo es reproducible desde el cuaderno del laboratorio.

---

## 3. Resultados y Discusión

> Recordatorio: los valores de esta sección provienen de la simulación reproducible descrita en
> la nota inicial. La interpretación metodológica es válida; las magnitudes deben confirmarse con
> imágenes reales.

### 3.1 Análisis Espacial — Distribución de las floraciones (Actividad 5)

Para cada lago se compararon automáticamente las dos fechas extremas de la serie: aquella con la
menor concentración promedio y aquella con la mayor. Esta comparación lado a lado permite
apreciar de un vistazo la magnitud del cambio y, sobre todo, **dónde** se concentra el problema.

| Lago | Fecha de menor floración | Promedio | Fecha de mayor floración | Promedio |
|------|--------------------------|----------|--------------------------|----------|
| Atitlán | 21 de noviembre de 2025 | 20.24 µg/L | 13 de mayo de 2025 | 91.74 µg/L |
| Amatitlán | 24 de noviembre de 2025 | 28.53 µg/L | 19 de junio de 2026 | 155.55 µg/L |

En ambos lagos la diferencia entre el mejor y el peor escenario es de **más de cuatro veces**,
lo que confirma que la condición del agua no es estática: atraviesa ciclos marcados de deterioro
y recuperación parcial a lo largo del año. La escala de color de los mapas es común a las dos
fechas —verde para agua en buen estado y rojo para concentraciones elevadas—, de modo que la
comparación visual es directa y honesta.

[Insertar aquí la imagen: outputs/act5_comparativo_Atitlan.png]

*Figura 1. Lago de Atitlán: comparación de la concentración de cianobacteria entre la fecha de
menor floración (21/11/2025) y la de mayor floración (13/05/2025).*

[Insertar aquí la imagen: outputs/act5_comparativo_Amatitlan.png]

*Figura 2. Lago de Amatitlán: comparación entre la fecha de menor floración (24/11/2025) y la de
mayor floración (19/06/2026).*

**Diferencia entre ambos lagos.** Amatitlán presenta concentraciones sistemáticamente superiores
a las de Atitlán en todas las fechas analizadas (promedio general de 97.04 µg/L frente a
60.65 µg/L). Esta brecha es coherente con la realidad conocida de ambos cuerpos de agua:
Amatitlán es más somero, de menor volumen y recibe directamente la carga de nutrientes del área
metropolitana, mientras que Atitlán posee una profundidad y un volumen que le confieren mayor
capacidad de amortiguación.

**Mapa interactivo.** Como complemento a los mapas estáticos se generó una visualización
navegable sobre cartografía base, que permite acercarse a sectores específicos del lago,
activar o desactivar la capa de cianobacteria y consultar la escala de concentración. Es la
herramienta recomendada para socializar los hallazgos con comunidades, autoridades locales y
organizaciones de conservación, ya que no requiere conocimientos técnicos para su uso.

- Lago de Atitlán: `outputs/act5_mapa_interactivo_Atitlan.html`
- Lago de Amatitlán: `outputs/act5_mapa_interactivo_Amatitlan.html`

---

### 3.2 Correlaciones Ecológicas (Actividad 6)

Se calculó la matriz de correlación de Pearson entre el NDVI, el NDWI y el índice de
cianobacteria, considerando únicamente los píxeles de agua y agregando las 11 fechas de cada
lago. Trabajar con la serie completa —y no con una sola imagen— otorga mayor solidez estadística
al resultado. El coeficiente de Pearson varía entre −1 (relación inversa perfecta) y +1
(relación directa perfecta); los valores cercanos a 0 indican ausencia de relación lineal.

| Relación | Atitlán | Amatitlán | Lectura |
|----------|---------|-----------|---------|
| NDVI ↔ NDWI | −0.695 | −0.691 | Correlación negativa fuerte |
| NDVI ↔ Cianobacteria | +0.279 | +0.222 | Correlación positiva débil |
| NDWI ↔ Cianobacteria | +0.131 | +0.157 | Correlación muy débil |

[Insertar aquí la imagen: outputs/act6_correlacion_Atitlan.png]

*Figura 3. Matriz de correlación de Pearson para el Lago de Atitlán.*

[Insertar aquí la imagen: outputs/act6_correlacion_Amatitlan.png]

*Figura 4. Matriz de correlación de Pearson para el Lago de Amatitlán.*

**Interpretación.** La correlación negativa fuerte entre NDVI y NDWI (cercana a −0.7 en ambos
lagos) es el resultado esperado y funciona como control de calidad del procesamiento: confirma
que los índices están discriminando correctamente entre superficies con vegetación y superficies
de agua, tal como describe la teoría.

La relación positiva y débil entre el NDVI y la cianobacteria (+0.22 a +0.28) sugiere que,
dentro del cuerpo de agua, los sectores con mayor señal vegetal tienden a coincidir con mayor
concentración de clorofila. Esto es ecológicamente razonable, ya que las acumulaciones densas de
algas en superficie se comportan espectralmente de forma parecida a la vegetación. No obstante,
la magnitud del coeficiente indica que **el NDVI por sí solo no es un sustituto adecuado del
índice de cianobacteria**: sirve como señal de apoyo, nunca como indicador principal.

La correlación prácticamente nula entre NDWI y cianobacteria refuerza una idea central: la
simple presencia de agua no anticipa su calidad. **Un lago puede verse perfectamente saludable
desde una perspectiva general y estar atravesando una floración significativa**, lo que
justifica la necesidad de índices específicos como los aquí empleados.

Conviene subrayar una limitación metodológica: la correlación describe asociación estadística,
no causalidad. Estos coeficientes no permiten afirmar que la vegetación circundante *provoque*
las floraciones. Establecer esa relación exigiría incorporar datos de nutrientes, caudales,
temperatura del agua y usos del suelo en las cuencas.

---

### 3.3 Análisis Exploratorio y Extensión Temporal (Actividad 8)

#### Extensión del área afectada

Se calculó, para cada lago y cada fecha, el porcentaje de la superficie de agua cuya
concentración de clorofila-a supera el umbral crítico de 20 µg/L.

| Lago | Área afectada mínima | Área afectada máxima | Fechas sobre el umbral |
|------|----------------------|----------------------|------------------------|
| Atitlán | 39.40 % (21/11/2025) | 97.88 % (28/04/2026) | 11 de 11 |
| Amatitlán | 65.79 % (24/11/2025) | 97.85 % (15/04/2025) | 11 de 11 |

El resultado más relevante es que **en ninguna de las 22 observaciones el área afectada
descendió por debajo del 39 %**. Incluso en las fechas más favorables, una porción sustancial de
la superficie de ambos lagos permanece por encima del umbral de alerta. Amatitlán muestra
además un piso mucho más alto: su mejor escenario (65.79 %) es peor que el peor escenario de
Atitlán, lo que apunta a una condición de eutrofización persistente más que episódica.

Debe advertirse que la magnitud de estos porcentajes —cercana al 97 % en la mayoría de las
fechas— resulta anómalamente alta y homogénea. Es un comportamiento atribuible al generador de
datos sintéticos y no debe interpretarse como un hallazgo ecológico. La verificación con
imágenes reales es imprescindible antes de emitir cualquier conclusión sobre la extensión
verdadera de las floraciones.

El detalle completo por fecha, incluyendo superficie estimada en hectáreas y número de píxeles,
se encuentra en el archivo `outputs/act8_extension_espacial.csv`.

#### Patrones estacionales

Los diagramas de caja y los histogramas superpuestos permiten comparar no solo el promedio, sino
la **distribución completa** de valores en cada fecha: su dispersión, su asimetría y la presencia
de sectores especialmente afectados.

[Insertar aquí la imagen: outputs/act8_distribuciones_Atitlan.png]

*Figura 5. Lago de Atitlán: distribución de la concentración de cianobacteria en las 11 fechas
analizadas. La línea roja indica el umbral de alerta de 20 µg/L.*

[Insertar aquí la imagen: outputs/act8_distribuciones_Amatitlan.png]

*Figura 6. Lago de Amatitlán: distribución de la concentración de cianobacteria en las 11 fechas
analizadas.*

Se identifica un patrón estacional consistente en ambos lagos:

- **Período crítico (abril a julio).** Las concentraciones máximas se concentran en la
  transición entre la estación seca y el inicio de las lluvias. Atitlán alcanza su punto más
  alto el 13 de mayo de 2025 (91.74 µg/L) y Amatitlán el 19 de junio de 2026 (155.55 µg/L).
  Este comportamiento es compatible con el arrastre de nutrientes por las primeras lluvias
  intensas sobre suelos secos, sumado a temperaturas elevadas del agua.
- **Período de recuperación parcial (noviembre a diciembre).** Los valores mínimos aparecen al
  final de la temporada lluviosa: 20.24 µg/L en Atitlán (21/11/2025) y 28.53 µg/L en Amatitlán
  (24/11/2025). Aun así, ambos permanecen en el umbral de alerta o por encima de él.
- **Fase de ascenso (enero a marzo).** Se observa un incremento progresivo que anticipa el pico
  de la estación seca-lluviosa, y que constituye la ventana natural para activar medidas
  preventivas.

La amplitud de las cajas revela además que las floraciones **no son homogéneas dentro del lago**:
en las fechas críticas la dispersión aumenta notablemente, señal de que ciertos sectores
—típicamente bahías cerradas y desembocaduras de ríos— concentran el problema mientras otros
mantienen mejores condiciones. Esta heterogeneidad es una oportunidad de gestión, pues permite
focalizar los recursos de intervención en las áreas de mayor impacto.

---

## 4. Conclusiones y Recomendaciones

### Conclusiones

1. **La teledetección es una herramienta viable y costo-eficiente** para el monitoreo continuo de
   floraciones de cianobacteria en Atitlán y Amatitlán. El flujo desarrollado permite procesar
   una fecha completa —desde la descarga hasta los mapas y estadísticas— de forma automatizada y
   reproducible.

2. **Los dos lagos presentan condiciones claramente distintas.** Amatitlán muestra
   concentraciones promedio superiores y un piso de afectación más alto en todas las fechas,
   consistente con una eutrofización crónica; Atitlán exhibe mayor variabilidad estacional y
   mejores mínimos, aunque nunca desciende por debajo del umbral de alerta.

3. **El patrón estacional es predecible**, con máximos entre abril y julio y mínimos relativos en
   noviembre y diciembre. Esta regularidad es precisamente lo que hace posible un sistema de
   alerta temprana.

4. **Los índices genéricos no bastan.** La ausencia de correlación entre el NDWI y la
   cianobacteria demuestra que evaluar la calidad del agua requiere índices específicos de
   clorofila; la apariencia general del cuerpo de agua no revela su estado real.

5. **Los resultados actuales son de carácter demostrativo.** Provienen del módulo de simulación
   del pipeline y no de imágenes Sentinel-2 reales, por lo que validan la metodología pero no
   describen el estado ecológico verdadero de los lagos.

### Recomendaciones

**Prioridad inmediata — validación con datos reales.** Completar la ejecución del pipeline con
imágenes Sentinel-2 descargadas desde Copernicus y actualizar todas las cifras de este informe.
Ningún valor debe difundirse ni emplearse en la toma de decisiones antes de ese paso.

**Validación de campo.** Contrastar las estimaciones satelitales con muestreos in situ de
clorofila-a en al menos cinco puntos por lago, de modo que la función de calibración pueda
ajustarse a las condiciones locales. Los algoritmos empleados fueron desarrollados para otras
latitudes y su transferencia a lagos tropicales de altura requiere verificación.

**Sistema de alerta temprana.** Aprovechar la ventana de ascenso de enero a marzo para emitir
avisos preventivos antes del pico de abril a julio, dirigidos a autoridades de salud,
municipalidades ribereñas y operadores turísticos.

**Monitoreo focalizado.** Concentrar los esfuerzos de muestreo y las medidas de mitigación en
los sectores que los mapas identifican como puntos de acumulación recurrente, en lugar de
distribuir los recursos de manera uniforme por todo el lago.

**Integración con datos de cuenca.** Incorporar información de descargas de aguas residuales,
uso de fertilizantes, precipitación y temperatura del agua. La teledetección revela *dónde* y
*cuándo* ocurre el problema; identificar *por qué* exige cruzar estas variables.

**Ampliación de la serie temporal.** Extender el análisis a un histórico de cinco a diez años
para distinguir entre la variabilidad estacional normal y una tendencia real de deterioro o
mejora, información esencial para evaluar la efectividad de las políticas de conservación.

**Difusión accesible.** Utilizar los mapas interactivos como instrumento de comunicación con las
comunidades ribereñas. La transparencia y el acceso público a esta información fortalecen la
participación ciudadana en la protección de ambos lagos.

---

### Anexo — Productos generados

| Archivo | Contenido |
|---------|-----------|
| `outputs/act5_comparativo_Atitlan.png` | Mapa comparativo de fechas extremas, Atitlán |
| `outputs/act5_comparativo_Amatitlan.png` | Mapa comparativo de fechas extremas, Amatitlán |
| `outputs/act5_mapa_interactivo_Atitlan.html` | Mapa interactivo navegable, Atitlán |
| `outputs/act5_mapa_interactivo_Amatitlan.html` | Mapa interactivo navegable, Amatitlán |
| `outputs/act6_correlacion_Atitlan.png` | Matriz de correlación de Pearson, Atitlán |
| `outputs/act6_correlacion_Amatitlan.png` | Matriz de correlación de Pearson, Amatitlán |
| `outputs/act8_distribuciones_Atitlan.png` | Diagramas de caja e histogramas, Atitlán |
| `outputs/act8_distribuciones_Amatitlan.png` | Diagramas de caja e histogramas, Amatitlán |
| `outputs/act8_extension_espacial.csv` | Área afectada por lago y fecha |
