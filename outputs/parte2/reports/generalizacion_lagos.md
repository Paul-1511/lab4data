# Ejercicio 7 — Generalizacion entre lagos

> Resultados del perfil **standard**.

Generado: 2026-08-23 22:15:28 | Semilla: 42 | Dataset 2.0 (hash 65825179def16792)  
Respuesta: `high_cyano_8` (>= 8 ug/L) | Predictores: B02, B03, B07, B08, B8A, B11, B12, NDWI

---

Modelo empleado: **`random_forest`**, seleccionado por su PR-AUC medio en **validacion espacial** (no por la division aleatoria).

## 7.1-7.3 Resultados

| experimento | train | test | etiqueta | prevalencia_train | prevalencia_test | pr_auc | roc_auc | recall | precision | f2 | mcc | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Atitlan | Amatitlan | umbral_0.5 | 0.0006 | 0.1451 | 0.7851 | 0.9518 | 0.0159 | 0.9286 | 0.0198 | 0.1111 | 924 | 57072 | 71 | 341579 |
| A | Atitlan | Amatitlan | umbral_operacional | 0.0006 | 0.1451 | 0.7851 | 0.9518 | 0.0401 | 0.9230 | 0.0496 | 0.1758 | 2324 | 55672 | 194 | 341456 |
| B | Amatitlan | Atitlan | umbral_0.5 | 0.1451 | 0.0006 | 0.5081 | 0.9939 | 0.7744 | 0.4690 | 0.6852 | 0.6024 | 1665 | 485 | 1885 | 3352829 |
| B | Amatitlan | Atitlan | umbral_operacional | 0.1451 | 0.0006 | 0.5081 | 0.9939 | 0.1312 | 0.5949 | 0.1554 | 0.2792 | 282 | 1868 | 192 | 3354522 |

## 7.4-7.6 Cambio de prevalencia y direccionalidad

La prevalencia de la clase positiva es **0.064 % en Atitlan** frente a **14.51 % en Amatitlan**: un factor de unas 227 veces. Ese cambio de *probabilidad a priori* afecta sobre todo a la **Precision**, que depende de la proporcion real de positivos en el conjunto evaluado, mientras que el Recall es menos sensible.

- **Experimento A (Atitlan -> Amatitlan):** se entrena donde la clase positiva es rarisima y se evalua donde es comun. El modelo parte de muy pocos ejemplos positivos, asi que sus estimaciones son **inestables**; conviene leer las metricas con cautela.
- **Experimento B (Amatitlan -> Atitlan):** se entrena donde hay abundantes positivos y se evalua donde son excepcionales. Es previsible una caida fuerte de Precision por el cambio de prior, aunque el Recall pueda mantenerse.

Los pesos de clase y el umbral operacional se calcularon **exclusivamente con el lago de entrenamiento**; el lago de prueba no intervino en ninguna decision, ni siquiera para ajustar el umbral.

**Los resultados desfavorables se reportan tal cual.** No se modifico el umbral de 8 ug/L ni ningun otro parametro para mejorar la transferencia: hacerlo invalidaria el experimento.
