"""
Modelado y evaluacion — Laboratorio 4, Parte 2 (Ejercicios 4, 5, 6 y 7).

    4. Construccion de modelos (Regresion Logistica, Random Forest, XGBoost).
    5. Evaluacion con metricas apropiadas al desbalance.
    6. Validacion espacial por bloques de 1 km + validacion temporal cronologica.
    7. Generalizacion entre lagos.

Trabaja sobre el dataset real de los Ejercicios 1-3 (22 particiones Parquet,
3.75 millones de observaciones Sentinel-2 L1C). No usa datos sinteticos.

Modos:
    python modelos_parte2.py --dry-run
    python modelos_parte2.py --train-random
    python modelos_parte2.py --validate-spatial
    python modelos_parte2.py --validate-temporal
    python modelos_parte2.py --cross-lake
    python modelos_parte2.py --all
    python modelos_parte2.py --validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preparar_dataset_ml import (  # noqa: E402
    TARGET_COLUMN, TARGET_THRESHOLD_UG_L, UMBRALES_CANDIDATOS,
    PREDICTORES_PRINCIPALES, EXCLUIDAS_POR_FUGA, COLUMNAS_AUXILIARES,
    DATASET_VERSION, BLOQUE_ESPACIAL_M, PIXELS_DIR, hash_esquema,
    combinaciones_oficiales,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

PRIMARY_PREDICTORS = list(PREDICTORES_PRINCIPALES)

# Columnas que jamas pueden entrar a X.
PROHIBIDAS = {
    "B04", "B05", "NDCI", "chlorophyll", "FAI", "NDVI",
    "high_cyano_8", "high_cyano_20", "high_cyano_25", "high_cyano_50",
    "water_mask", "valid_data", "lake", "date", "year", "month", "season",
    "row", "col", "x_utm", "y_utm", "longitude", "latitude", "spatial_block_1km",
}

BASE = ROOT / "outputs" / "parte2"
MODELS_DIR = BASE / "models"
SPLITS_DIR = BASE / "splits"
TUNING_DIR = BASE / "tuning"
METRICS_DIR = BASE / "metrics"
FIGURES_DIR = BASE / "figures"
REPORTS_DIR = BASE / "reports"
LOG_DIR = BASE / "logs"
SUBCARPETAS = ["random", "spatial", "temporal", "cross_lake"]

TEST_SIZE = 0.30
N_FOLDS_ESPACIALES = 5

# Recall minimo exigido al umbral operacional. Se fija ANTES de ver el test.
RECALL_MINIMO_OPERACIONAL = 0.80

PERFILES = {
    # muestra: None = dataset completo. rf_max_samples acota el bootstrap del
    # Random Forest para que no construya cada arbol sobre 3.75 M de filas.
    "smoke": {"muestra": 50_000, "n_iter": 3, "cv_folds": 2,
              "tuning_max": 20_000, "rf_estimators": 30, "rf_max_samples": 0.5,
              "xgb_estimators": 60, "descripcion": "PRUEBA de humo: NO son resultados finales"},
    "standard": {"muestra": None, "n_iter": 12, "cv_folds": 3,
                 "tuning_max": 400_000, "rf_estimators": 200, "rf_max_samples": 0.25,
                 "xgb_estimators": 400, "descripcion": "Configuracion reproducible del laboratorio"},
    "full": {"muestra": None, "n_iter": 25, "cv_folds": 5,
             "tuning_max": 800_000, "rf_estimators": 400, "rf_max_samples": 0.5,
             "xgb_estimators": 800, "descripcion": "Conjunto completo, mayor costo"},
}

MODELOS_DISPONIBLES = ["logistic", "random_forest", "xgboost"]

LOGGER = logging.getLogger("modelos_parte2")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def configurar_logging(verbose=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ruta = LOG_DIR / f"modelos_{datetime.now():%Y%m%d_%H%M%S}.log"
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    c = logging.StreamHandler(sys.stdout)
    c.setLevel(logging.DEBUG if verbose else logging.INFO)
    c.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(c)
    f = logging.FileHandler(ruta, encoding="utf-8")
    f.setLevel(logging.DEBUG)
    f.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(f)
    return ruta


def aplicar_perfil_a_rutas(perfil: str) -> None:
    """
    El perfil smoke escribe en outputs/parte2/smoke/ para no contaminar jamas
    los artefactos entregables de standard/full.
    """
    global MODELS_DIR, SPLITS_DIR, TUNING_DIR, METRICS_DIR, FIGURES_DIR, REPORTS_DIR
    raiz = BASE / "smoke" if perfil == "smoke" else BASE
    MODELS_DIR, SPLITS_DIR = raiz / "models", raiz / "splits"
    TUNING_DIR, METRICS_DIR = raiz / "tuning", raiz / "metrics"
    FIGURES_DIR = raiz / "figures"
    REPORTS_DIR = raiz / "reports" if perfil == "smoke" else BASE / "reports"


def crear_carpetas():
    for d in (MODELS_DIR, SPLITS_DIR, TUNING_DIR, REPORTS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for sub in SUBCARPETAS:
        (METRICS_DIR / sub).mkdir(parents=True, exist_ok=True)
        (FIGURES_DIR / sub).mkdir(parents=True, exist_ok=True)


def presupuesto_hilos(n_jobs_total: int, n_configs: int = 1) -> tuple[int, int]:
    """
    Reparte un presupuesto global de hilos entre la busqueda y el modelo.

    Combinar RandomizedSearchCV(n_jobs=N) con un modelo que tambien usa N hilos
    crea N*N procesos compitiendo por 16 CPUs: la sobresuscripcion degrada el
    rendimiento en vez de mejorarlo. Se reparte el presupuesto de forma que
    busqueda * modelo <= presupuesto.
    """
    n_jobs_total = max(1, n_jobs_total)
    if n_configs <= 1:
        return 1, n_jobs_total
    busqueda = max(1, min(n_configs, int(np.sqrt(n_jobs_total))))
    modelo = max(1, n_jobs_total // busqueda)
    return busqueda, modelo


def limitar_blas(n: int) -> None:
    """Evita que BLAS/OpenMP abran sus propios hilos por encima del presupuesto."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(max(1, n))


def n_jobs_seguro() -> int:
    """Valor recomendado: deja una CPU libre y no supera 8, porque por encima
    el coste de coordinacion supera la ganancia en estos volumenes."""
    cpus = os.cpu_count() or 2
    return max(1, min(8, cpus - 1))


# ---------------------------------------------------------------------------
# Escritura atomica y checkpoints
# ---------------------------------------------------------------------------
def escribir_json(ruta: Path, datos: dict) -> Path:
    """Escribe a temporal, valida releyendo y renombra atomicamente."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False, default=str)
    with open(tmp, encoding="utf-8") as fh:
        json.load(fh)
    tmp.replace(ruta)
    return ruta


def escribir_csv(ruta: Path, df: pd.DataFrame) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    if pd.read_csv(tmp).empty and not df.empty:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{ruta.name}: el temporal quedo vacio")
    tmp.replace(ruta)
    return ruta


def firma_config(args, datos) -> str:
    """Huella de la configuracion. Si cambia, los checkpoints previos dejan de
    ser compatibles y la fase se recalcula."""
    base = json.dumps({
        "dataset": hash_esquema(), "version": DATASET_VERSION,
        "perfil": args.profile, "features": PRIMARY_PREDICTORS,
        "target": TARGET_COLUMN, "seed": SEED,
        "modelos": sorted(args.models or MODELOS_DISPONIBLES),
        "n_filas": int(len(datos["y"])),
    }, sort_keys=True)
    return hashlib.sha256(base.encode()).hexdigest()[:16]


def fusionar_metricas(ruta: Path, df_nuevo: pd.DataFrame, modelos: list) -> pd.DataFrame:
    """
    Invalidacion SELECTIVA por modelo.

    Sustituye en el CSV existente solo las filas de los modelos recalculados y
    conserva intactas las de los demas. Evita repetir horas de entrenamiento de
    Random Forest y XGBoost cuando el unico cambio afecta a Regresion Logistica.
    """
    if not ruta.exists():
        return df_nuevo
    try:
        previo = pd.read_csv(ruta)
    except Exception:
        return df_nuevo
    if "modelo" not in previo.columns:
        return df_nuevo
    conservado = previo[~previo["modelo"].isin(modelos)]
    fusion = pd.concat([conservado, df_nuevo], ignore_index=True, sort=False)
    LOGGER.info("  Fusion selectiva en %s: %d filas conservadas de otros modelos, "
                "%d recalculadas", ruta.name, len(conservado), len(df_nuevo))
    return fusion


def checkpoint_valido(ruta: Path, firma: str, force: bool) -> dict | None:
    """Devuelve el checkpoint si existe, esta completo y su firma coincide."""
    if force or not ruta.exists():
        return None
    try:
        with open(ruta, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return None
    if d.get("firma_config") != firma or not d.get("completo"):
        return None
    return d


def verificar_sin_fuga(columnas, contexto="X"):
    """
    Detiene el proceso si una columna prohibida entra al conjunto de predictores.

    Se llama antes de cada entrenamiento, no solo al inicio: es la ultima linea
    de defensa contra la fuga de informacion.
    """
    intrusas = [c for c in columnas if c in PROHIBIDAS]
    assert not intrusas, (
        f"FUGA DE INFORMACION en {contexto}: {intrusas}. "
        f"Estas columnas pertenecen a la cadena de la respuesta o son "
        f"identificadores de agrupacion, y no pueden usarse como predictores.")
    assert list(columnas) == PRIMARY_PREDICTORS, (
        f"El conjunto de predictores no coincide con PRIMARY_PREDICTORS.\n"
        f"  esperado: {PRIMARY_PREDICTORS}\n  recibido: {list(columnas)}")
    return True


def entorno() -> dict:
    import sklearn
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cpus": os.cpu_count(), "seed": SEED,
        "dataset_version": DATASET_VERSION, "schema_hash": hash_esquema(),
    }
    try:
        import pyarrow
        info["pyarrow"] = pyarrow.__version__
    except ImportError:
        info["pyarrow"] = None
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except ImportError:
        info["xgboost"] = None
    try:
        import shap
        info["shap"] = shap.__version__
    except ImportError:
        info["shap"] = None
    return info


# ---------------------------------------------------------------------------
# Lectura eficiente del dataset
# ---------------------------------------------------------------------------
def cargar_datos(perfil: str, semilla: int = SEED) -> dict:
    """
    Lee solo las columnas necesarias mediante proyeccion en PyArrow.

    De las 33 columnas del dataset se leen 9 (8 predictores + respuesta) mas 3
    de agrupacion. Nunca se materializan las 33.
    """
    import pyarrow.dataset as ds

    columnas = PRIMARY_PREDICTORS + [TARGET_COLUMN, "spatial_block_1km"]
    verificar_sin_fuga(PRIMARY_PREDICTORS, "lectura")

    t0 = time.time()
    dataset = ds.dataset(PIXELS_DIR, format="parquet", partitioning="hive")
    tabla = dataset.to_table(columns=columnas + ["lake", "date"])
    n_total = tabla.num_rows

    X = np.column_stack([
        tabla.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
        for c in PRIMARY_PREDICTORS
    ])
    y = tabla.column(TARGET_COLUMN).to_numpy(zero_copy_only=False).astype(np.int8)

    # Los identificadores de grupo se factorizan a enteros: guardar 3.75 M de
    # cadenas costaria decenas de MB innecesarios.
    bloques_raw = pd.Series(tabla.column("spatial_block_1km").to_pylist())
    bloque_cod, bloque_nombres = pd.factorize(bloques_raw, sort=True)
    lake_raw = pd.Series(tabla.column("lake").to_pylist()).astype(str)
    lake_cod, lake_nombres = pd.factorize(lake_raw, sort=True)
    date_raw = pd.Series(tabla.column("date").to_pylist()).astype(str)
    date_cod, date_nombres = pd.factorize(date_raw, sort=True)
    del tabla, bloques_raw, lake_raw, date_raw

    # --- Verificaciones de integridad ---
    if not np.isfinite(X).all():
        raise ValueError("X contiene NaN o infinitos: el dataset no es apto para "
                         "entrenar. Revise preparar_dataset_ml.py --validate")
    valores_y = set(np.unique(y).tolist())
    if not valores_y.issubset({0, 1}):
        raise ValueError(f"y contiene valores distintos de 0/1: {valores_y}")
    if len(valores_y) < 2:
        raise ValueError("y tiene una sola clase; no se puede clasificar")

    indices = np.arange(n_total, dtype=np.int64)

    # --- Submuestreo solo para el perfil smoke ---
    muestra = PERFILES[perfil]["muestra"]
    if muestra is not None and muestra < n_total:
        from sklearn.model_selection import train_test_split
        indices, _ = train_test_split(indices, train_size=muestra,
                                      stratify=y, random_state=semilla)
        indices = np.sort(indices)
        X, y = X[indices], y[indices]
        bloque_cod, lake_cod, date_cod = (bloque_cod[indices], lake_cod[indices],
                                          date_cod[indices])
        LOGGER.warning("PERFIL SMOKE: submuestra de %s filas. NO son resultados finales.",
                       f"{len(y):,}")

    memoria_mb = (X.nbytes + y.nbytes + bloque_cod.nbytes
                  + lake_cod.nbytes + date_cod.nbytes) / 1e6
    LOGGER.info("Dataset cargado: %s filas x %d predictores en %.1f s (%.1f MB en RAM)",
                f"{len(y):,}", X.shape[1], time.time() - t0, memoria_mb)
    LOGGER.info("  Positivos: %s (%.4f %%) | Negativos: %s",
                f"{int(y.sum()):,}", 100 * y.mean(), f"{int((y == 0).sum()):,}")

    return {"X": X, "y": y, "bloque": bloque_cod, "bloque_nombres": bloque_nombres,
            "lake": lake_cod, "lake_nombres": list(lake_nombres),
            "date": date_cod, "date_nombres": list(date_nombres),
            "n_total_dataset": n_total, "memoria_mb": memoria_mb,
            "features": list(PRIMARY_PREDICTORS)}


def hash_indices(indices: np.ndarray) -> str:
    """Firma reproducible de un conjunto de indices, para demostrar que los tres
    modelos usaron exactamente las mismas observaciones."""
    return hashlib.sha256(np.ascontiguousarray(np.sort(indices)).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Division aleatoria 70/30 (Ejercicio 4.2)
# ---------------------------------------------------------------------------
def crear_split_70_30(y: np.ndarray, semilla: int = SEED) -> dict:
    """
    Division estratificada 70/30 con semilla fija.

    El conjunto de prueba conserva la prevalencia natural: no se balancea, no se
    submuestrea y no se usa para ajustar hiperparametros ni el umbral.
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=semilla, shuffle=True)
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    return {
        "train_idx": train_idx, "test_idx": test_idx,
        "train_hash": hash_indices(train_idx), "test_hash": hash_indices(test_idx),
        "n_train": len(train_idx), "n_test": len(test_idx),
        "prevalencia_train": float(y[train_idx].mean()),
        "prevalencia_test": float(y[test_idx].mean()),
        "positivos_train": int(y[train_idx].sum()),
        "positivos_test": int(y[test_idx].sum()),
        "semilla": semilla, "test_size": TEST_SIZE,
        "estratificado_por": TARGET_COLUMN,
    }


def muestra_de_tuning(y_train: np.ndarray, perfil: str, semilla: int = SEED):
    """
    Submuestra determinista tomada EXCLUSIVAMENTE del entrenamiento.

    Conserva todos los positivos (son el recurso escaso) y toma negativos de
    forma reproducible hasta el tamano del perfil. Nunca toca el conjunto de
    prueba.
    """
    tope = PERFILES[perfil]["tuning_max"]
    pos = np.flatnonzero(y_train == 1)
    neg = np.flatnonzero(y_train == 0)
    if len(y_train) <= tope:
        return np.arange(len(y_train)), {"estrategia": "entrenamiento completo",
                                         "n": len(y_train),
                                         "prevalencia": float(y_train.mean())}
    n_neg = max(tope - len(pos), len(pos))
    rng = np.random.default_rng(semilla)
    neg_sel = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    sel = np.sort(np.concatenate([pos, neg_sel]))
    return sel, {"estrategia": "todos los positivos + negativos aleatorios (semilla fija)",
                 "n": int(len(sel)), "positivos": int(len(pos)),
                 "negativos": int(len(neg_sel)),
                 "prevalencia": float(y_train[sel].mean())}


# ---------------------------------------------------------------------------
# Modelos (Ejercicio 4.1)
# ---------------------------------------------------------------------------
def construir_modelo(nombre: str, perfil: str, n_jobs: int, scale_pos_weight=None):
    """Devuelve el estimador base. scale_pos_weight se calcula solo con el
    entrenamiento, nunca con el test."""
    cfg = PERFILES[perfil]

    if nombre == "logistic":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        # saga escala bien a millones de filas y admite penalizacion l1/l2.
        return Pipeline([
            ("escalador", StandardScaler()),
            # API de scikit-learn >= 1.8: `penalty` quedo deprecado y se sustituye
            # por `l1_ratio`, que interpola de forma continua entre las dos
            # regularizaciones:
            #     l1_ratio = 0  ->  penalizacion L2 pura  (equivale a penalty="l2")
            #     l1_ratio = 1  ->  penalizacion L1 pura  (equivale a penalty="l1")
            # La equivalencia se comprobo numericamente: con los mismos C, solver
            # y semilla los coeficientes coinciden exactamente (diferencia 0.0).
            # Usar l1_ratio evita ademas las combinaciones inconsistentes que
            # producia mezclar `penalty="l1"` con `l1_ratio=0.0`.
            # n_jobs no se pasa: desde 1.8 no tiene efecto en LogisticRegression.
            ("modelo", LogisticRegression(
                class_weight="balanced", solver="saga", max_iter=3000,
                tol=1e-4, l1_ratio=0.0, random_state=SEED)),
        ])

    if nombre == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        # max_samples acota el bootstrap: sin el, cada arbol se construiria sobre
        # 2.6 M de filas y el bosque no cabria en memoria.
        return RandomForestClassifier(
            n_estimators=cfg["rf_estimators"], max_depth=18,
            min_samples_leaf=20, max_features="sqrt",
            max_samples=cfg["rf_max_samples"],
            class_weight="balanced_subsample",
            n_jobs=n_jobs, random_state=SEED)

    if nombre == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            tree_method="hist", n_estimators=cfg["xgb_estimators"],
            max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 1.0,
            eval_metric="aucpr", early_stopping_rounds=30,
            n_jobs=n_jobs, random_state=SEED, verbosity=0)

    raise ValueError(f"Modelo desconocido: {nombre}")


def espacio_hiperparametros(nombre: str) -> dict:
    """Rejilla para la busqueda aleatoria. Los nombres siguen la convencion del
    Pipeline cuando corresponde."""
    if nombre == "logistic":
        # l1_ratio 0.0 = L2 pura; 1.0 = L1 pura. Sustituye a `penalty`, deprecado
        # en scikit-learn 1.8. Se exploran ademas dos mezclas elastic-net.
        return {"modelo__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "modelo__l1_ratio": [0.0, 0.25, 0.75, 1.0]}
    if nombre == "random_forest":
        return {"max_depth": [10, 14, 18, 24, None],
                "min_samples_leaf": [5, 10, 20, 50],
                "max_features": ["sqrt", "log2", 0.5]}
    if nombre == "xgboost":
        return {"max_depth": [4, 6, 8, 10],
                "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 5, 10, 20]}
    raise ValueError(nombre)


# ---------------------------------------------------------------------------
# Metricas (Ejercicio 5.1)
# ---------------------------------------------------------------------------
def calcular_metricas(y_true, y_prob, umbral: float, etiqueta: str = "") -> dict:
    """Conjunto completo de metricas. Accuracy NO es la metrica principal."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        balanced_accuracy_score, matthews_corrcoef, brier_score_loss)

    y_pred = (y_prob >= umbral).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    especificidad = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan

    return {
        "etiqueta": etiqueta, "umbral": float(umbral),
        "n": int(len(y_true)), "prevalencia_real": float(np.mean(y_true)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(especificidad),
        "fpr": float(fpr), "fnr": float(fnr),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def umbral_operacional(y_val, p_val, recall_minimo=RECALL_MINIMO_OPERACIONAL) -> dict:
    """
    Elige el umbral de decision usando SOLO datos de validacion internos al
    entrenamiento. Nunca se usa el conjunto de prueba.

    Criterio ambiental: un falso negativo (no avisar de una zona con floracion)
    es mas costoso que un falso positivo (una inspeccion de campo innecesaria).
    Por eso se exige primero un Recall minimo y, entre los umbrales que lo
    cumplen, se toma el de mayor Precision. Si ninguno lo alcanza, se recurre al
    umbral que maximiza F2, que ya pondera el Recall por encima de la Precision.
    """
    from sklearn.metrics import precision_recall_curve, fbeta_score

    precision, recall, umbrales = precision_recall_curve(y_val, p_val)
    precision, recall = precision[:-1], recall[:-1]

    cumplen = recall >= recall_minimo
    if cumplen.any():
        idx = int(np.argmax(np.where(cumplen, precision, -np.inf)))
        criterio = (f"Recall >= {recall_minimo:.2f} y, entre esos, maxima Precision")
    else:
        f2 = np.divide(5 * precision * recall, 4 * precision + recall,
                       out=np.zeros_like(precision), where=(4 * precision + recall) > 0)
        idx = int(np.argmax(f2))
        criterio = (f"ningun umbral alcanza Recall {recall_minimo:.2f}; "
                    "se maximiza F2 (pondera Recall sobre Precision)")

    u = float(umbrales[idx])
    y_pred = (p_val >= u).astype(int)
    return {"umbral": u, "criterio": criterio,
            "recall_minimo_exigido": recall_minimo,
            "recall_validacion": float(recall[idx]),
            "precision_validacion": float(precision[idx]),
            "f2_validacion": float(fbeta_score(y_val, y_pred, beta=2, zero_division=0)),
            "origen": "validacion interna del entrenamiento (nunca el test)"}


# ---------------------------------------------------------------------------
# MODO --dry-run
# ---------------------------------------------------------------------------
def calibrar_tiempos(n_jobs: int) -> dict:
    """
    Mide el costo real de cada modelo sobre una muestra pequena para extrapolar,
    en lugar de inventar estimaciones. Los modelos de calibracion se descartan.
    """
    from sklearn.datasets import make_classification
    n_cal = 30_000
    Xc, yc = make_classification(n_samples=n_cal, n_features=len(PRIMARY_PREDICTORS),
                                 n_informative=5, weights=[0.984, 0.016],
                                 random_state=SEED)
    tiempos = {}
    for nombre in MODELOS_DISPONIBLES:
        try:
            if nombre == "xgboost":
                import xgboost as xgb
                m = xgb.XGBClassifier(tree_method="hist", n_estimators=100,
                                      max_depth=6, n_jobs=n_jobs,
                                      random_state=SEED, verbosity=0)
            else:
                m = construir_modelo(nombre, "smoke", n_jobs)
            t0 = time.time()
            m.fit(Xc, yc)
            tiempos[nombre] = (time.time() - t0) / n_cal  # segundos por fila
        except Exception as exc:
            LOGGER.warning("  No se pudo calibrar %s: %s", nombre, exc)
            tiempos[nombre] = None
    return tiempos


def ejecutar_dry_run(args) -> int:
    perfil = args.profile
    cfg = PERFILES[perfil]

    LOGGER.info("=" * 86)
    LOGGER.info("MODO DRY-RUN — PERFIL: %s", perfil.upper())
    LOGGER.info("  %s", cfg["descripcion"])
    LOGGER.info("=" * 86)

    # --- Entorno ---
    info = entorno()
    LOGGER.info("")
    LOGGER.info("ENTORNO")
    for k in ["python", "numpy", "pandas", "pyarrow", "scikit_learn", "xgboost",
              "shap", "cpus"]:
        estado = info.get(k)
        LOGGER.info("  %-14s: %s", k, estado if estado is not None else "NO INSTALADO")
    LOGGER.info("  %-14s: %s (hash %s)", "dataset", info["dataset_version"],
                info["schema_hash"])

    if info["xgboost"] is None:
        LOGGER.error("xgboost no esta instalado: el Ejercicio 4.1 lo exige.")
        return 2
    if info["shap"] is None:
        LOGGER.warning("  shap NO instalado: se requerira para el Ejercicio 8 "
                       "(no se usa en este turno).")

    recomendado = n_jobs_seguro()
    n_jobs = args.n_jobs if args.n_jobs else recomendado
    modelos = args.models if args.models else MODELOS_DISPONIBLES
    n_busq, n_mod = presupuesto_hilos(n_jobs, PERFILES[perfil]["n_iter"])
    LOGGER.info("  %-14s: %d (recomendado seguro: %d)", "n_jobs", n_jobs, recomendado)
    LOGGER.info("  %-14s: %d busqueda x %d modelo (evita sobresuscripcion; usar los "
                "16 hilos en ambos niveles daria 16x16 procesos)",
                "reparto", n_busq, n_mod)
    LOGGER.info("  %-14s: %s", "modelos", ", ".join(modelos))

    # --- Datos ---
    LOGGER.info("")
    LOGGER.info("DATOS")
    datos = cargar_datos(perfil)
    X, y = datos["X"], datos["y"]
    n = len(y)
    pos, neg = int(y.sum()), int((y == 0).sum())

    LOGGER.info("  Columnas del dataset       : 33 (se proyectan %d)",
                len(PRIMARY_PREDICTORS) + 4)
    LOGGER.info("  Predictores                : %s", ", ".join(PRIMARY_PREDICTORS))
    LOGGER.info("  Respuesta                  : %s (>= %.0f ug/L)",
                TARGET_COLUMN, TARGET_THRESHOLD_UG_L)
    LOGGER.info("  Filas cargadas             : %s de %s",
                f"{n:,}", f"{datos['n_total_dataset']:,}")
    LOGGER.info("  Positivos / Negativos      : %s / %s (%.4f %%)",
                f"{pos:,}", f"{neg:,}", 100 * pos / n)
    LOGGER.info("  Memoria en RAM             : %.1f MB", datos["memoria_mb"])
    LOGGER.info("  Pico estimado al entrenar  : ~%.0f MB (copias internas de sklearn)",
                datos["memoria_mb"] * 3.5)

    verificar_sin_fuga(datos["features"], "dry-run")
    LOGGER.info("  Verificacion de fuga       : OK, ningun predictor prohibido")

    # --- Split 70/30 ---
    LOGGER.info("")
    LOGGER.info("EJERCICIO 4.2 — DIVISION ALEATORIA 70/30")
    split = crear_split_70_30(y)
    LOGGER.info("  Entrenamiento : %s filas | %s positivos (%.4f %%)",
                f"{split['n_train']:,}", f"{split['positivos_train']:,}",
                100 * split["prevalencia_train"])
    LOGGER.info("  Prueba        : %s filas | %s positivos (%.4f %%)",
                f"{split['n_test']:,}", f"{split['positivos_test']:,}",
                100 * split["prevalencia_test"])
    LOGGER.info("  Firma train / test : %s / %s", split["train_hash"], split["test_hash"])
    LOGGER.info("  El conjunto de prueba conserva la prevalencia natural y sera")
    LOGGER.info("  identico para los %d modelos.", len(modelos))

    spw = split["positivos_train"]
    spw = (split["n_train"] - spw) / spw if spw else 1.0
    LOGGER.info("  scale_pos_weight (XGBoost, solo con train): %.2f", spw)

    # --- Tuning ---
    LOGGER.info("")
    LOGGER.info("EJERCICIO 4.3 — AJUSTE DE HIPERPARAMETROS")
    sel_tuning, info_tuning = muestra_de_tuning(y[split["train_idx"]], perfil)
    LOGGER.info("  Estrategia         : RandomizedSearchCV, %d configuraciones x %d folds",
                cfg["n_iter"], cfg["cv_folds"])
    LOGGER.info("  Metrica de seleccion: average_precision (PR-AUC), NO Accuracy")
    LOGGER.info("  Muestra de tuning  : %s filas (%s)",
                f"{info_tuning['n']:,}", info_tuning["estrategia"])
    LOGGER.info("  Prevalencia tuning : %.4f %%", 100 * info_tuning["prevalencia"])
    LOGGER.info("  Origen             : exclusivamente del entrenamiento; el test no "
                "interviene")

    # --- Folds espaciales ---
    LOGGER.info("")
    LOGGER.info("EJERCICIO 6 — VALIDACION ESPACIAL (bloques de %.0f m)", BLOQUE_ESPACIAL_M)
    folds_esp, aviso_esp = evaluar_folds_espaciales(datos)
    n_bloques = len(np.unique(datos["bloque"]))
    bloques_pos = len(np.unique(datos["bloque"][y == 1]))
    LOGGER.info("  Bloques totales          : %d", n_bloques)
    LOGGER.info("  Bloques con positivos    : %d", bloques_pos)
    LOGGER.info("  Folds solicitados        : %d", N_FOLDS_ESPACIALES)
    LOGGER.info("  Folds validos            : %d", sum(f["valido"] for f in folds_esp))
    LOGGER.info("")
    LOGGER.info("  %-6s %10s %10s %9s %11s %-22s %s",
                "Fold", "Train", "Val", "Pos.val", "Prev.val", "Lagos en val", "Estado")
    for f in folds_esp:
        LOGGER.info("  %-6d %10s %10s %9s %10.4f%% %-22s %s",
                    f["fold"], f"{f['n_train']:,}", f"{f['n_val']:,}",
                    f"{f['pos_val']:,}", 100 * f["prev_val"],
                    ",".join(f["lagos_val"]),
                    "OK" if f["valido"] else f"INVALIDO ({f['motivo']})")
    for a in aviso_esp:
        LOGGER.warning("  AVISO: %s", a)

    # --- Folds temporales ---
    LOGGER.info("")
    LOGGER.info("VALIDACION TEMPORAL (ventana expansiva, cronologica)")
    folds_temp, aviso_temp = evaluar_folds_temporales(datos)
    LOGGER.info("  Fechas por lago          : 11")
    LOGGER.info("  Folds temporales validos : %d de %d",
                sum(f["valido"] for f in folds_temp), len(folds_temp))
    LOGGER.info("")
    LOGGER.info("  %-6s %-24s %-12s %10s %9s %11s %s",
                "Fold", "Train (hasta)", "Test", "Train", "Pos.test", "Prev.test", "Estado")
    for f in folds_temp:
        LOGGER.info("  %-6d %-24s %-12s %10s %9s %10.4f%% %s",
                    f["fold"], f["train_hasta"], f["test_fecha"],
                    f"{f['n_train']:,}", f"{f['pos_test']:,}",
                    100 * f["prev_test"],
                    "OK" if f["valido"] else f"INVALIDO ({f['motivo']})")
    for a in aviso_temp:
        LOGGER.warning("  AVISO: %s", a)

    # --- Cross-lake ---
    LOGGER.info("")
    LOGGER.info("EJERCICIO 7 — GENERALIZACION ENTRE LAGOS")
    for i, nombre in enumerate(datos["lake_nombres"]):
        m = datos["lake"] == i
        LOGGER.info("  %-11s: %s filas | %s positivos (%.4f %%)",
                    nombre, f"{int(m.sum()):,}", f"{int(y[m].sum()):,}",
                    100 * y[m].mean())
    LOGGER.info("  Experimento A: entrenar Atitlan  -> evaluar Amatitlan")
    LOGGER.info("  Experimento B: entrenar Amatitlan -> evaluar Atitlan")
    LOGGER.info("  El cambio de prevalencia entre lagos altera la probabilidad a priori")
    LOGGER.info("  y afectara sobre todo a la Precision.")

    # --- Conteo de entrenamientos y tiempo ---
    LOGGER.info("")
    LOGGER.info("COSTO ESTIMADO")
    LOGGER.info("  Midiendo velocidad real por modelo (calibracion descartable)...")
    seg_por_fila = calibrar_tiempos(n_jobs)

    n_folds_esp_validos = sum(f["valido"] for f in folds_esp)
    n_folds_temp_validos = sum(f["valido"] for f in folds_temp)
    entrenamientos = {
        "tuning": len(modelos) * cfg["n_iter"] * cfg["cv_folds"],
        "modelo final (random)": len(modelos),
        "validacion espacial": len(modelos) * n_folds_esp_validos,
        "validacion temporal": len(modelos) * n_folds_temp_validos,
        "generalizacion entre lagos": 2,
    }
    total_entrenamientos = sum(entrenamientos.values())
    LOGGER.info("")
    for etapa, cantidad in entrenamientos.items():
        LOGGER.info("  %-28s: %3d entrenamientos", etapa, cantidad)
    LOGGER.info("  %-28s: %3d", "TOTAL", total_entrenamientos)

    LOGGER.info("")
    LOGGER.info("  Tiempo estimado por etapa (medido, no supuesto):")
    total_seg = 0.0
    n_train = split["n_train"]
    for nombre in modelos:
        s = seg_por_fila.get(nombre)
        if s is None:
            LOGGER.warning("    %-14s: no calibrado", nombre)
            continue
        t_tuning = s * info_tuning["n"] * cfg["n_iter"] * cfg["cv_folds"]
        t_final = s * n_train
        t_esp = s * n_train * 0.8 * n_folds_esp_validos
        t_temp = s * n_train * 0.5 * n_folds_temp_validos
        t_modelo = t_tuning + t_final + t_esp + t_temp
        total_seg += t_modelo
        LOGGER.info("    %-14s: tuning %5.1f min | final %4.1f min | espacial %5.1f min "
                    "| temporal %5.1f min | subtotal %5.1f min",
                    nombre, t_tuning / 60, t_final / 60, t_esp / 60,
                    t_temp / 60, t_modelo / 60)
    total_seg += seg_por_fila.get("xgboost", 0) or 0 * n_train * 2
    LOGGER.info("")
    LOGGER.info("  TIEMPO TOTAL ESTIMADO      : %.0f min (~%.1f h)",
                total_seg / 60, total_seg / 3600)
    LOGGER.info("  (medicion sobre 30.000 filas y extrapolacion lineal; el Random "
                "Forest puede desviarse mas)")

    disco_mb = {"modelos serializados": 150 * len(modelos) / 3,
                "splits e indices": len(y) * 8 * 2 / 1e6,
                "metricas y tablas": 2, "figuras": 15}
    LOGGER.info("")
    LOGGER.info("  Disco estimado:")
    for k, v in disco_mb.items():
        LOGGER.info("    %-24s: %6.1f MB", k, v)
    LOGGER.info("    %-24s: %6.1f MB", "TOTAL", sum(disco_mb.values()))

    # --- Advertencias ---
    LOGGER.info("")
    LOGGER.info("ADVERTENCIAS")
    advertencias = [
        "La division aleatoria 70/30 mezcla pixeles vecinos del mismo bloque, asi "
        "que producira estimaciones OPTIMISTAS. Se incluye porque el enunciado la "
        "exige (4.2) y sirve de contraste frente a la validacion espacial.",
        f"Desbalance 1:{neg/pos:.0f}. Accuracy no puede ser la metrica principal: "
        f"predecir siempre la clase mayoritaria daria {100*neg/n:.3f} %.",
        "Los positivos estan concentrados: 37.7 % en 5 bloques y 87.6 % en 5 fechas. "
        "La validacion espacial y la temporal seran mucho mas exigentes.",
        f"Atitlan tiene una prevalencia {14.51/0.064:.0f} veces menor que Amatitlan; "
        "el Experimento A partira de una base con muy pocos positivos.",
        "El umbral operacional se fijara con validacion interna del entrenamiento; "
        "el conjunto de prueba no interviene en ninguna decision.",
    ]
    if perfil == "smoke":
        advertencias.insert(0, "PERFIL SMOKE: los resultados NO son validos para el "
                               "laboratorio, solo comprueban que el pipeline corre.")
    if info["shap"] is None:
        advertencias.append("shap no esta instalado; sera necesario para el Ejercicio 8.")
    for a in advertencias:
        LOGGER.warning("  - %s", a)

    # --- Persistir el plan ---
    crear_carpetas()
    plan = {
        "perfil": perfil, "descripcion": cfg["descripcion"],
        "generado": datetime.now().isoformat(timespec="seconds"),
        "entorno": info, "n_jobs": n_jobs, "modelos": modelos,
        "features": PRIMARY_PREDICTORS, "target": TARGET_COLUMN,
        "split": {k: v for k, v in split.items() if not isinstance(v, np.ndarray)},
        "tuning": info_tuning, "scale_pos_weight": float(spw),
        "folds_espaciales": folds_esp, "folds_temporales": folds_temp,
        "entrenamientos": entrenamientos, "total_entrenamientos": total_entrenamientos,
        "tiempo_estimado_min": total_seg / 60,
        "memoria_mb": datos["memoria_mb"], "disco_estimado_mb": sum(disco_mb.values()),
        "advertencias": advertencias,
    }
    ruta_plan = REPORTS_DIR / f"dry_run_plan_{perfil}.json"
    with open(ruta_plan, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False, default=str)
    LOGGER.info("")
    LOGGER.info("Plan guardado en: %s", ruta_plan.relative_to(ROOT))

    preparar_plantillas_reportes(perfil)

    LOGGER.info("")
    LOGGER.info("ESTADO DE LAS RUTAS")
    for ruta, funcion in [("--train-random", ejecutar_train_random),
                          ("--validate-spatial", ejecutar_validate_spatial),
                          ("--validate-temporal", ejecutar_validate_temporal),
                          ("--cross-lake", ejecutar_cross_lake),
                          ("--all", ejecutar_all)]:
        LOGGER.info("  %-20s: IMPLEMENTADA (%s)", ruta, funcion.__name__)

    LOGGER.info("")
    LOGGER.info("FASES QUE EJECUTARA --all: validacion inicial, split 70/30 con tuning,")
    LOGGER.info("  validacion espacial, validacion temporal, seleccion del mejor modelo,")
    LOGGER.info("  generalizacion entre lagos, consolidacion y validacion final.")

    aplicar_perfil_a_rutas(perfil)
    checkpoints = []
    for nombre, ruta in [("random", SPLITS_DIR / f"checkpoint_random_{perfil}.json"),
                         ("spatial", METRICS_DIR / "spatial" / f"checkpoint_{perfil}.json"),
                         ("temporal", METRICS_DIR / "temporal" / f"checkpoint_{perfil}.json"),
                         ("cross_lake", METRICS_DIR / "cross_lake" / f"checkpoint_{perfil}.json")]:
        checkpoints.append(f"{nombre}={'si' if ruta.exists() else 'no'}")
    LOGGER.info("  Checkpoints disponibles: %s", ", ".join(checkpoints))
    LOGGER.info("  Las fases con checkpoint compatible se omiten; --force las recalcula.")

    LOGGER.info("")
    LOGGER.info("DRY-RUN correcto. No se entreno ningun modelo definitivo.")
    LOGGER.info("")
    LOGGER.info("COMANDO RECOMENDADO A CONTINUACION:")
    LOGGER.info("  python modelos_parte2.py --all --profile %s --n-jobs %d",
                perfil, recomendado)
    return 0


# ---------------------------------------------------------------------------
# Planificacion de folds
# ---------------------------------------------------------------------------
def evaluar_folds_espaciales(datos, n_folds=N_FOLDS_ESPACIALES):
    """
    Construye los folds espaciales y comprueba su validez ANTES de entrenar.

    Se usa StratifiedGroupKFold: reparte los bloques intentando conservar la
    proporcion de la clase positiva en cada fold. Ningun bloque puede aparecer a
    la vez en entrenamiento y validacion.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y, bloque, lake = datos["y"], datos["bloque"], datos["lake"]
    nombres_lago = datos["lake_nombres"]
    avisos = []

    bloques_pos = len(np.unique(bloque[y == 1]))
    if bloques_pos < n_folds:
        avisos.append(f"Solo hay {bloques_pos} bloques con positivos para {n_folds} "
                      "folds; algunos quedarian sin clase positiva.")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = []
    for i, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=bloque), start=1):
        b_tr, b_va = set(np.unique(bloque[tr])), set(np.unique(bloque[va]))
        solapan = b_tr & b_va
        pos_va, pos_tr = int(y[va].sum()), int(y[tr].sum())
        valido, motivo = True, ""
        if solapan:
            valido, motivo = False, f"{len(solapan)} bloques compartidos"
        elif pos_va == 0:
            valido, motivo = False, "sin positivos en validacion"
        elif pos_tr == 0:
            valido, motivo = False, "sin positivos en entrenamiento"
        elif len(np.unique(y[va])) < 2:
            valido, motivo = False, "una sola clase en validacion"

        folds.append({
            "fold": i, "n_train": int(len(tr)), "n_val": int(len(va)),
            "bloques_train": len(b_tr), "bloques_val": len(b_va),
            "bloques_compartidos": len(solapan),
            "pos_train": pos_tr, "pos_val": pos_va,
            "prev_train": float(y[tr].mean()), "prev_val": float(y[va].mean()),
            "lagos_val": [nombres_lago[j] for j in np.unique(lake[va])],
            "lagos_train": [nombres_lago[j] for j in np.unique(lake[tr])],
            "valido": valido, "motivo": motivo,
        })
    return folds, avisos


def evaluar_folds_temporales(datos):
    """
    Validacion cronologica con ventana expansiva (rolling origin).

    Para cada corte se entrena con TODAS las fechas anteriores de ambos lagos y
    se evalua con la siguiente. Nunca entra informacion futura al entrenamiento,
    y una fecha jamas aparece a la vez en entrenamiento y prueba.
    """
    y, date_cod = datos["y"], datos["date"]
    fechas = datos["date_nombres"]          # ya vienen ordenadas alfabeticamente = cronologicamente
    avisos = []

    orden = np.argsort(fechas)
    fechas_ordenadas = [fechas[i] for i in orden]

    folds = []
    # Se arranca con al menos 4 fechas de historia para tener suficiente base.
    minimo_historia = 4
    for k in range(minimo_historia, len(fechas_ordenadas)):
        fecha_test = fechas_ordenadas[k]
        idx_test_fecha = fechas.index(fecha_test)
        anteriores = [fechas.index(f) for f in fechas_ordenadas[:k]]

        m_tr = np.isin(date_cod, anteriores)
        m_te = date_cod == idx_test_fecha
        pos_tr, pos_te = int(y[m_tr].sum()), int(y[m_te].sum())

        valido, motivo = True, ""
        if pos_te == 0:
            valido, motivo = False, "sin positivos en la fecha de prueba"
        elif len(np.unique(y[m_te])) < 2:
            valido, motivo = False, "una sola clase en la fecha de prueba"
        elif pos_tr == 0:
            valido, motivo = False, "sin positivos en el historico"

        folds.append({
            "fold": k - minimo_historia + 1,
            "train_hasta": f"{fechas_ordenadas[0]} .. {fechas_ordenadas[k-1]}",
            "test_fecha": fecha_test,
            "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
            "pos_train": pos_tr, "pos_test": pos_te,
            "prev_train": float(y[m_tr].mean()) if m_tr.any() else 0.0,
            "prev_test": float(y[m_te].mean()) if m_te.any() else 0.0,
            "valido": valido, "motivo": motivo,
        })

    if not any(f["valido"] for f in folds):
        avisos.append("Ningun fold temporal es valido.")
    avisos.append("Las fechas de ambos lagos se ordenan en una unica linea temporal "
                  "comun; cada corte entrena solo con el pasado.")
    return folds, avisos


# ---------------------------------------------------------------------------
# Plantillas de reportes (se completan al entrenar)
# ---------------------------------------------------------------------------
def preparar_plantillas_reportes(perfil: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plantillas = {
        "modelado_evaluacion.md": ("Ejercicios 4 y 5 — Modelado y evaluacion",
                                   ["Configuracion y division 70/30",
                                    "Ajuste de hiperparametros",
                                    "Metricas en el conjunto de prueba",
                                    "Umbral 0.5 frente a umbral operacional",
                                    "Curvas ROC y Precision-Recall",
                                    "Interpretacion ambiental de los errores (5.3)"]),
        "validacion_espacial.md": ("Ejercicio 6 — Validacion espacial",
                                   ["Bloques de 1 km y asignacion de folds",
                                    "Metricas por fold",
                                    "Comparacion con la division aleatoria",
                                    "Efecto de la autocorrelacion espacial"]),
        "validacion_temporal.md": ("Validacion temporal (rubrica)",
                                   ["Esquema de ventana expansiva",
                                    "Metricas por fold temporal",
                                    "Comparacion con aleatoria y espacial"]),
        "generalizacion_lagos.md": ("Ejercicio 7 — Generalizacion entre lagos",
                                    ["Experimento A: Atitlan -> Amatitlan",
                                     "Experimento B: Amatitlan -> Atitlan",
                                     "Cambio de prevalencia y probabilidad a priori",
                                     "Direccionalidad de la transferencia"]),
    }
    for archivo, (titulo, secciones) in plantillas.items():
        ruta = REPORTS_DIR / archivo
        if ruta.exists() and "PENDIENTE" not in ruta.read_text(encoding="utf-8")[:400]:
            continue  # ya tiene resultados reales: no se sobrescribe
        with open(ruta, "w", encoding="utf-8") as fh:
            fh.write(f"# {titulo}\n\n")
            fh.write("> ## ESTADO: PENDIENTE DE ENTRENAMIENTO\n>\n")
            fh.write(f"> Plantilla generada por `--dry-run` con perfil **{perfil}** el "
                     f"{datetime.now():%Y-%m-%d %H:%M}.\n>\n")
            fh.write("> Este documento **no contiene resultados** todavia. Se completa "
                     "al ejecutar el modo correspondiente.\n>\n")
            fh.write("> **Los resultados del perfil `smoke` nunca deben usarse como "
                     "conclusiones del laboratorio**: solo sirven para comprobar que el "
                     "pipeline corre. Las conclusiones deben proceder de `standard` o "
                     "`full`.\n\n")
            fh.write(f"Dataset: version {DATASET_VERSION}, respuesta `{TARGET_COLUMN}` "
                     f"(>= {TARGET_THRESHOLD_UG_L:.0f} ug/L).  \n")
            fh.write(f"Predictores: {', '.join(PRIMARY_PREDICTORS)}.  \n")
            fh.write(f"Semilla: {SEED}.\n\n---\n\n")
            for s in secciones:
                fh.write(f"## {s}\n\n_Pendiente._\n\n")
    LOGGER.info("Plantillas de reporte preparadas en %s", REPORTS_DIR.relative_to(ROOT))


# ---------------------------------------------------------------------------
# Modos de entrenamiento (implementados; se ejecutan en el siguiente turno)
# ---------------------------------------------------------------------------
def _marca(perfil: str) -> str:
    return ("SMOKE - NO ENTREGABLE" if perfil == "smoke"
            else f"perfil {perfil}")


def ajustar_hiperparametros(nombre, X, y, perfil, n_jobs, spw, args):
    """
    Busqueda aleatoria sobre una muestra determinista del ENTRENAMIENTO.

    La metrica de seleccion es average_precision (PR-AUC), nunca Accuracy: con
    1.6 % de positivos la exactitud no distingue modelos utiles.
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    cfg = PERFILES[perfil]
    ruta = TUNING_DIR / f"tuning_{nombre}_{perfil}.json"
    firma = hashlib.sha256(f"{nombre}{perfil}{len(y)}{SEED}".encode()).hexdigest()[:16]
    previo = checkpoint_valido(ruta, firma, args.force)
    if previo:
        LOGGER.info("    [OMITIR] tuning de %s reutilizado", nombre)
        return previo["mejores_parametros"], previo

    sel, info_m = muestra_de_tuning(y, perfil)
    Xt, yt = X[sel], y[sel]
    n_busq, n_mod = presupuesto_hilos(n_jobs, cfg["n_iter"])

    base = construir_modelo(nombre, perfil, n_mod, spw)
    if nombre == "xgboost":
        base.set_params(early_stopping_rounds=None)   # incompatible con CV

    t0 = time.time()
    busqueda = RandomizedSearchCV(
        base, espacio_hiperparametros(nombre), n_iter=cfg["n_iter"],
        scoring="average_precision",
        cv=StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=SEED),
        n_jobs=n_busq, random_state=SEED, refit=False, error_score="raise")
    busqueda.fit(Xt, yt)
    dur = time.time() - t0

    registro = {
        "modelo": nombre, "perfil": perfil, "firma_config": firma, "completo": True,
        "espacio": {k: [str(x) for x in v]
                    for k, v in espacio_hiperparametros(nombre).items()},
        "n_configuraciones": cfg["n_iter"], "folds": cfg["cv_folds"],
        "metrica_seleccion": "average_precision (PR-AUC)",
        "mejores_parametros": busqueda.best_params_,
        "mejor_score": float(busqueda.best_score_),
        "muestra_tuning": info_m, "segundos": dur, "semilla": SEED,
        "n_entrenamiento_disponible": int(len(y)),
        "hilos_busqueda": n_busq, "hilos_modelo": n_mod,
        "origen_muestra": "exclusivamente del conjunto de entrenamiento",
    }
    escribir_json(ruta, registro)
    LOGGER.info("    %s: PR-AUC=%.4f en %.1f s | %s",
                nombre, busqueda.best_score_, dur, busqueda.best_params_)
    return busqueda.best_params_, registro


def entrenar_y_evaluar(nombre, Xtr, ytr, Xte, yte, perfil, n_jobs, params=None,
                       usar_early_stopping=True, semilla=SEED):
    """
    Entrena un modelo y devuelve probabilidades sobre el test y sobre una
    validacion interna extraida del entrenamiento (para fijar el umbral).
    """
    from sklearn.model_selection import train_test_split

    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    modelo = construir_modelo(nombre, perfil, n_jobs, spw)
    if params:
        modelo.set_params(**params)

    idx = np.arange(len(ytr))
    if len(np.unique(ytr)) > 1 and ytr.sum() >= 10:
        tr2, val = train_test_split(idx, test_size=0.2, stratify=ytr,
                                    random_state=semilla)
    else:
        tr2, val = idx, idx

    t0 = time.time()
    if nombre == "xgboost" and usar_early_stopping and len(val) and val is not idx:
        modelo.fit(Xtr[tr2], ytr[tr2], eval_set=[(Xtr[val], ytr[val])], verbose=False)
    else:
        if nombre == "xgboost":
            modelo.set_params(early_stopping_rounds=None)
        modelo.fit(Xtr[tr2], ytr[tr2])
    dur = time.time() - t0

    return {"modelo": modelo, "p_val": modelo.predict_proba(Xtr[val])[:, 1],
            "y_val": ytr[val], "p_test": modelo.predict_proba(Xte)[:, 1],
            "segundos": dur, "scale_pos_weight": float(spw)}


def figuras_evaluacion(resultados, y_test, carpeta, perfil, sufijo=""):
    """Curvas ROC y PR, matrices de confusion, comparativa y calibracion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (roc_curve, precision_recall_curve,
                                 confusion_matrix)
    from sklearn.calibration import calibration_curve

    carpeta.mkdir(parents=True, exist_ok=True)
    titulo_extra = f" [{_marca(perfil)}]"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for nombre, r in resultados.items():
        fpr, tpr, _ = roc_curve(y_test, r["p_test"])
        axes[0].plot(fpr, tpr, lw=2,
                     label=f"{nombre} (AUC={r['metricas_05']['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(y_test, r["p_test"])
        axes[1].plot(rc, pr, lw=2,
                     label=f"{nombre} (PR-AUC={r['metricas_05']['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
                title="Curva ROC")
    axes[0].legend()
    axes[1].axhline(float(np.mean(y_test)), ls="--", c="grey", lw=1,
                    label=f"azar ({100*np.mean(y_test):.2f} %)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
    axes[1].legend()
    fig.suptitle("Curvas de desempeno" + titulo_extra, fontweight="bold")
    fig.tight_layout(); fig.savefig(carpeta / f"curvas_roc_pr{sufijo}.png", dpi=140)
    plt.close(fig)

    n = len(resultados)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9), squeeze=False)
    # squeeze=False garantiza forma (2, n) tambien cuando n == 1.
    for j, (nombre, r) in enumerate(resultados.items()):
        for i, (u, etiqueta) in enumerate([(0.5, "umbral 0.5"),
                                           (r["umbral_op"]["umbral"], "operacional")]):
            cm = confusion_matrix(y_test, (r["p_test"] >= u).astype(int), labels=[0, 1])
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{cm[a,b]:,}\n{100*norm[a,b]:.1f} %",
                            ha="center", va="center",
                            color="white" if norm[a, b] > 0.5 else "black", fontsize=9)
            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], ["Real 0", "Real 1"])
            ax.set_title(f"{nombre}\n{etiqueta}", fontsize=10)
    fig.suptitle("Matrices de confusion (absoluta y normalizada por fila)" + titulo_extra,
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(carpeta / f"matrices_confusion{sufijo}.png", dpi=140)
    plt.close(fig)

    metricas = ["pr_auc", "recall", "precision", "f2", "mcc", "balanced_accuracy"]
    df = pd.DataFrame({n_: [r["metricas_op"][m] for m in metricas]
                       for n_, r in resultados.items()}, index=metricas)
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(kind="bar", ax=ax)
    ax.set(title="Comparativa de metricas (umbral operacional)" + titulo_extra,
           ylabel="Valor")
    ax.legend(title="Modelo")
    fig.tight_layout(); fig.savefig(carpeta / f"comparativa_metricas{sufijo}.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nombre, r in resultados.items():
        axes[0].hist(r["p_test"][y_test == 0], bins=50, alpha=0.4, density=True,
                     label=f"{nombre} clase 0")
        axes[0].hist(r["p_test"][y_test == 1], bins=50, alpha=0.4, density=True,
                     label=f"{nombre} clase 1")
        try:
            pt, pp = calibration_curve(y_test, r["p_test"], n_bins=10, strategy="quantile")
            axes[1].plot(pp, pt, marker="o", label=nombre)
        except Exception:
            pass
    axes[0].set(xlabel="Probabilidad predicha", ylabel="Densidad",
                title="Distribucion de probabilidades por clase")
    axes[0].legend(fontsize=7)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(xlabel="Probabilidad media predicha", ylabel="Fraccion real de positivos",
                title="Curva de calibracion")
    axes[1].legend()
    fig.suptitle("Diagnostico de probabilidades" + titulo_extra, fontweight="bold")
    fig.tight_layout(); fig.savefig(carpeta / f"probabilidades{sufijo}.png", dpi=140)
    plt.close(fig)


def ejecutar_train_random(args, datos=None) -> int:
    """Ejercicios 4 y 5: division 70/30, tuning, entrenamiento y evaluacion."""
    import joblib

    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    n_jobs = args.n_jobs or n_jobs_seguro()
    limitar_blas(max(1, n_jobs // 2))
    modelos = args.models or MODELOS_DISPONIBLES

    LOGGER.info("=" * 86)
    LOGGER.info("EJERCICIOS 4 y 5 — DIVISION ALEATORIA 70/30 [%s]", _marca(perfil))
    LOGGER.info("=" * 86)

    if datos is None:
        datos = cargar_datos(perfil)
    X, y = datos["X"], datos["y"]
    verificar_sin_fuga(datos["features"], "train-random")
    firma = firma_config(args, datos)

    ck = SPLITS_DIR / f"checkpoint_random_{perfil}.json"
    if checkpoint_valido(ck, firma, args.force):
        LOGGER.info("Fase ya completada con la misma configuracion. Use --force para rehacer.")
        return 0

    split = crear_split_70_30(y)
    LOGGER.info("Train %s (%s pos) | Test %s (%s pos) | prevalencia test %.4f %%",
                f"{split['n_train']:,}", f"{split['positivos_train']:,}",
                f"{split['n_test']:,}", f"{split['positivos_test']:,}",
                100 * split["prevalencia_test"])
    LOGGER.info("Firma del test: %s (identica para los %d modelos)",
                split["test_hash"], len(modelos))
    if perfil != "smoke" and split["test_hash"] != "67bc81d40aad20b0":
        LOGGER.warning("La firma del test difiere de la del dry-run (%s). Es esperable "
                       "si cambio el numero de filas o la semilla.", split["test_hash"])

    tr, te = split["train_idx"], split["test_idx"]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    resultados, filas, params_por_modelo = {}, [], {}
    LOGGER.info("")
    LOGGER.info("Ajuste de hiperparametros (muestra del entrenamiento):")
    for nombre in modelos:
        spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
        params, _ = ajustar_hiperparametros(nombre, Xtr, ytr, perfil, n_jobs, spw, args)
        params_por_modelo[nombre] = params

    LOGGER.info("")
    LOGGER.info("Entrenamiento final y evaluacion:")
    for nombre in modelos:
        r = entrenar_y_evaluar(nombre, Xtr, ytr, Xte, yte, perfil, n_jobs,
                               params_por_modelo[nombre])
        u_op = umbral_operacional(r["y_val"], r["p_val"])
        m05 = calcular_metricas(yte, r["p_test"], 0.5, "umbral_0.5")
        mop = calcular_metricas(yte, r["p_test"], u_op["umbral"], "umbral_operacional")
        r.update({"umbral_op": u_op, "metricas_05": m05, "metricas_op": mop})
        resultados[nombre] = r

        joblib.dump(r["modelo"], MODELS_DIR / f"{nombre}_random_{perfil}.joblib",
                    compress=3)
        for m in (m05, mop):
            filas.append({"modelo": nombre, "perfil": perfil, "semilla": SEED, **m})
        LOGGER.info("  %-14s %5.1f s | u=0.5: PR-AUC=%.4f R=%.3f P=%.3f | "
                    "op(%.3f): R=%.3f P=%.3f F2=%.3f",
                    nombre, r["segundos"], m05["pr_auc"], m05["recall"], m05["precision"],
                    u_op["umbral"], mop["recall"], mop["precision"], mop["f2"])

    df = fusionar_metricas(METRICS_DIR / "random" / f"metricas_{perfil}.csv",
                           pd.DataFrame(filas), modelos)
    escribir_csv(METRICS_DIR / "random" / f"metricas_{perfil}.csv", df)
    escribir_json(SPLITS_DIR / "split_70_30.json",
                  {**{k: v for k, v in split.items() if not isinstance(v, np.ndarray)},
                   "perfil": perfil,
                   "modelos_que_usaron_este_test": sorted(modelos)})
    ruta_u = METRICS_DIR / "random" / f"umbrales_{perfil}.json"
    umbrales = json.load(open(ruta_u, encoding="utf-8")) if ruta_u.exists() else {}
    umbrales.update({n: r["umbral_op"] for n, r in resultados.items()})
    escribir_json(ruta_u, umbrales)

    # Coeficientes / importancias
    imp = {}
    for nombre, r in resultados.items():
        mod = r["modelo"]
        if nombre == "logistic":
            imp[nombre] = dict(zip(PRIMARY_PREDICTORS,
                                   mod.named_steps["modelo"].coef_[0].tolist()))
        elif hasattr(mod, "feature_importances_"):
            imp[nombre] = dict(zip(PRIMARY_PREDICTORS,
                                   mod.feature_importances_.tolist()))
    ruta_imp = METRICS_DIR / "random" / f"importancias_{perfil}.json"
    if ruta_imp.exists():
        previo_imp = json.load(open(ruta_imp, encoding="utf-8"))
        previo_imp.update(imp)
        imp = previo_imp
    escribir_json(ruta_imp, imp)

    figuras_evaluacion(resultados, yte, FIGURES_DIR / "random", perfil)
    reporte_modelado(df, resultados, split, params_por_modelo, perfil, imp)

    escribir_json(ck, {"firma_config": firma, "completo": True, "perfil": perfil,
                       "generado": datetime.now().isoformat(timespec="seconds"),
                       "mejor_pr_auc": float(df["pr_auc"].max())})
    LOGGER.info("")
    LOGGER.info("Ejercicios 4 y 5 completados.")
    return 0


def ejecutar_validate_spatial(args, datos=None) -> int:
    """Ejercicio 6: validacion espacial por bloques de 1 km."""
    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    n_jobs = args.n_jobs or n_jobs_seguro()
    modelos = args.models or MODELOS_DISPONIBLES

    LOGGER.info("=" * 86)
    LOGGER.info("EJERCICIO 6 — VALIDACION ESPACIAL (bloques de %.0f m) [%s]",
                BLOQUE_ESPACIAL_M, _marca(perfil))
    LOGGER.info("=" * 86)

    if datos is None:
        datos = cargar_datos(perfil)
    X, y, bloque = datos["X"], datos["y"], datos["bloque"]
    verificar_sin_fuga(datos["features"], "spatial")
    firma = firma_config(args, datos)

    ck = METRICS_DIR / "spatial" / f"checkpoint_{perfil}.json"
    if checkpoint_valido(ck, firma, args.force):
        LOGGER.info("Fase ya completada. Use --force para rehacer.")
        return 0

    n_folds = 2 if perfil == "smoke" else N_FOLDS_ESPACIALES
    folds, avisos = evaluar_folds_espaciales(datos, n_folds)
    compartidos = sum(f["bloques_compartidos"] for f in folds)
    if compartidos:
        LOGGER.error("Hay %d bloques compartidos entre train y validacion.", compartidos)
        return 1
    LOGGER.info("Folds: %d | bloques compartidos: 0 | validos: %d",
                len(folds), sum(f["valido"] for f in folds))

    params = {}
    for nombre in modelos:
        ruta = TUNING_DIR / f"tuning_{nombre}_{perfil}.json"
        if ruta.exists():
            params[nombre] = json.load(open(ruta, encoding="utf-8"))["mejores_parametros"]
        else:
            params[nombre] = {}
    LOGGER.info("Hiperparametros: los fijados en el tuning previo; los folds de "
                "validacion espacial no intervienen en su seleccion.")

    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    filas = []
    for i, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=bloque), start=1):
        info_fold = folds[i - 1]
        if not info_fold["valido"]:
            LOGGER.warning("  Fold %d INVALIDO (%s): se omite y se registra.",
                           i, info_fold["motivo"])
            filas.append({"fold": i, "modelo": None, "valido": False,
                          "motivo": info_fold["motivo"]})
            continue
        for nombre in modelos:
            r = entrenar_y_evaluar(nombre, X[tr], y[tr], X[va], y[va], perfil,
                                   n_jobs, params[nombre])
            u = umbral_operacional(r["y_val"], r["p_val"])
            m = calcular_metricas(y[va], r["p_test"], u["umbral"], f"fold{i}")
            filas.append({"fold": i, "modelo": nombre, "valido": True, "motivo": "",
                          "perfil": perfil, "umbral_op": u["umbral"],
                          "bloques_train": info_fold["bloques_train"],
                          "bloques_val": info_fold["bloques_val"],
                          "bloques_compartidos": 0,
                          "lagos_val": ",".join(info_fold["lagos_val"]), **m})
            LOGGER.info("  fold %d | %-14s PR-AUC=%.4f R=%.3f P=%.3f F2=%.3f",
                        i, nombre, m["pr_auc"], m["recall"], m["precision"], m["f2"])

    df = fusionar_metricas(METRICS_DIR / "spatial" / f"metricas_por_fold_{perfil}.csv",
                           pd.DataFrame(filas), modelos)
    escribir_csv(METRICS_DIR / "spatial" / f"metricas_por_fold_{perfil}.csv", df)

    validas = df[df["valido"] == True]  # noqa: E712
    agg = (validas.groupby("modelo")[["pr_auc", "roc_auc", "recall", "precision",
                                      "f1", "f2", "mcc", "balanced_accuracy"]]
           .agg(["mean", "std", "min", "max"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    escribir_csv(METRICS_DIR / "spatial" / f"agregado_{perfil}.csv",
                 agg.reset_index())
    LOGGER.info("")
    LOGGER.info("Agregado por modelo (media +/- desv):")
    for nombre, fila in agg.iterrows():
        LOGGER.info("  %-14s PR-AUC=%.4f +/- %.4f  [%.4f, %.4f]", nombre,
                    fila["pr_auc_mean"], fila["pr_auc_std"],
                    fila["pr_auc_min"], fila["pr_auc_max"])

    figura_folds_espaciales(datos, folds, perfil)
    mejor = agg["pr_auc_mean"].idxmax()
    escribir_json(METRICS_DIR / "spatial" / f"mejor_modelo_{perfil}.json",
                  {"mejor_modelo": mejor,
                   "pr_auc_medio": float(agg.loc[mejor, "pr_auc_mean"]),
                   "criterio": "mayor PR-AUC medio en validacion espacial",
                   "perfil": perfil})
    LOGGER.info("  Mejor modelo bajo validacion espacial: %s", mejor)

    reporte_espacial(df, agg, folds, perfil, mejor)
    escribir_json(ck, {"firma_config": firma, "completo": True, "perfil": perfil,
                       "mejor_modelo": mejor})
    return 0


def figura_folds_espaciales(datos, folds, perfil):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedGroupKFold

    y, bloque, lake = datos["y"], datos["bloque"], datos["lake"]
    n_folds = len(folds)
    asignacion = np.zeros(len(y), dtype=np.int8)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for i, (_, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=bloque), start=1):
        asignacion[va] = i

    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(y), size=min(120_000, len(y)), replace=False)
    fig, axes = plt.subplots(1, len(datos["lake_nombres"]), figsize=(8 * len(datos["lake_nombres"]), 6.5))
    axes = np.atleast_1d(axes)
    for ax, (j, nombre) in zip(axes, enumerate(datos["lake_nombres"])):
        m = sel[lake[sel] == j]
        s = ax.scatter(bloque[m] % 97, bloque[m] // 97, c=asignacion[m],
                       cmap="tab10", s=6, vmin=1, vmax=n_folds)
        ax.set_title(f"{nombre}: fold espacial asignado a cada bloque")
        ax.set_xlabel("indice de bloque (mod 97)"); ax.set_ylabel("indice de bloque // 97")
        fig.colorbar(s, ax=ax, label="fold", ticks=range(1, n_folds + 1))
    fig.suptitle(f"Asignacion de bloques de 1 km a folds espaciales [{_marca(perfil)}]",
                 fontweight="bold")
    fig.tight_layout()
    (FIGURES_DIR / "spatial").mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "spatial" / f"folds_espaciales_{perfil}.png", dpi=140)
    plt.close(fig)


def ejecutar_validate_temporal(args, datos=None) -> int:
    """Validacion cronologica con ventana expansiva."""
    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    n_jobs = args.n_jobs or n_jobs_seguro()
    modelos = args.models or MODELOS_DISPONIBLES

    LOGGER.info("=" * 86)
    LOGGER.info("VALIDACION TEMPORAL (ventana expansiva) [%s]", _marca(perfil))
    LOGGER.info("=" * 86)

    if datos is None:
        datos = cargar_datos(perfil)
    X, y, date_cod = datos["X"], datos["y"], datos["date"]
    fechas = datos["date_nombres"]
    verificar_sin_fuga(datos["features"], "temporal")
    firma = firma_config(args, datos)

    ck = METRICS_DIR / "temporal" / f"checkpoint_{perfil}.json"
    if checkpoint_valido(ck, firma, args.force):
        LOGGER.info("Fase ya completada. Use --force para rehacer.")
        return 0

    folds, _ = evaluar_folds_temporales(datos)
    if perfil == "smoke":
        validos = [f for f in folds if f["valido"]]
        folds = validos[:2] if len(validos) >= 2 else validos
    LOGGER.info("Folds temporales a ejecutar: %d", len(folds))

    params = {}
    for nombre in modelos:
        ruta = TUNING_DIR / f"tuning_{nombre}_{perfil}.json"
        params[nombre] = (json.load(open(ruta, encoding="utf-8"))["mejores_parametros"]
                          if ruta.exists() else {})

    fechas_ordenadas = sorted(fechas)
    filas = []
    for f in folds:
        if not f["valido"]:
            filas.append({"fold": f["fold"], "modelo": None, "valido": False,
                          "motivo": f["motivo"], "test_fecha": f["test_fecha"]})
            LOGGER.warning("  Fold %d omitido: %s", f["fold"], f["motivo"])
            continue
        k = fechas_ordenadas.index(f["test_fecha"])
        anteriores = [fechas.index(x) for x in fechas_ordenadas[:k]]
        m_tr = np.isin(date_cod, anteriores)
        m_te = date_cod == fechas.index(f["test_fecha"])
        # Garantia explicita: ninguna fecha esta a la vez en train y test.
        assert not set(np.unique(date_cod[m_tr])) & set(np.unique(date_cod[m_te])), \
            "Fuga temporal: una fecha aparece en entrenamiento y prueba"

        inestable = f["prev_test"] < 1e-4 or f["prev_test"] > 0.5
        for nombre in modelos:
            r = entrenar_y_evaluar(nombre, X[m_tr], y[m_tr], X[m_te], y[m_te],
                                   perfil, n_jobs, params[nombre])
            u = umbral_operacional(r["y_val"], r["p_val"])
            m = calcular_metricas(y[m_te], r["p_test"], u["umbral"], f"fold{f['fold']}")
            filas.append({"fold": f["fold"], "modelo": nombre, "valido": True,
                          "motivo": "", "perfil": perfil,
                          "train_hasta": f["train_hasta"], "test_fecha": f["test_fecha"],
                          "prevalencia_inestable": inestable,
                          "umbral_op": u["umbral"], **m})
            LOGGER.info("  fold %-2d %-12s %-14s PR-AUC=%.4f R=%.3f P=%.3f%s",
                        f["fold"], f["test_fecha"], nombre, m["pr_auc"],
                        m["recall"], m["precision"],
                        "  [prevalencia extrema]" if inestable else "")

    df = fusionar_metricas(METRICS_DIR / "temporal" / f"metricas_por_fold_{perfil}.csv",
                           pd.DataFrame(filas), modelos)
    escribir_csv(METRICS_DIR / "temporal" / f"metricas_por_fold_{perfil}.csv", df)

    validas = df[(df["valido"] == True)]  # noqa: E712
    estables = validas[validas["prevalencia_inestable"] == False]  # noqa: E712
    agg = (validas.groupby("modelo")[["pr_auc", "recall", "precision", "f2", "mcc"]]
           .agg(["mean", "std", "min", "max"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    escribir_csv(METRICS_DIR / "temporal" / f"agregado_{perfil}.csv", agg.reset_index())
    if not estables.empty:
        agg_e = (estables.groupby("modelo")[["pr_auc", "recall", "precision"]]
                 .agg(["mean", "std"]).round(4))
        agg_e.columns = ["_".join(c) for c in agg_e.columns]
        escribir_csv(METRICS_DIR / "temporal" / f"agregado_estables_{perfil}.csv",
                     agg_e.reset_index())
        LOGGER.info("")
        LOGGER.info("Se reportan dos agregados: TODOS los folds y solo los de "
                    "prevalencia no extrema (%d de %d), para no promediar "
                    "silenciosamente folds inestables.",
                    estables["fold"].nunique(), validas["fold"].nunique())

    reporte_temporal(df, agg, perfil)
    escribir_json(ck, {"firma_config": firma, "completo": True, "perfil": perfil})
    return 0


def ejecutar_cross_lake(args, datos=None) -> int:
    """Ejercicio 7: generalizacion entre lagos."""
    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    n_jobs = args.n_jobs or n_jobs_seguro()

    LOGGER.info("=" * 86)
    LOGGER.info("EJERCICIO 7 — GENERALIZACION ENTRE LAGOS [%s]", _marca(perfil))
    LOGGER.info("=" * 86)

    if datos is None:
        datos = cargar_datos(perfil)
    X, y, lake = datos["X"], datos["y"], datos["lake"]
    nombres = datos["lake_nombres"]
    verificar_sin_fuga(datos["features"], "cross-lake")
    firma = firma_config(args, datos)

    ck = METRICS_DIR / "cross_lake" / f"checkpoint_{perfil}.json"
    if checkpoint_valido(ck, firma, args.force):
        LOGGER.info("Fase ya completada. Use --force para rehacer.")
        return 0

    ruta_mejor = METRICS_DIR / "spatial" / f"mejor_modelo_{perfil}.json"
    if ruta_mejor.exists():
        mejor = json.load(open(ruta_mejor, encoding="utf-8"))["mejor_modelo"]
        LOGGER.info("Modelo seleccionado por VALIDACION ESPACIAL: %s", mejor)
    else:
        mejor = (args.models or MODELOS_DISPONIBLES)[-1]
        LOGGER.warning("No hay resultado espacial; se usa %s. Ejecute antes "
                       "--validate-spatial para elegir correctamente.", mejor)

    params = {}
    ruta_t = TUNING_DIR / f"tuning_{mejor}_{perfil}.json"
    if ruta_t.exists():
        params = json.load(open(ruta_t, encoding="utf-8"))["mejores_parametros"]

    filas, resultados = [], {}
    for i_tr, i_te in [(nombres.index("Atitlan"), nombres.index("Amatitlan")),
                       (nombres.index("Amatitlan"), nombres.index("Atitlan"))]:
        origen, destino = nombres[i_tr], nombres[i_te]
        etiqueta = "A" if origen == "Atitlan" else "B"
        m_tr, m_te = lake == i_tr, lake == i_te
        prev_tr, prev_te = float(y[m_tr].mean()), float(y[m_te].mean())
        LOGGER.info("")
        LOGGER.info("Experimento %s: entrenar %s -> evaluar %s", etiqueta, origen, destino)
        LOGGER.info("  Prevalencia entrenamiento %.4f %% -> prueba %.4f %% (factor %.0fx)",
                    100 * prev_tr, 100 * prev_te,
                    max(prev_tr, prev_te) / max(1e-12, min(prev_tr, prev_te)))

        if y[m_tr].sum() < 5:
            LOGGER.error("  %s tiene menos de 5 positivos: experimento no ejecutable.",
                         origen)
            filas.append({"experimento": etiqueta, "train": origen, "test": destino,
                          "ejecutable": False,
                          "motivo": "positivos insuficientes en el lago de entrenamiento"})
            continue

        # scale_pos_weight y umbral salen SOLO del lago de entrenamiento.
        r = entrenar_y_evaluar(mejor, X[m_tr], y[m_tr], X[m_te], y[m_te], perfil,
                               n_jobs, params)
        u = umbral_operacional(r["y_val"], r["p_val"])
        m05 = calcular_metricas(y[m_te], r["p_test"], 0.5, "umbral_0.5")
        mop = calcular_metricas(y[m_te], r["p_test"], u["umbral"], "umbral_operacional")
        r.update({"umbral_op": u, "metricas_05": m05, "metricas_op": mop})
        resultados[f"{etiqueta}: {origen}->{destino}"] = r
        for m in (m05, mop):
            filas.append({"experimento": etiqueta, "train": origen, "test": destino,
                          "ejecutable": True, "motivo": "", "modelo": mejor,
                          "perfil": perfil, "prevalencia_train": prev_tr,
                          "prevalencia_test": prev_te,
                          "scale_pos_weight": r["scale_pos_weight"],
                          "umbral_op": u["umbral"], **m})
        LOGGER.info("  u=0.5       PR-AUC=%.4f R=%.3f P=%.3f", m05["pr_auc"],
                    m05["recall"], m05["precision"])
        LOGGER.info("  operacional R=%.3f P=%.3f F2=%.3f TP=%d FN=%d",
                    mop["recall"], mop["precision"], mop["f2"], mop["TP"], mop["FN"])

    df = pd.DataFrame(filas)
    escribir_csv(METRICS_DIR / "cross_lake" / f"metricas_{perfil}.csv", df)
    if resultados:
        for etiqueta, r in resultados.items():
            sub = {etiqueta: r}
            y_te = y[lake == nombres.index(etiqueta.split("->")[1].strip())]
            figuras_evaluacion(sub, y_te, FIGURES_DIR / "cross_lake", perfil,
                               sufijo=f"_{etiqueta[0]}")
    reporte_cross_lake(df, perfil, mejor)
    escribir_json(ck, {"firma_config": firma, "completo": True, "perfil": perfil,
                       "modelo": mejor})
    return 0


def ejecutar_all(args) -> int:
    """Ejecuta las cuatro fases en orden, deteniendo las dependientes si una falla."""
    perfil = args.profile
    LOGGER.info("#" * 86)
    LOGGER.info("PIPELINE COMPLETO — PERFIL %s", perfil.upper())
    if perfil == "smoke":
        LOGGER.warning("PERFIL SMOKE: los resultados NO son entregables.")
    LOGGER.info("#" * 86)

    t0 = time.time()
    codigo_validacion = ejecutar_validate(args)
    if codigo_validacion > 1:
        LOGGER.error("La validacion inicial fallo; se detiene el pipeline.")
        return codigo_validacion

    aplicar_perfil_a_rutas(perfil)
    datos = cargar_datos(perfil)   # se carga una sola vez para las cuatro fases

    fases = [("random", ejecutar_train_random), ("spatial", ejecutar_validate_spatial),
             ("temporal", ejecutar_validate_temporal), ("cross_lake", ejecutar_cross_lake)]
    estado = {}
    for nombre, funcion in fases:
        LOGGER.info("")
        try:
            codigo = funcion(args, datos)
        except Exception as exc:
            LOGGER.exception("Fase %s fallo: %s", nombre, exc)
            estado[nombre] = f"ERROR: {exc}"
            if nombre in ("random", "spatial"):
                LOGGER.error("Las fases dependientes no se ejecutan. Los artefactos "
                             "previos validos se conservan; puede reanudar con el "
                             "mismo comando.")
                break
            continue
        estado[nombre] = "OK" if codigo == 0 else f"codigo {codigo}"
        if codigo != 0 and nombre in ("random", "spatial"):
            LOGGER.error("Fase %s devolvio %d; se detienen las dependientes.",
                         nombre, codigo)
            break

    LOGGER.info("")
    LOGGER.info("=" * 86)
    LOGGER.info("RESUMEN DEL PIPELINE (%.1f min)", (time.time() - t0) / 60)
    for nombre, _ in fases:
        LOGGER.info("  %-12s: %s", nombre, estado.get(nombre, "no ejecutada"))
    consolidar_metricas(perfil)
    final = ejecutar_validate(args)
    return 0 if all(v == "OK" for v in estado.values()) and final == 0 else 1


def consolidar_metricas(perfil: str) -> None:
    """Reune las metricas de las cuatro fases en un unico CSV comparativo."""
    partes = []
    for fase in SUBCARPETAS:
        for ruta in (METRICS_DIR / fase).glob(f"*{perfil}.csv"):
            if "agregado" in ruta.name or "checkpoint" in ruta.name:
                continue
            try:
                d = pd.read_csv(ruta)
                d.insert(0, "fase", fase)
                partes.append(d)
            except Exception:
                continue
    if partes:
        todo = pd.concat(partes, ignore_index=True, sort=False)
        escribir_csv(METRICS_DIR / f"consolidado_{perfil}.csv", todo)
        LOGGER.info("  Metricas consolidadas: %s",
                    (METRICS_DIR / f"consolidado_{perfil}.csv").relative_to(ROOT))


def _cabecera(fh, titulo, perfil):
    fh.write(f"# {titulo}\n\n")
    if perfil == "smoke":
        fh.write("> ## RESULTADOS DE PERFIL SMOKE — NO ENTREGABLES\n>\n"
                 "> Proceden de una submuestra reducida y solo comprueban que el "
                 "pipeline corre. **No deben usarse como conclusiones del "
                 "laboratorio.** Las conclusiones deben provenir del perfil "
                 "`standard` o `full`.\n\n")
    else:
        fh.write(f"> Resultados del perfil **{perfil}**.\n\n")
    fh.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S} | Semilla: {SEED} | "
             f"Dataset {DATASET_VERSION} (hash {hash_esquema()})  \n")
    fh.write(f"Respuesta: `{TARGET_COLUMN}` (>= {TARGET_THRESHOLD_UG_L:.0f} ug/L) | "
             f"Predictores: {', '.join(PRIMARY_PREDICTORS)}\n\n---\n\n")


def _tabla_metricas(fh, df, columnas=None):
    cols = columnas or ["modelo", "etiqueta", "umbral", "pr_auc", "roc_auc", "recall",
                        "precision", "f1", "f2", "mcc", "balanced_accuracy",
                        "specificity", "brier", "TP", "FN", "FP", "TN"]
    cols = [c for c in cols if c in df.columns]
    fh.write("| " + " | ".join(cols) + " |\n")
    fh.write("|" + "---|" * len(cols) + "\n")
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        fh.write("| " + " | ".join(vals) + " |\n")
    fh.write("\n")


def reporte_modelado(df, resultados, split, params, perfil, importancias):
    with open(REPORTS_DIR / "modelado_evaluacion.md", "w", encoding="utf-8") as fh:
        _cabecera(fh, "Ejercicios 4 y 5 — Modelado y evaluacion", perfil)

        fh.write("## 4.2 Division 70/30\n\n")
        fh.write(f"- Entrenamiento: **{split['n_train']:,}** filas, "
                 f"{split['positivos_train']:,} positivos "
                 f"({100*split['prevalencia_train']:.4f} %)\n")
        fh.write(f"- Prueba: **{split['n_test']:,}** filas, "
                 f"{split['positivos_test']:,} positivos "
                 f"({100*split['prevalencia_test']:.4f} %)\n")
        fh.write(f"- Firma del conjunto de prueba: `{split['test_hash']}` — "
                 "identica para los tres modelos, lo que demuestra que la "
                 "comparacion es justa.\n")
        fh.write("- El test conserva la prevalencia natural: no se balanceo, no se "
                 "submuestreo y no intervino en el ajuste de hiperparametros ni en "
                 "la eleccion del umbral.\n\n")
        fh.write("> **Advertencia.** Esta division mezcla pixeles vecinos del mismo "
                 "bloque de 1 km, que son casi identicos entre si. Por eso sus "
                 "resultados son **optimistas**. Se incluye porque el enunciado la "
                 "exige (4.2) y sirve de contraste frente a la validacion espacial.\n\n")

        fh.write("## 4.3 Ajuste de hiperparametros\n\n")
        fh.write("Busqueda aleatoria sobre una muestra determinista tomada solo del "
                 "entrenamiento. Metrica de seleccion: **average_precision (PR-AUC)**, "
                 "nunca Accuracy.\n\n")
        # Se leen de los JSON persistidos, no del estado en memoria: asi el
        # reporte queda completo aunque solo se haya recalculado un modelo.
        fh.write("| Modelo | Configuraciones | Folds | Mejores hiperparametros | "
                 "PR-AUC tuning | Segundos |\n|---|---|---|---|---|---|\n")
        espacios = {}
        for n in MODELOS_DISPONIBLES:
            rt = TUNING_DIR / f"tuning_{n}_{perfil}.json"
            if not rt.exists():
                continue
            t = json.load(open(rt, encoding="utf-8"))
            espacios[n] = t["espacio"]
            fh.write(f"| `{n}` | {t['n_configuraciones']} | {t['folds']} | "
                     f"{t['mejores_parametros']} | {t['mejor_score']:.4f} | "
                     f"{t['segundos']:.1f} |\n")
        fh.write("\nEspacio de hiperparametros explorado:\n\n")
        for n, e in espacios.items():
            fh.write(f"- `{n}`: {e}\n")
        fh.write("\n> **Nota sobre Regresion Logistica.** Se migro de `penalty` "
                 "(deprecado en scikit-learn 1.8) a `l1_ratio`: `l1_ratio=0` "
                 "equivale a L2 y `l1_ratio=1` a L1. La equivalencia se comprobo "
                 "numericamente: con el mismo C, solver y semilla los coeficientes "
                 "coinciden exactamente. Por eso la configuracion seleccionada, "
                 "antes `penalty='l2', C=0.01`, aparece ahora como "
                 "`l1_ratio=0.0, C=0.01`, y sus metricas son identicas.\n\n")

        fh.write("## 5.1 Metricas en el conjunto de prueba\n\n")
        _tabla_metricas(fh, df)
        fh.write("**Accuracy no se usa como metrica principal**: con "
                 f"{100*split['prevalencia_test']:.3f} % de positivos, predecir "
                 "siempre la clase mayoritaria daria "
                 f"{100*(1-split['prevalencia_test']):.3f} % sin detectar nada.\n\n")

        fh.write("## Umbral 0.5 frente a umbral operacional\n\n")
        fh.write("| Modelo | Umbral | Criterio | Recall (val) | Precision (val) |\n")
        fh.write("|---|---|---|---|---|\n")
        ruta_u = METRICS_DIR / "random" / f"umbrales_{perfil}.json"
        todos_u = json.load(open(ruta_u, encoding="utf-8")) if ruta_u.exists() else {}
        for n in MODELOS_DISPONIBLES:
            u = todos_u.get(n)
            if not u:
                continue
            fh.write(f"| `{n}` | {u['umbral']:.4f} | {u['criterio']} | "
                     f"{u['recall_validacion']:.4f} | {u['precision_validacion']:.4f} |\n")
        fh.write(f"\nEl umbral se eligio **exigiendo un Recall minimo de "
                 f"{RECALL_MINIMO_OPERACIONAL:.2f}** y maximizando la Precision entre "
                 "los que lo cumplen, usando **solo la validacion interna del "
                 "entrenamiento**. El conjunto de prueba nunca intervino.\n\n")

        ruta_i = METRICS_DIR / "random" / f"importancias_{perfil}.json"
        if ruta_i.exists():
            importancias = json.load(open(ruta_i, encoding="utf-8"))
        if importancias:
            fh.write("## Importancia de las variables\n\n")
            fh.write("| Variable | " + " | ".join(importancias) + " |\n")
            fh.write("|---|" + "---|" * len(importancias) + "\n")
            for v in PRIMARY_PREDICTORS:
                fh.write(f"| `{v}` | " + " | ".join(
                    f"{importancias[m].get(v, float('nan')):.4f}" for m in importancias)
                    + " |\n")
            fh.write("\n")

        mejor = df.loc[df["pr_auc"].idxmax(), "modelo"]
        fh.write("## 5.2 Comparacion\n\n")
        fh.write(f"Segun PR-AUC en el mismo conjunto de prueba, el mejor modelo de esta "
                 f"division es **`{mejor}`**. La comparacion definitiva, sin embargo, "
                 "debe hacerse con la validacion espacial.\n\n")

        fh.write("## 5.3 Interpretacion ambiental de los errores\n\n")
        fh.write("**Falso positivo (FP):** se marca como zona de alta cianobacteria un "
                 "area que no lo es. Coste: una inspeccion de campo innecesaria, alarma "
                 "infundada, posible perdida de confianza si se repite.\n\n")
        fh.write("**Falso negativo (FN):** una zona con floracion alta pasa "
                 "desapercibida. Coste: no se emite aviso, puede haber exposicion "
                 "recreativa o consumo de agua sin advertencia, y se pierde la ventana "
                 "de intervencion temprana.\n\n")
        fh.write("**Cual importa mas reducir.** En vigilancia ambiental el **falso "
                 "negativo es el error mas grave**: un aviso de mas cuesta una visita; "
                 "un aviso de menos puede costar salud publica. Por eso:\n\n")
        fh.write("- La metrica prioritaria es el **Recall** de la clase positiva.\n")
        fh.write("- Se compara con **PR-AUC**, que resume el compromiso "
                 "Precision-Recall en datos desbalanceados mucho mejor que ROC-AUC.\n")
        fh.write("- Se reporta **F2**, que pondera el Recall por encima de la "
                 "Precision, coherente con esa prioridad.\n\n")
        fh.write("> **Alcance.** Esto es una herramienta de **cribado**, no una "
                 "medicion confirmatoria. No hubo validacion in situ; el algoritmo "
                 "CyanoLakes reporta MAPE 42.3 % y RMSE relativo 95.8 %, y fue "
                 "calibrado para *Microcystis aeruginosa* sobre datos simulados. "
                 "**No permite diagnosticar toxicidad**: la clorofila mide biomasa, "
                 "no toxinas ni especies.\n")
    LOGGER.info("  Reporte: %s", (REPORTS_DIR / "modelado_evaluacion.md").relative_to(ROOT))


def reporte_espacial(df, agg, folds, perfil, mejor):
    with open(REPORTS_DIR / "validacion_espacial.md", "w", encoding="utf-8") as fh:
        _cabecera(fh, "Ejercicio 6 — Validacion espacial", perfil)
        fh.write(f"## 6.1-6.2 Bloques de {BLOQUE_ESPACIAL_M:.0f} m y folds\n\n")
        fh.write("| Fold | Bloques train | Bloques val | Compartidos | Obs. train | "
                 "Obs. val | Positivos val | Prevalencia val | Lagos val |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for f in folds:
            fh.write(f"| {f['fold']} | {f['bloques_train']} | {f['bloques_val']} | "
                     f"**{f['bloques_compartidos']}** | {f['n_train']:,} | "
                     f"{f['n_val']:,} | {f['pos_val']:,} | {100*f['prev_val']:.4f} % | "
                     f"{','.join(f['lagos_val'])} |\n")
        fh.write("\nSe uso `StratifiedGroupKFold` sobre `spatial_block_1km`: **ningun "
                 "bloque aparece a la vez en entrenamiento y validacion**, y la "
                 "proporcion de la clase positiva se conserva en cada fold.\n\n")

        fh.write("## 6.4 Metricas por fold\n\n")
        _tabla_metricas(fh, df[df["valido"] == True],  # noqa: E712
                        ["fold", "modelo", "pr_auc", "roc_auc", "recall", "precision",
                         "f2", "mcc", "TP", "FN"])
        fh.write("## Agregado por modelo\n\n")
        fh.write("| Modelo | PR-AUC medio | Desv. | Min | Max | Recall medio |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for n, r in agg.iterrows():
            fh.write(f"| `{n}` | {r['pr_auc_mean']:.4f} | {r['pr_auc_std']:.4f} | "
                     f"{r['pr_auc_min']:.4f} | {r['pr_auc_max']:.4f} | "
                     f"{r['recall_mean']:.4f} |\n")
        fh.write(f"\n**Modelo mas robusto bajo validacion espacial: `{mejor}`** "
                 "(mayor PR-AUC medio). Es el que se usa en el Ejercicio 7.\n\n")

        fh.write("## 6.5-6.6 Comparacion con la division aleatoria\n\n")
        ruta_r = METRICS_DIR / "random" / f"metricas_{perfil}.csv"
        if ruta_r.exists():
            dr = pd.read_csv(ruta_r)
            dr = dr[dr["etiqueta"] == "umbral_operacional"]
            fh.write("| Modelo | PR-AUC aleatorio | PR-AUC espacial | Caida absoluta | "
                     "Caida relativa |\n|---|---|---|---|---|\n")
            for n in agg.index:
                fila = dr[dr["modelo"] == n]
                if fila.empty:
                    continue
                a = float(fila["pr_auc"].iloc[0]); e = float(agg.loc[n, "pr_auc_mean"])
                fh.write(f"| `{n}` | {a:.4f} | {e:.4f} | {a-e:+.4f} | "
                         f"{100*(e-a)/a if a else float('nan'):+.1f} % |\n")
            fh.write("\n")
        fh.write("**Por que cae el desempeno.** La division aleatoria coloca pixeles "
                 "vecinos del mismo bloque en entrenamiento y prueba a la vez. Como la "
                 "clorofila esta espacialmente autocorrelacionada, el modelo puede "
                 "reconocer un vecino casi identico en lugar de generalizar. La "
                 "validacion espacial elimina esa ventaja al separar bloques enteros, "
                 "asi que **estima mucho mejor la capacidad de predecir en una zona "
                 "nueva**.\n")
    LOGGER.info("  Reporte: %s", (REPORTS_DIR / "validacion_espacial.md").relative_to(ROOT))


def reporte_temporal(df, agg, perfil):
    with open(REPORTS_DIR / "validacion_temporal.md", "w", encoding="utf-8") as fh:
        _cabecera(fh, "Validacion temporal (exigida por la rubrica)", perfil)
        fh.write("## Esquema\n\n")
        fh.write("Ventana expansiva (*rolling origin*): las fechas de ambos lagos se "
                 "ordenan en una unica linea temporal y, en cada corte, se entrena con "
                 "**todas las fechas anteriores** y se evalua con la siguiente. Una "
                 "asercion explicita impide que una fecha aparezca a la vez en "
                 "entrenamiento y prueba, de modo que **no entra informacion futura** "
                 "al entrenamiento.\n\n")
        fh.write("## Metricas por fold\n\n")
        validas = df[df["valido"] == True]  # noqa: E712
        _tabla_metricas(fh, validas,
                        ["fold", "test_fecha", "modelo", "prevalencia_real", "pr_auc",
                         "recall", "precision", "f2", "prevalencia_inestable"])
        inest = validas[validas["prevalencia_inestable"] == True]  # noqa: E712
        if not inest.empty:
            fh.write(f"**{inest['fold'].nunique()} folds tienen prevalencia extrema** "
                     "(por debajo de 0.01 % o por encima de 50 %). En ellos la "
                     "Precision y el PR-AUC son muy inestables. Por eso se publican "
                     "**dos agregados**: uno con todos los folds y otro solo con los "
                     "de prevalencia no extrema, en vez de promediar silenciosamente.\n\n")
        fh.write("## Agregado (todos los folds)\n\n")
        fh.write("| Modelo | PR-AUC medio | Desv. | Min | Max |\n|---|---|---|---|---|\n")
        for n, r in agg.iterrows():
            fh.write(f"| `{n}` | {r['pr_auc_mean']:.4f} | {r['pr_auc_std']:.4f} | "
                     f"{r['pr_auc_min']:.4f} | {r['pr_auc_max']:.4f} |\n")
        fh.write("\n## Comparacion de las tres estrategias\n\n")
        fh.write("| Estrategia | Que estima |\n|---|---|\n")
        fh.write("| Aleatoria 70/30 | Optimista: mezcla vecinos del mismo bloque |\n")
        fh.write("| **Espacial por bloques** | **Capacidad de predecir en una zona "
                 "nueva del lago** |\n")
        fh.write("| **Temporal expansiva** | **Capacidad de predecir una fecha futura** |\n")
        fh.write("\nLa espacial responde a *donde*; la temporal responde a *cuando*. "
                 "Para un sistema de alerta que debe anticipar floraciones futuras, la "
                 "**temporal** es la referencia mas honesta; para extrapolar a zonas no "
                 "muestreadas, lo es la **espacial**. La aleatoria no debe usarse como "
                 "estimacion de desempeno real.\n")
    LOGGER.info("  Reporte: %s", (REPORTS_DIR / "validacion_temporal.md").relative_to(ROOT))


def reporte_cross_lake(df, perfil, modelo):
    with open(REPORTS_DIR / "generalizacion_lagos.md", "w", encoding="utf-8") as fh:
        _cabecera(fh, "Ejercicio 7 — Generalizacion entre lagos", perfil)
        fh.write(f"Modelo empleado: **`{modelo}`**, seleccionado por su PR-AUC medio en "
                 "**validacion espacial** (no por la division aleatoria).\n\n")
        fh.write("## 7.1-7.3 Resultados\n\n")
        _tabla_metricas(fh, df[df.get("ejecutable", True) == True],  # noqa: E712
                        ["experimento", "train", "test", "etiqueta", "prevalencia_train",
                         "prevalencia_test", "pr_auc", "roc_auc", "recall", "precision",
                         "f2", "mcc", "TP", "FN", "FP", "TN"])
        fh.write("## 7.4-7.6 Cambio de prevalencia y direccionalidad\n\n")
        fh.write("La prevalencia de la clase positiva es **0.064 % en Atitlan** frente "
                 "a **14.51 % en Amatitlan**: un factor de unas 227 veces. Ese cambio "
                 "de *probabilidad a priori* afecta sobre todo a la **Precision**, que "
                 "depende de la proporcion real de positivos en el conjunto evaluado, "
                 "mientras que el Recall es menos sensible.\n\n")
        fh.write("- **Experimento A (Atitlan -> Amatitlan):** se entrena donde la clase "
                 "positiva es rarisima y se evalua donde es comun. El modelo parte de "
                 "muy pocos ejemplos positivos, asi que sus estimaciones son "
                 "**inestables**; conviene leer las metricas con cautela.\n")
        fh.write("- **Experimento B (Amatitlan -> Atitlan):** se entrena donde hay "
                 "abundantes positivos y se evalua donde son excepcionales. Es "
                 "previsible una caida fuerte de Precision por el cambio de prior, "
                 "aunque el Recall pueda mantenerse.\n\n")
        fh.write("Los pesos de clase y el umbral operacional se calcularon "
                 "**exclusivamente con el lago de entrenamiento**; el lago de prueba no "
                 "intervino en ninguna decision, ni siquiera para ajustar el umbral.\n\n")
        fh.write("**Los resultados desfavorables se reportan tal cual.** No se modifico "
                 "el umbral de 8 ug/L ni ningun otro parametro para mejorar la "
                 "transferencia: hacerlo invalidaria el experimento.\n")
    LOGGER.info("  Reporte: %s", (REPORTS_DIR / "generalizacion_lagos.md").relative_to(ROOT))


def ejecutar_report_only(args) -> int:
    """
    Regenera los cuatro reportes a partir de los artefactos ya guardados, sin
    reentrenar nada. Necesario tras un recalculo parcial por modelo, para que
    las tablas vuelvan a contener los tres modelos.
    """
    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    LOGGER.info("Regenerando reportes desde artefactos (sin reentrenar) [%s]",
                _marca(perfil))

    ruta_split = SPLITS_DIR / "split_70_30.json"
    if not ruta_split.exists():
        LOGGER.error("Falta %s: ejecute --train-random antes.", ruta_split)
        return 2
    split = json.load(open(ruta_split, encoding="utf-8"))

    df_r = pd.read_csv(METRICS_DIR / "random" / f"metricas_{perfil}.csv")
    reporte_modelado(df_r, {}, split, {}, perfil, {})

    ruta_sp = METRICS_DIR / "spatial" / f"metricas_por_fold_{perfil}.csv"
    if ruta_sp.exists():
        df_s = pd.read_csv(ruta_sp)
        agg = pd.read_csv(METRICS_DIR / "spatial" / f"agregado_{perfil}.csv"
                          ).set_index("modelo")
        plan = REPORTS_DIR / f"dry_run_plan_{perfil}.json"
        folds = (json.load(open(plan, encoding="utf-8"))["folds_espaciales"]
                 if plan.exists() else [])
        mejor = json.load(open(METRICS_DIR / "spatial" / f"mejor_modelo_{perfil}.json",
                               encoding="utf-8"))["mejor_modelo"]
        reporte_espacial(df_s, agg, folds, perfil, mejor)

    ruta_tp = METRICS_DIR / "temporal" / f"metricas_por_fold_{perfil}.csv"
    if ruta_tp.exists():
        df_t = pd.read_csv(ruta_tp)
        agg_t = pd.read_csv(METRICS_DIR / "temporal" / f"agregado_{perfil}.csv"
                            ).set_index("modelo")
        reporte_temporal(df_t, agg_t, perfil)

    ruta_cl = METRICS_DIR / "cross_lake" / f"metricas_{perfil}.csv"
    if ruta_cl.exists():
        df_c = pd.read_csv(ruta_cl)
        modelo = json.load(open(METRICS_DIR / "cross_lake" / f"checkpoint_{perfil}.json",
                                encoding="utf-8")).get("modelo", "random_forest")
        reporte_cross_lake(df_c, perfil, modelo)

    consolidar_metricas(perfil)
    generar_inventario(perfil)
    LOGGER.info("Reportes regenerados.")
    return 0


def generar_inventario(perfil: str) -> pd.DataFrame:
    """Inventario versionable de todos los artefactos de los Ejercicios 4-7."""
    mapa_ejercicio = {"random": "4 y 5 (division 70/30 y evaluacion)",
                      "spatial": "6 (validacion espacial)",
                      "temporal": "6 (validacion temporal, rubrica)",
                      "cross_lake": "7 (generalizacion entre lagos)",
                      "models": "4 (modelos entrenados)",
                      "tuning": "4.3 (ajuste de hiperparametros)",
                      "splits": "4.2 (division 70/30)",
                      "reports": "4-7 (reportes)"}
    filas = []
    for ruta in sorted(BASE.rglob("*")):
        if not ruta.is_file() or "smoke" in ruta.parts or "logs" in ruta.parts:
            continue
        rel = ruta.relative_to(ROOT)
        partes = ruta.relative_to(BASE).parts
        grupo = next((p for p in partes if p in mapa_ejercicio), partes[0])
        if not any(k in str(rel) for k in
                   ["metrics", "figures", "models", "tuning", "splits", "reports"]):
            continue
        estado = "OK"
        if ruta.stat().st_size == 0:
            estado = "VACIO"
        elif ruta.suffix == ".md" and "PENDIENTE" in ruta.read_text(
                encoding="utf-8", errors="ignore")[:400]:
            estado = "PENDIENTE"
        filas.append({
            "ruta": str(rel).replace("\\", "/"),
            "tipo": ruta.suffix.lstrip(".") or "sin_extension",
            "tamano_kb": round(ruta.stat().st_size / 1024, 1),
            "perfil": perfil if perfil in ruta.name or ruta.suffix in (".png", ".md")
                      else "standard",
            "ejercicio": mapa_ejercicio.get(grupo, "auxiliar"),
            "estado_validacion": estado,
        })
    df = pd.DataFrame(filas)
    escribir_csv(REPORTS_DIR / "inventario_ejercicios_4_7.csv", df)
    return df


def ejecutar_validate(args) -> int:
    """Valida el estado del pipeline y de los artefactos producidos."""
    aplicar_perfil_a_rutas(args.profile)
    crear_carpetas()
    criticos, avisos, lineas = [], [], []

    # --- Dataset de los Ejercicios 1-3 ---
    lineas.append(f"Version de dataset esperada: {DATASET_VERSION}")
    particiones = list(PIXELS_DIR.rglob("*.parquet")) if PIXELS_DIR.exists() else []
    lineas.append(f"Particiones Parquet        : {len(particiones)}")
    if len(particiones) != 22:
        criticos.append(f"Se esperaban 22 particiones, hay {len(particiones)}")

    # --- Predictores y fuga ---
    try:
        verificar_sin_fuga(PRIMARY_PREDICTORS, "validate")
        lineas.append(f"Predictores sin fuga       : {len(PRIMARY_PREDICTORS)} "
                      f"({', '.join(PRIMARY_PREDICTORS)})")
    except AssertionError as exc:
        criticos.append(str(exc))
    if TARGET_COLUMN != "high_cyano_8":
        criticos.append(f"La respuesta es {TARGET_COLUMN}, se esperaba high_cyano_8")
    else:
        lineas.append(f"Respuesta                  : {TARGET_COLUMN}")

    # --- Entorno ---
    info = entorno()
    if info["xgboost"] is None:
        criticos.append("xgboost no esta instalado (obligatorio, Ejercicio 4.1)")
    else:
        lineas.append(f"xgboost                    : {info['xgboost']}")
    if info["shap"] is None:
        avisos.append("shap no instalado (necesario para el Ejercicio 8)")

    # --- Artefactos de entrenamiento ---
    metricas = list(METRICS_DIR.rglob("*.csv")) + list(METRICS_DIR.rglob("*.json"))
    lineas.append(f"Archivos de metricas       : {len(metricas)}")
    if not metricas:
        avisos.append("Todavia no hay metricas: ejecute --all para entrenar")

    # --- Los tres modelos deben compartir el mismo test ---
    ruta_split = SPLITS_DIR / "split_70_30.json"
    if ruta_split.exists():
        with open(ruta_split, encoding="utf-8") as fh:
            s = json.load(fh)
        lineas.append(f"Firma del test             : {s.get('test_hash')}")
        usados = s.get("modelos_que_usaron_este_test", [])
        if usados and len(set(usados)) != len(usados):
            criticos.append("Hay modelos duplicados en el registro del split")
    else:
        avisos.append("No existe splits/split_70_30.json: aun no se ha entrenado")

    # --- Resultados smoke no valen como entrega ---
    for ruta in REPORTS_DIR.glob("*.md"):
        texto = ruta.read_text(encoding="utf-8")
        if "PERFIL: SMOKE" in texto.upper() and "PENDIENTE" not in texto[:400].upper():
            criticos.append(f"{ruta.name} contiene resultados de perfil smoke: no son "
                            "validos como entrega final")

    # --- Cuatro fases completas ---
    fases = {"random": SPLITS_DIR / f"checkpoint_random_{args.profile}.json",
             "spatial": METRICS_DIR / "spatial" / f"checkpoint_{args.profile}.json",
             "temporal": METRICS_DIR / "temporal" / f"checkpoint_{args.profile}.json",
             "cross_lake": METRICS_DIR / "cross_lake" / f"checkpoint_{args.profile}.json"}
    completas = [f for f, r in fases.items() if r.exists()]
    lineas.append(f"Fases completadas          : {len(completas)}/4 ({', '.join(completas)})")
    for f, r in fases.items():
        if not r.exists():
            criticos.append(f"Fase {f} sin checkpoint: no esta completa")

    # --- Modelos entrenados ---
    modelos_f = list(MODELS_DIR.glob(f"*_random_{args.profile}.joblib"))
    lineas.append(f"Modelos serializados       : {len(modelos_f)}")
    if len(modelos_f) < 3:
        criticos.append(f"Se esperaban 3 modelos, hay {len(modelos_f)}")

    # --- Metricas por fase ---
    for fase, esperado in [("random", 6), ("spatial", 15), ("temporal", 48),
                           ("cross_lake", 4)]:
        ruta = next((METRICS_DIR / fase).glob(f"metricas*{args.profile}.csv"), None)
        if ruta is None:
            criticos.append(f"Faltan metricas de la fase {fase}")
            continue
        d = pd.read_csv(ruta)
        n_val = len(d[d["valido"] == True]) if "valido" in d.columns else len(d)  # noqa: E712
        lineas.append(f"Metricas {fase:<12}      : {n_val} filas validas")
        if n_val < esperado:
            criticos.append(f"Fase {fase}: {n_val} filas, se esperaban >= {esperado}")
        obligatorias = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                        "TP", "TN", "FP", "FN", "mcc", "brier", "specificity"]
        faltan_m = [c for c in obligatorias if c not in d.columns]
        if faltan_m:
            criticos.append(f"Fase {fase}: faltan metricas {faltan_m}")

    # --- Folds espaciales disjuntos ---
    ruta_sp = METRICS_DIR / "spatial" / f"metricas_por_fold_{args.profile}.csv"
    if ruta_sp.exists():
        d = pd.read_csv(ruta_sp)
        if "bloques_compartidos" in d.columns:
            comp = int(d["bloques_compartidos"].fillna(0).sum())
            lineas.append(f"Bloques espaciales compartidos: {comp}")
            if comp:
                criticos.append(f"Hay {comp} bloques en train y validacion a la vez")
        n_folds = d[d["valido"] == True]["fold"].nunique()  # noqa: E712
        if n_folds != 5:
            criticos.append(f"Se esperaban 5 folds espaciales, hay {n_folds}")

    # --- Orden temporal ---
    ruta_tp = METRICS_DIR / "temporal" / f"metricas_por_fold_{args.profile}.csv"
    if ruta_tp.exists():
        d = pd.read_csv(ruta_tp)
        v = d[d["valido"] == True]  # noqa: E712
        lineas.append(f"Folds temporales           : {v['fold'].nunique()}")
        if "train_hasta" in v.columns and "test_fecha" in v.columns:
            malos = [r for _, r in v.iterrows()
                     if str(r["test_fecha"]) <= str(r["train_hasta"]).split("..")[-1].strip()]
            if malos:
                criticos.append(f"{len(malos)} folds temporales con fecha de prueba "
                                "anterior o igual al fin del entrenamiento")

    # --- Dos experimentos cross-lake ---
    ruta_cl = METRICS_DIR / "cross_lake" / f"metricas_{args.profile}.csv"
    if ruta_cl.exists():
        d = pd.read_csv(ruta_cl)
        exps = sorted(d["experimento"].dropna().unique())
        lineas.append(f"Experimentos entre lagos   : {len(exps)} ({', '.join(exps)})")
        if len(exps) != 2:
            criticos.append(f"Se esperaban 2 experimentos, hay {len(exps)}")

    # --- Reportes completos, sin plantillas pendientes ---
    reportes = ["modelado_evaluacion.md", "validacion_espacial.md",
                "validacion_temporal.md", "generalizacion_lagos.md"]
    for nombre in reportes:
        r = REPORTS_DIR / nombre
        if not r.exists():
            criticos.append(f"Falta el reporte {nombre}")
            continue
        texto = r.read_text(encoding="utf-8")
        if "PENDIENTE" in texto[:600] or "_Pendiente._" in texto:
            criticos.append(f"{nombre} sigue siendo una plantilla PENDIENTE")
    lineas.append(f"Reportes completos         : {len(reportes)}")

    # --- Coherencia codigo <-> hiperparametros guardados ---
    for n in MODELOS_DISPONIBLES:
        rt = TUNING_DIR / f"tuning_{n}_{args.profile}.json"
        if not rt.exists():
            criticos.append(f"Falta el tuning de {n}")
            continue
        guardados = json.load(open(rt, encoding="utf-8"))["mejores_parametros"]
        validos = set(espacio_hiperparametros(n))
        incompatibles = [k for k in guardados if k not in validos]
        if incompatibles:
            criticos.append(f"{n}: hiperparametros guardados {incompatibles} ya no "
                            "existen en el espacio del codigo (artefacto obsoleto)")
    lineas.append("Coherencia codigo/hiperparametros: verificada")

    # --- Inventario ---
    inv = generar_inventario(args.profile)
    lineas.append(f"Inventario                 : {len(inv)} artefactos")
    pend = inv[inv["estado_validacion"] != "OK"]
    if not pend.empty:
        criticos.append(f"{len(pend)} artefactos no validos: "
                        f"{pend['ruta'].tolist()[:3]}")

    # --- Notebook actualizado ---
    nb = ROOT / "lab4-2.ipynb"
    if nb.exists():
        t = nb.read_text(encoding="utf-8")
        tiene_4_7 = all(k in t for k in ["Ejercicio 4", "Ejercicio 5", "Ejercicio 6",
                                         "Ejercicio 7"])
        lineas.append(f"Notebook con Ejercicios 4-7: {tiene_4_7}")
        if not tiene_4_7:
            criticos.append("lab4-2.ipynb no incluye los Ejercicios 4 a 7")
        if "smoke" in t.lower() and "NO ENTREGABLE" not in t:
            avisos.append("El notebook menciona smoke; verifique que no muestre "
                          "resultados smoke")

    # --- Ningun artefacto smoke mezclado ---
    smoke_fuera = [p for p in BASE.rglob("*smoke*")
                   if p.is_file() and "smoke" not in p.relative_to(BASE).parts[:1]]
    if smoke_fuera:
        criticos.append(f"Artefactos smoke fuera de su carpeta: {smoke_fuera[:3]}")
    else:
        lineas.append("Aislamiento smoke          : correcto")

    ruta_rep = REPORTS_DIR / "validacion_pipeline.txt"
    with open(ruta_rep, "w", encoding="utf-8") as fh:
        fh.write("VALIDACION DEL PIPELINE DE MODELADO - LAB 4 PARTE 2\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        for l in lineas:
            fh.write(l + "\n")
        fh.write(f"\nAvisos  : {len(avisos)}\n")
        for a in avisos:
            fh.write(f"  - {a}\n")
        fh.write(f"\nCriticos: {len(criticos)}\n")
        for c in criticos:
            fh.write(f"  - {c}\n")
        fh.write("\nRESULTADO: " + ("FALLIDA" if criticos else "CORRECTA") + "\n")

    for l in lineas:
        LOGGER.info("  %s", l)
    for a in avisos:
        LOGGER.warning("AVISO: %s", a)
    for c in criticos:
        LOGGER.error("CRITICO: %s", c)
    LOGGER.info("")
    LOGGER.info("Reporte: %s", ruta_rep.relative_to(ROOT))
    if criticos:
        LOGGER.error("")
        LOGGER.error("VALIDACION FALLIDA (%d problemas criticos)", len(criticos))
        return 1
    LOGGER.info("")
    LOGGER.info("VALIDACION CORRECTA")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def construir_parser():
    p = argparse.ArgumentParser(
        prog="modelos_parte2.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Modelado y evaluacion del Laboratorio 4, Parte 2 (Ejercicios 4-7).\n\n"
            "Regresion Logistica, Random Forest y XGBoost sobre el dataset real de\n"
            "3.75 millones de observaciones Sentinel-2 L1C. Incluye validacion\n"
            "aleatoria 70/30, espacial por bloques de 1 km, temporal cronologica y\n"
            "generalizacion entre lagos.\n\n"
            "No usa datos sinteticos y no permite variables con fuga de informacion."),
        epilog=(
            "PERFILES\n"
            "  smoke     muestra pequena; SOLO para comprobar que el pipeline corre.\n"
            "            Sus resultados NUNCA son conclusiones del laboratorio.\n"
            "  standard  configuracion reproducible del laboratorio (recomendada).\n"
            "  full      conjunto completo, mayor costo computacional.\n\n"
            "EJEMPLOS\n"
            "  python modelos_parte2.py --dry-run --profile standard\n"
            "  python modelos_parte2.py --all --profile standard --n-jobs 15\n"
            "  python modelos_parte2.py --validate\n\n"
            "CODIGOS DE SALIDA\n"
            "  0 correcto | 1 errores criticos | 2 falta una dependencia o un requisito\n"))

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument("--dry-run", action="store_true",
                      help="Planifica y estima costos sin entrenar.")
    modo.add_argument("--train-random", action="store_true",
                      help="Ejercicios 4 y 5: division 70/30 y evaluacion.")
    modo.add_argument("--validate-spatial", action="store_true",
                      help="Ejercicio 6: validacion espacial por bloques de 1 km.")
    modo.add_argument("--validate-temporal", action="store_true",
                      help="Validacion temporal cronologica (rubrica).")
    modo.add_argument("--cross-lake", action="store_true",
                      help="Ejercicio 7: generalizacion entre lagos.")
    modo.add_argument("--all", action="store_true",
                      help="Ejecuta las cuatro etapas en orden.")
    modo.add_argument("--validate", action="store_true",
                      help="Valida el pipeline y los artefactos.")
    modo.add_argument("--report-only", action="store_true", dest="report_only",
                      help="Regenera los reportes desde los artefactos, sin reentrenar.")

    p.add_argument("--profile", choices=list(PERFILES), default="standard",
                   help="Perfil de ejecucion (por omision: standard).")
    p.add_argument("--models", nargs="+", choices=MODELOS_DISPONIBLES, default=None,
                   help="Modelos a usar (por omision: los tres).")
    p.add_argument("--n-jobs", type=int, default=None, dest="n_jobs",
                   help="Hilos. Por omision: CPUs disponibles menos una.")
    p.add_argument("--force", action="store_true",
                   help="Rehace artefactos aunque ya existan.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Mensajes de depuracion.")
    return p


def main(argv=None):
    parser = construir_parser()
    args = parser.parse_args(argv)
    configurar_logging(args.verbose)
    try:
        if args.dry_run:
            return ejecutar_dry_run(args)
        if args.validate:
            return ejecutar_validate(args)
        if args.train_random:
            return ejecutar_train_random(args)
        if args.validate_spatial:
            return ejecutar_validate_spatial(args)
        if args.validate_temporal:
            return ejecutar_validate_temporal(args)
        if args.cross_lake:
            return ejecutar_cross_lake(args)
        if args.report_only:
            return ejecutar_report_only(args)
        if args.all:
            return ejecutar_all(args)
    except KeyboardInterrupt:
        LOGGER.warning("\nInterrumpido por la usuaria.")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        LOGGER.exception("Error no controlado: %s: %s", type(exc).__name__, exc)
        return 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
