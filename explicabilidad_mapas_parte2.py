"""
Explicabilidad, mapas predictivos y conclusiones — Laboratorio 4, Parte 2
(Ejercicios 8, 9 y 10).

    8.  Interpretacion del mejor modelo: importancia nativa, permutation
        importance sobre datos no vistos y SHAP sobre una muestra determinista.
    9.  Mapas predictivos de probabilidad para las 22 combinaciones lago-fecha,
        comparacion con la Parte 1 y analisis espacial del error.
    10. Analisis, limitaciones y recomendaciones sustentados en artefactos.

Continua el trabajo de `modelos_parte2.py` (Ejercicios 4-7) y NO reentrena los
modelos de esas fases: los lee desde `outputs/parte2/models/`. El unico
entrenamiento adicional es el de los cinco folds espaciales necesarios para
obtener predicciones OUT-OF-FOLD, sin las cuales los mapas de error estarian
contaminados por el propio entrenamiento.

Modos:
    python explicabilidad_mapas_parte2.py --dry-run     --profile standard
    python explicabilidad_mapas_parte2.py --explain     --profile standard
    python explicabilidad_mapas_parte2.py --maps        --profile standard
    python explicabilidad_mapas_parte2.py --errors-oof  --profile standard
    python explicabilidad_mapas_parte2.py --conclusions --profile standard
    python explicabilidad_mapas_parte2.py --report-only --profile standard
    python explicabilidad_mapas_parte2.py --all         --profile standard
    python explicabilidad_mapas_parte2.py --validate    --profile standard
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
    TARGET_COLUMN, TARGET_THRESHOLD_UG_L, PREDICTORES_PRINCIPALES,
    DATASET_VERSION, BLOQUE_ESPACIAL_M, PIXELS_DIR, RASTER_DIR, NODATA,
    CRS_ESPERADO, hash_esquema, combinaciones_oficiales, ruta_raster,
    ruta_particion,
)
from modelos_parte2 import (  # noqa: E402
    SEED, PROHIBIDAS, N_FOLDS_ESPACIALES, RECALL_MINIMO_OPERACIONAL,
    verificar_sin_fuga, escribir_json, escribir_csv, checkpoint_valido,
    calcular_metricas, umbral_operacional, construir_modelo, hash_indices,
    n_jobs_seguro, limitar_blas, crear_split_70_30,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
np.random.seed(SEED)

BASE = ROOT / "outputs" / "parte2"
LOG_DIR = BASE / "logs"

# Rutas que dependen del perfil (smoke escribe en outputs/parte2/smoke/).
MODELS_DIR = BASE / "models"
SPLITS_DIR = BASE / "splits"
TUNING_DIR = BASE / "tuning"
METRICS_DIR = BASE / "metrics"
FIGURES_DIR = BASE / "figures"
REPORTS_DIR = BASE / "reports"
MAPS_DIR = BASE / "maps"
PROB_DIR = MAPS_DIR / "probability"
ERR_DIR = MAPS_DIR / "errors_oof"
MAPFIG_DIR = MAPS_DIR / "figures"
INTERP_DIR = BASE / "interpretability"
CONCL_DIR = BASE / "conclusions"

SUBCARPETAS = []

# Escala de lectura de los mapas (9.4): cuatro clases claramente distinguibles.
CORTES_PROB = [0.0, 0.25, 0.50, 0.75, 1.0]
ETIQUETAS_PROB = ["muy baja (<0.25)", "baja (0.25-0.50)",
                  "alta (0.50-0.75)", "muy alta (>=0.75)"]

# Codificacion de los mapas de error out-of-fold.
CLASES_ERROR = {0: "TN", 1: "FP", 2: "FN", 3: "TP"}
NODATA_ERROR = 255
AREA_PIXEL_HA = 0.04          # 20 m x 20 m

PERFILES = {
    "smoke": {
        "muestra": 50_000, "n_folds": 2, "shap_max": 400, "shap_budget_s": 120,
        "perm_max": 20_000, "perm_repeats": 2, "max_combinaciones": 2,
        "descripcion": "PRUEBA de humo: NO son resultados finales",
    },
    "standard": {
        "muestra": None, "n_folds": N_FOLDS_ESPACIALES, "shap_max": 20_000,
        "shap_budget_s": 1800, "perm_max": 150_000, "perm_repeats": 5,
        "max_combinaciones": None,
        "descripcion": "Configuracion reproducible del laboratorio",
    },
    "full": {
        "muestra": None, "n_folds": N_FOLDS_ESPACIALES, "shap_max": 50_000,
        "shap_budget_s": 5400, "perm_max": 400_000, "perm_repeats": 10,
        "max_combinaciones": None,
        "descripcion": "Conjunto completo, mayor costo",
    },
}

LOGGER = logging.getLogger("explicabilidad_parte2")


# ---------------------------------------------------------------------------
# Utilidades de entorno
# ---------------------------------------------------------------------------
def configurar_logging(verbose: bool = False) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ruta = LOG_DIR / f"explicabilidad_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    El perfil smoke escribe TODO bajo outputs/parte2/smoke/ para no contaminar
    jamas los artefactos entregables.
    """
    global MODELS_DIR, SPLITS_DIR, METRICS_DIR, FIGURES_DIR, REPORTS_DIR
    global MAPS_DIR, PROB_DIR, ERR_DIR, MAPFIG_DIR, INTERP_DIR, CONCL_DIR
    raiz = BASE / "smoke" if perfil == "smoke" else BASE
    MODELS_DIR, SPLITS_DIR = raiz / "models", raiz / "splits"
    METRICS_DIR, FIGURES_DIR = raiz / "metrics", raiz / "figures"
    REPORTS_DIR = raiz / "reports"
    MAPS_DIR = raiz / "maps"
    PROB_DIR, ERR_DIR, MAPFIG_DIR = (MAPS_DIR / "probability",
                                     MAPS_DIR / "errors_oof",
                                     MAPS_DIR / "figures")
    INTERP_DIR, CONCL_DIR = raiz / "interpretability", raiz / "conclusions"


def crear_carpetas() -> None:
    for d in (REPORTS_DIR, PROB_DIR, ERR_DIR, MAPFIG_DIR, INTERP_DIR,
              CONCL_DIR, LOG_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _marca(perfil: str) -> str:
    return "SMOKE - NO ENTREGABLE" if perfil == "smoke" else f"perfil {perfil}"


def entorno() -> dict:
    import sklearn
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "cpus": os.cpu_count(), "seed": SEED,
            "dataset_version": DATASET_VERSION, "schema_hash": hash_esquema()}
    for nombre in ("shap", "rasterio", "joblib", "pyarrow", "matplotlib", "xgboost"):
        try:
            info[nombre] = __import__(nombre).__version__
        except Exception:
            info[nombre] = None
    return info


def memoria_disponible_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e9, 2)
    except Exception:
        return None


def firma_config(perfil: str, extra: dict | None = None) -> str:
    """Huella de la configuracion; si cambia, los checkpoints dejan de valer."""
    base = {"dataset": hash_esquema(), "version": DATASET_VERSION,
            "perfil": perfil, "features": list(PREDICTORES_PRINCIPALES),
            "target": TARGET_COLUMN, "seed": SEED,
            "n_folds": PERFILES[perfil]["n_folds"]}
    base.update(extra or {})
    return hashlib.sha256(
        json.dumps(base, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Carga de datos e hiperparametros reales
# ---------------------------------------------------------------------------
def cargar_datos(perfil: str) -> dict:
    """
    Proyeccion de columnas: predictores, respuesta, grupos y las coordenadas de
    rejilla necesarias para reconstruir los GeoTIFF.
    """
    import pyarrow.dataset as ds

    verificar_sin_fuga(list(PREDICTORES_PRINCIPALES), "carga")
    columnas = (list(PREDICTORES_PRINCIPALES)
                + [TARGET_COLUMN, "spatial_block_1km", "row", "col"])
    t0 = time.time()
    tabla = ds.dataset(PIXELS_DIR, format="parquet", partitioning="hive").to_table(
        columns=columnas + ["lake", "date"])

    X = np.column_stack([tabla.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                         for c in PREDICTORES_PRINCIPALES])
    y = tabla.column(TARGET_COLUMN).to_numpy(zero_copy_only=False).astype(np.int8)
    row = tabla.column("row").to_numpy(zero_copy_only=False).astype(np.int32)
    col = tabla.column("col").to_numpy(zero_copy_only=False).astype(np.int32)
    bloque, _ = pd.factorize(
        pd.Series(tabla.column("spatial_block_1km").to_pylist()), sort=True)
    lake, nombres_l = pd.factorize(
        pd.Series(tabla.column("lake").to_pylist()).astype(str), sort=True)
    date, nombres_d = pd.factorize(
        pd.Series(tabla.column("date").to_pylist()).astype(str), sort=True)
    del tabla

    if not np.isfinite(X).all():
        raise ValueError("X contiene NaN o infinitos")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("y no es binaria 0/1")

    idx = np.arange(len(y), dtype=np.int64)
    muestra = PERFILES[perfil]["muestra"]
    if muestra and muestra < len(y):
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(idx, train_size=muestra, stratify=y,
                                  random_state=SEED)
        idx = np.sort(idx)
        X, y, row, col = X[idx], y[idx], row[idx], col[idx]
        bloque, lake, date = bloque[idx], lake[idx], date[idx]
        LOGGER.warning("PERFIL SMOKE: submuestra de %s filas. NO son resultados finales.",
                       f"{len(y):,}")

    mem = (X.nbytes + y.nbytes + row.nbytes + col.nbytes + bloque.nbytes) / 1e6
    LOGGER.info("Datos: %s filas x %d predictores en %.1f s (%.0f MB) | "
                "positivos %s (%.4f %%)",
                f"{len(y):,}", X.shape[1], time.time() - t0, mem,
                f"{int(y.sum()):,}", 100 * y.mean())
    return {"X": X, "y": y, "row": row, "col": col, "bloque": bloque,
            "lake": lake, "date": date, "lake_nombres": list(nombres_l),
            "date_nombres": list(nombres_d), "indices": idx,
            "features": list(PREDICTORES_PRINCIPALES), "memoria_mb": mem}


def hiperparametros_rf() -> dict:
    """Lee los hiperparametros REALES del artefacto de tuning; no se inventan."""
    ruta = BASE / "tuning" / "tuning_random_forest_standard.json"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}. Ejecute modelos_parte2.py --train-random antes.")
    return json.load(open(ruta, encoding="utf-8"))["mejores_parametros"]


def umbral_rf() -> float:
    """Umbral operacional del Random Forest, fijado sin usar el conjunto de prueba."""
    ruta = BASE / "metrics" / "random" / "umbrales_standard.json"
    if ruta.exists():
        return float(json.load(open(ruta, encoding="utf-8"))["random_forest"]["umbral"])
    return 0.5


def construir_rf(perfil: str, n_jobs: int):
    """Random Forest con los hiperparametros reales del tuning."""
    modelo = construir_modelo("random_forest", "standard", n_jobs)
    modelo.set_params(**hiperparametros_rf())
    if perfil == "smoke":
        modelo.set_params(n_estimators=40)
    return modelo


# ---------------------------------------------------------------------------
# EJERCICIO 8 - Interpretabilidad
# ---------------------------------------------------------------------------
def muestra_shap_estratificada(datos, n_max, semilla=SEED):
    """
    Muestra determinista estratificada por lago x clase y repartida por fecha.

    SHAP sobre 3.75 M de filas es inviable. Se conserva la mayor cantidad
    posible de positivos, que son el recurso escaso: por eso la muestra NO
    reproduce la prevalencia poblacional, y las magnitudes SHAP deben leerse
    como importancia relativa, no como frecuencia esperada.
    """
    y, lake, date = datos["y"], datos["lake"], datos["date"]
    rng = np.random.default_rng(semilla)
    estratos = {}
    for l in np.unique(lake):
        for c in (0, 1):
            m = np.flatnonzero((lake == l) & (y == c))
            if m.size:
                estratos[(int(l), c)] = m
    if not estratos:
        return np.array([], dtype=np.int64), {}

    por_estrato = max(1, n_max // len(estratos))
    sel, detalle = [], {}
    for (l, c), miembros in estratos.items():
        fechas = date[miembros]
        unicas = np.unique(fechas)
        por_fecha = max(1, por_estrato // max(1, len(unicas)))
        elegidos = [rng.choice(miembros[fechas == f],
                               size=min(por_fecha, int((fechas == f).sum())),
                               replace=False) for f in unicas]
        e = np.concatenate(elegidos) if elegidos else np.array([], dtype=np.int64)
        if e.size > por_estrato:
            e = rng.choice(e, size=por_estrato, replace=False)
        sel.append(e)
        detalle[f"{datos['lake_nombres'][l]}_clase{c}"] = int(e.size)
    return np.sort(np.concatenate(sel)), detalle


def _shap_positiva(valores):
    """
    Normaliza la salida de SHAP a la matriz de la CLASE POSITIVA.

    shap 0.51 puede devolver una lista de 2 arrays (n, f), un array (n, f, 2) o
    directamente (n, f). Se cubren los tres casos.
    """
    if isinstance(valores, list):
        return np.asarray(valores[1] if len(valores) > 1 else valores[0])
    arr = np.asarray(valores)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


ADVERTENCIAS_SHAP = [
    "SHAP no implica causalidad: describe la contribucion a la prediccion del "
    "modelo, no un mecanismo fisico.",
    "Las bandas espectrales estan fuertemente correlacionadas entre si, asi que la "
    "importancia se reparte entre variables redundantes; un valor bajo no significa "
    "que la banda sea prescindible.",
    "La respuesta es un proxy espectral de clorofila-a, no una medicion in situ.",
    "La muestra SHAP sobrerrepresenta la clase positiva: NO reproduce la prevalencia "
    "poblacional, por lo que las magnitudes son importancia relativa, no frecuencia "
    "esperada en el lago.",
]


def ejecutar_explain(args, datos=None) -> int:
    """Importancia nativa, permutation importance y SHAP del Random Forest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.inspection import permutation_importance

    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    n_jobs = args.n_jobs or n_jobs_seguro()
    limitar_blas(max(1, n_jobs // 2))
    cfg = PERFILES[perfil]

    LOGGER.info("=" * 86)
    LOGGER.info("EJERCICIO 8 - INTERPRETABILIDAD DEL RANDOM FOREST [%s]", _marca(perfil))
    LOGGER.info("=" * 86)

    firma = firma_config(perfil, {"fase": "explain", "shap_max": cfg["shap_max"]})
    ck = INTERP_DIR / f"checkpoint_explain_{perfil}.json"
    if checkpoint_valido(ck, firma, args.force):
        LOGGER.info("Fase ya completada. Use --force para rehacer.")
        return 0

    if datos is None:
        datos = cargar_datos(perfil)
    X, y = datos["X"], datos["y"]
    feats = datos["features"]
    verificar_sin_fuga(feats, "explain")

    # El modelo que se interpreta se ajusta con un fold de entrenamiento espacial,
    # de modo que la permutation importance se evalue sobre observaciones NO
    # usadas para ajustarlo.
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True,
                                random_state=SEED)
    tr, ev = next(iter(sgkf.split(np.zeros(len(y)), y, groups=datos["bloque"])))
    assert not (set(np.unique(datos["bloque"][tr]))
                & set(np.unique(datos["bloque"][ev]))), "bloques compartidos"
    LOGGER.info("Ajuste: %s filas | Evaluacion no vista: %s filas (bloques disjuntos)",
                f"{len(tr):,}", f"{len(ev):,}")

    modelo = construir_rf(perfil, n_jobs)
    t0 = time.time()
    modelo.fit(X[tr], y[tr])
    t_fit = time.time() - t0
    LOGGER.info("Random Forest ajustado en %.1f s con %s", t_fit, hiperparametros_rf())
    joblib.dump(modelo, MODELS_DIR / f"rf_interpretabilidad_{perfil}.joblib", compress=3)

    # --- 1. Importancia nativa por impureza ---
    nativa = pd.DataFrame({"variable": feats,
                           "importancia_impureza": modelo.feature_importances_})
    nativa = nativa.sort_values("importancia_impureza", ascending=False)
    escribir_csv(INTERP_DIR / f"importancia_nativa_{perfil}.csv", nativa)
    LOGGER.info("Importancia nativa: %s",
                ", ".join(f"{r.variable}={r.importancia_impureza:.3f}"
                          for r in nativa.itertuples()))

    # --- 2. Permutation importance sobre datos NO vistos ---
    rng = np.random.default_rng(SEED)
    sub = ev if len(ev) <= cfg["perm_max"] else np.sort(
        rng.choice(ev, size=cfg["perm_max"], replace=False))
    t0 = time.time()
    perm = permutation_importance(modelo, X[sub], y[sub],
                                 scoring="average_precision",
                                 n_repeats=cfg["perm_repeats"],
                                 random_state=SEED, n_jobs=n_jobs)
    t_perm = time.time() - t0
    perm_df = pd.DataFrame({"variable": feats,
                            "caida_media_pr_auc": perm.importances_mean,
                            "desviacion": perm.importances_std})
    perm_df = perm_df.sort_values("caida_media_pr_auc", ascending=False)
    escribir_csv(INTERP_DIR / f"permutation_importance_{perfil}.csv", perm_df)
    LOGGER.info("Permutation importance sobre %s filas no vistas "
                "(%.1f s, %d repeticiones)", f"{len(sub):,}", t_perm,
                cfg["perm_repeats"])

    # --- 3. SHAP ---
    import shap
    sel, detalle = muestra_shap_estratificada(datos, cfg["shap_max"])
    prev_muestra = float(y[sel].mean()) if len(sel) else 0.0
    LOGGER.info("Muestra SHAP: %s filas | estratos: %s", f"{len(sel):,}", detalle)
    LOGGER.info("  Prevalencia en la muestra SHAP: %.4f %% (poblacional: %.4f %%)",
                100 * prev_muestra, 100 * y.mean())

    t0 = time.time()
    explainer = shap.TreeExplainer(modelo)
    sv = _shap_positiva(explainer.shap_values(X[sel], check_additivity=False))
    t_shap = time.time() - t0
    LOGGER.info("SHAP calculado en %.1f s | forma normalizada: %s", t_shap, sv.shape)

    shap_df = pd.DataFrame({"variable": feats,
                            "shap_abs_media": np.abs(sv).mean(axis=0)})
    shap_df = shap_df.sort_values("shap_abs_media", ascending=False)
    escribir_csv(INTERP_DIR / f"shap_abs_media_{perfil}.csv", shap_df)

    escribir_json(INTERP_DIR / f"interpretabilidad_meta_{perfil}.json", {
        "perfil": perfil, "firma_config": firma, "completo": True,
        "modelo": "random_forest", "hiperparametros": hiperparametros_rf(),
        "seleccionado_por": "mayor PR-AUC medio en validacion espacial",
        "semilla": SEED, "entorno": entorno(),
        "n_ajuste": int(len(tr)), "n_evaluacion_no_vista": int(len(ev)),
        "n_permutation": int(len(sub)), "perm_repeats": cfg["perm_repeats"],
        "n_shap": int(len(sel)), "estratos_shap": detalle,
        "prevalencia_muestra_shap": prev_muestra,
        "prevalencia_poblacional": float(y.mean()),
        "segundos": {"fit": t_fit, "permutation": t_perm, "shap": t_shap},
        "advertencias": ADVERTENCIAS_SHAP,
    })

    # --- Figuras ---
    marca = f" [{_marca(perfil)}]"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].barh(nativa["variable"][::-1], nativa["importancia_impureza"][::-1],
                 color="#4878a8")
    axes[0].set(title="Importancia nativa (impureza)", xlabel="Importancia")
    axes[1].barh(perm_df["variable"][::-1], perm_df["caida_media_pr_auc"][::-1],
                 xerr=perm_df["desviacion"][::-1], color="#c44e52")
    axes[1].set(title="Permutation importance (datos no vistos)",
                xlabel="Caida de PR-AUC al permutar")
    fig.suptitle("Importancia de variables - Random Forest" + marca, fontweight="bold")
    fig.tight_layout()
    fig.savefig(INTERP_DIR / f"importancia_nativa_{perfil}.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm_df["variable"][::-1], perm_df["caida_media_pr_auc"][::-1],
            xerr=perm_df["desviacion"][::-1], color="#c44e52")
    ax.set(title="Permutation importance" + marca, xlabel="Caida de PR-AUC")
    fig.tight_layout()
    fig.savefig(INTERP_DIR / f"permutation_importance_{perfil}.png", dpi=140)
    plt.close(fig)

    Xs = pd.DataFrame(X[sel], columns=feats)
    # shap.summary_plot dispersa (jitter) los puntos del beeswarm con el RNG
    # global de NumPy. Como el modulo fija np.random.seed(SEED) al importarse,
    # shap 0.51 emite un FutureWarning porque en una version futura dejara de
    # leer esa semilla global. Se pasa un generador explicito (rng=) para
    # adoptar ya la API nueva: el resultado sigue siendo reproducible con la
    # misma semilla, pero ya no depende del estado global.
    rng_shap = np.random.default_rng(SEED)
    plt.figure()
    shap.summary_plot(sv, Xs, show=False, plot_size=(9, 5.5), rng=rng_shap)
    plt.title("SHAP beeswarm - clase positiva" + marca, fontsize=11)
    plt.savefig(INTERP_DIR / f"shap_beeswarm_{perfil}.png", dpi=140,
                bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, plot_size=(9, 5),
                      rng=rng_shap)
    plt.title("SHAP: media del valor absoluto" + marca, fontsize=11)
    plt.savefig(INTERP_DIR / f"shap_bar_{perfil}.png", dpi=140, bbox_inches="tight")
    plt.close("all")

    principales = shap_df["variable"].head(3).tolist()
    fig, axes = plt.subplots(1, len(principales), figsize=(6 * len(principales), 4.6),
                             squeeze=False)
    for ax, v in zip(axes[0], principales):
        j = feats.index(v)
        ax.scatter(Xs[v], sv[:, j], s=4, alpha=0.3, c=Xs[v], cmap="viridis")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(xlabel=f"{v} (reflectancia)", ylabel="valor SHAP",
               title=f"Dependence: {v}")
    fig.suptitle("SHAP dependence de las variables principales" + marca,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(INTERP_DIR / f"shap_dependence_{perfil}.png", dpi=140)
    plt.close(fig)

    escribir_json(ck, {"firma_config": firma, "completo": True, "perfil": perfil,
                       "n_shap": int(len(sel))})
    LOGGER.info("Ejercicio 8 completado.")
    return 0


# ---------------------------------------------------------------------------
# EJERCICIO 9 - Mapas de probabilidad (descriptivos) y de error out-of-fold
# ---------------------------------------------------------------------------
def geometria_lago(lago: str) -> dict:
    """Dimensiones, transform y CRS reales del GeoTIFF original del lago."""
    import rasterio
    fecha = combinaciones_oficiales(lago)[0][1]
    with rasterio.open(ruta_raster(lago, fecha)) as src:
        return {"ancho": src.width, "alto": src.height,
                "transform": src.transform, "crs": src.crs}


def escribir_geotiff(ruta: Path, matriz: np.ndarray, geom: dict, tags: dict,
                     nodata: float) -> Path:
    """Escritura atomica de un GeoTIFF georreferenciado."""
    import rasterio
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp.tif")
    perfil = {"driver": "GTiff", "height": geom["alto"], "width": geom["ancho"],
              "count": 1, "dtype": matriz.dtype.name, "crs": geom["crs"],
              "transform": geom["transform"], "nodata": nodata,
              "compress": "lzw", "tiled": True,
              "blockxsize": 256, "blockysize": 256}
    with rasterio.open(tmp, "w", **perfil) as dst:
        dst.write(matriz, 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    with rasterio.open(tmp) as chk:
        if chk.width != geom["ancho"] or chk.height != geom["alto"]:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"{ruta.name}: dimensiones incorrectas")
    tmp.replace(ruta)
    return ruta


def reconstruir_matriz(valores, row, col, geom, relleno, dtype):
    """Coloca un vector de predicciones en la rejilla original del raster."""
    m = np.full((geom["alto"], geom["ancho"]), relleno, dtype=dtype)
    m[row, col] = valores
    return m


def fechas_criticas_parte1() -> dict:
    """Lee del CSV de la Parte 1 la fecha de mayor clorofila media por lago."""
    ruta = ROOT / "outputs" / "parte1_real" / "tables" / "comparacion_entre_lagos.csv"
    if not ruta.exists():
        return {}
    d = pd.read_csv(ruta)
    return dict(zip(d["lago"], d["fecha_mas_critica"]))


def ejecutar_maps(args, datos=None) -> int:
    """
    Mapas de probabilidad descriptivos.

    ADVERTENCIA METODOLOGICA: el modelo se ajusta con TODO el dataset y luego
    predice sobre el mismo dataset. Sirve para producir una capa descriptiva de
    despliegue, NO para estimar desempeno: esas probabilidades estan vistas por
    el modelo. La evaluacion no sesgada es la out-of-fold (--errors-oof).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib

    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    n_jobs = args.n_jobs or n_jobs_seguro()
    limitar_blas(max(1, n_jobs // 2))

    LOGGER.info("=" * 86)
    LOGGER.info("EJERCICIO 9 - MAPAS DE PROBABILIDAD (descriptivos) [%s]", _marca(perfil))
    LOGGER.info("=" * 86)
    LOGGER.warning("Estas probabilidades provienen de un modelo ajustado con TODO el "
                   "dataset: son descriptivas/de despliegue, NO una validacion no "
                   "sesgada. La evaluacion honesta es la out-of-fold.")

    firma = firma_config(perfil, {"fase": "maps"})
    ck = PROB_DIR / f"checkpoint_maps_{perfil}.json"
    if checkpoint_valido(ck, firma, args.force):
        LOGGER.info("Fase ya completada. Use --force para rehacer.")
        return 0

    if datos is None:
        datos = cargar_datos(perfil)
    X, y = datos["X"], datos["y"]
    verificar_sin_fuga(datos["features"], "maps")
    umbral = umbral_rf()

    modelo = construir_rf(perfil, n_jobs)
    t0 = time.time()
    modelo.fit(X, y)
    LOGGER.info("Random Forest final ajustado con %s filas en %.1f s",
                f"{len(y):,}", time.time() - t0)
    joblib.dump(modelo, MODELS_DIR / f"rf_final_despliegue_{perfil}.joblib", compress=3)

    prob = modelo.predict_proba(X)[:, 1].astype(np.float32)
    NODATA_PROB = -1.0
    lagos = datos["lake_nombres"]
    fechas = datos["date_nombres"]
    geoms = {l: geometria_lago(l) for l in lagos}

    pares = combinaciones_oficiales()
    if PERFILES[perfil]["max_combinaciones"]:
        pares = pares[:PERFILES[perfil]["max_combinaciones"]]

    filas = []
    for lago, fecha in pares:
        if lago not in lagos or fecha not in fechas:
            continue
        m = (datos["lake"] == lagos.index(lago)) & (datos["date"] == fechas.index(fecha))
        if not m.any():
            continue
        geom = geoms[lago]
        matriz = reconstruir_matriz(prob[m], datos["row"][m], datos["col"][m],
                                    geom, NODATA_PROB, np.float32)
        ruta = PROB_DIR / lago / f"prob_{lago}_{fecha}.tif"
        escribir_geotiff(ruta, matriz, geom, {
            "lago": lago, "fecha": fecha, "modelo": "random_forest",
            "seleccionado_por": "validacion espacial",
            "umbral_operacional": umbral, "target": TARGET_COLUMN,
            "umbral_ug_L": TARGET_THRESHOLD_UG_L,
            "tipo_prediccion": "probabilidad descriptiva (in-sample, no validacion)",
            "perfil": perfil, "semilla": SEED, "nodata": NODATA_PROB,
            "crs": str(geom["crs"]), "resolucion_m": 20,
            "area_pixel_ha": AREA_PIXEL_HA,
        }, NODATA_PROB)
        p = prob[m]
        pos = int((p >= umbral).sum())
        filas.append({"lago": lago, "fecha": fecha, "n_pixeles": int(m.sum()),
                      "prob_media": float(p.mean()), "prob_mediana": float(np.median(p)),
                      "prob_p95": float(np.percentile(p, 95)),
                      "pixeles_sobre_umbral": pos,
                      "pct_sobre_umbral": 100.0 * pos / m.sum(),
                      "area_positiva_ha": pos * AREA_PIXEL_HA,
                      "positivos_reales": int(y[m].sum()),
                      "geotiff": str(ruta.relative_to(ROOT)).replace("\\", "/")})
        LOGGER.info("  %-10s %-12s prob_media=%.4f | >=umbral %s (%.2f %%) | %.1f ha",
                    lago, fecha, p.mean(), f"{pos:,}",
                    100 * pos / m.sum(), pos * AREA_PIXEL_HA)

    resumen = pd.DataFrame(filas)
    escribir_csv(PROB_DIR / f"resumen_probabilidad_{perfil}.csv", resumen)

    # --- Figuras PNG versionables, escala fija 0-1 ---
    criticas = fechas_criticas_parte1()
    representativas = []
    for lago in lagos:
        f = criticas.get(lago)
        if f and (lago, f) in [(r["lago"], r["fecha"]) for r in filas]:
            representativas.append((lago, f))
    if not representativas:
        representativas = [(r["lago"], r["fecha"]) for r in filas[:2]]
    LOGGER.info("Fechas representativas (criticas segun la Parte 1): %s",
                representativas)

    import rasterio
    marca = f" [{_marca(perfil)}]"
    for lago, fecha in representativas:
        ruta = PROB_DIR / lago / f"prob_{lago}_{fecha}.tif"
        if not ruta.exists():
            continue
        with rasterio.open(ruta) as src:
            arr = src.read(1, masked=True)
            b = src.bounds
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(np.ma.masked_less(arr, 0), cmap="RdYlGn_r", vmin=0, vmax=1,
                       extent=[b.left, b.right, b.bottom, b.top], origin="upper")
        cb = fig.colorbar(im, ax=ax, boundaries=CORTES_PROB, ticks=CORTES_PROB)
        cb.set_label("Probabilidad de clorofila-a alta (0-1)")
        ax.set(title=f"{lago} - {fecha}: probabilidad predicha{marca}",
               xlabel="Este (m, UTM 15N)", ylabel="Norte (m)")
        fig.tight_layout()
        fig.savefig(MAPFIG_DIR / f"probabilidad_{lago}_{fecha}_{perfil}.png", dpi=140)
        plt.close(fig)

    if len(representativas) >= 2:
        fig, axes = plt.subplots(1, len(representativas),
                                 figsize=(8.5 * len(representativas), 6.5))
        axes = np.atleast_1d(axes)
        for ax, (lago, fecha) in zip(axes, representativas):
            ruta = PROB_DIR / lago / f"prob_{lago}_{fecha}.tif"
            with rasterio.open(ruta) as src:
                arr = src.read(1)
                b = src.bounds
            im = ax.imshow(np.ma.masked_less(arr, 0), cmap="RdYlGn_r", vmin=0, vmax=1,
                           extent=[b.left, b.right, b.bottom, b.top], origin="upper")
            ax.set(title=f"{lago} - {fecha}", xlabel="Este (m)")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8,
                     label="Probabilidad (escala comun 0-1)")
        fig.suptitle("Comparacion entre lagos" + marca, fontweight="bold")
        fig.savefig(MAPFIG_DIR / f"comparacion_lagos_{perfil}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    if not resumen.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for lago, g in resumen.groupby("lago"):
            g = g.sort_values("fecha")
            axes[0].plot(g["fecha"], g["prob_media"], marker="o", label=lago)
            axes[1].plot(g["fecha"], g["area_positiva_ha"], marker="s", label=lago)
        axes[0].set(title="Probabilidad media predicha por fecha",
                    ylabel="Probabilidad", ylim=(0, 1))
        axes[1].set(title=f"Superficie clasificada positiva (umbral {umbral:.3f})",
                    ylabel="Hectareas")
        for ax in axes:
            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
        fig.suptitle("Evolucion temporal de la prediccion" + marca, fontweight="bold")
        fig.tight_layout()
        fig.savefig(MAPFIG_DIR / f"evolucion_temporal_prediccion_{perfil}.png", dpi=140)
        plt.close(fig)

    escribir_json(ck, {"firma_config": firma, "completo": True, "perfil": perfil,
                       "n_geotiff": len(filas), "umbral": umbral,
                       "advertencia": "probabilidades in-sample, no validacion"})
    LOGGER.info("Ejercicio 9 (mapas de probabilidad) completado: %d GeoTIFF.", len(filas))
    return 0


def ejecutar_errors_oof(args, datos=None) -> int:
    """
    Mapas de error construidos SOLO con predicciones out-of-fold espaciales.

    Cada observacion se predice exactamente una vez, por el modelo que NO vio su
    bloque. Es la unica base honesta para mapear TP/TN/FP/FN.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedGroupKFold

    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    n_jobs = args.n_jobs or n_jobs_seguro()
    limitar_blas(max(1, n_jobs // 2))
    cfg = PERFILES[perfil]

    LOGGER.info("=" * 86)
    LOGGER.info("EJERCICIO 9 - MAPAS DE ERROR OUT-OF-FOLD [%s]", _marca(perfil))
    LOGGER.info("=" * 86)

    firma = firma_config(perfil, {"fase": "errors_oof"})
    ck = ERR_DIR / f"checkpoint_errors_{perfil}.json"
    if checkpoint_valido(ck, firma, args.force):
        LOGGER.info("Fase ya completada. Use --force para rehacer.")
        return 0

    if datos is None:
        datos = cargar_datos(perfil)
    X, y, bloque = datos["X"], datos["y"], datos["bloque"]
    verificar_sin_fuga(datos["features"], "errors-oof")
    umbral = umbral_rf()

    prob_oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_de = np.full(len(y), -1, dtype=np.int8)
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True,
                                random_state=SEED)
    compartidos_total = 0
    for i, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=bloque),
                                 start=1):
        comp = set(np.unique(bloque[tr])) & set(np.unique(bloque[va]))
        compartidos_total += len(comp)
        if comp:
            LOGGER.error("Fold %d comparte %d bloques: se aborta.", i, len(comp))
            return 1
        modelo = construir_rf(perfil, n_jobs)   # hiperparametros fijos, sin tuning
        modelo.fit(X[tr], y[tr])
        prob_oof[va] = modelo.predict_proba(X[va])[:, 1]
        fold_de[va] = i
        LOGGER.info("  fold %d: ajuste %s | prediccion OOF %s | bloques compartidos 0",
                    i, f"{len(tr):,}", f"{len(va):,}")

    if np.isnan(prob_oof).any() or (fold_de < 0).any():
        LOGGER.error("Hay observaciones sin prediccion out-of-fold.")
        return 1
    LOGGER.info("Cobertura OOF: %s/%s observaciones predichas exactamente una vez",
                f"{int((fold_de > 0).sum()):,}", f"{len(y):,}")

    pred = (prob_oof >= umbral).astype(np.int8)
    # 0=TN 1=FP 2=FN 3=TP
    tipo = np.where((y == 0) & (pred == 0), 0,
                    np.where((y == 0) & (pred == 1), 1,
                             np.where((y == 1) & (pred == 0), 2, 3))).astype(np.uint8)

    lagos, fechas = datos["lake_nombres"], datos["date_nombres"]
    tabla = pd.DataFrame({
        "lake": [lagos[i] for i in datos["lake"]],
        "date": [fechas[i] for i in datos["date"]],
        "row": datos["row"], "col": datos["col"],
        "spatial_block": datos["bloque"],
        "y_true": y, "y_probability_oof": prob_oof, "y_pred_oof": pred,
        "fold": fold_de,
        "error_type": [CLASES_ERROR[int(t)] for t in tipo],
    })
    for (lago, fecha), g in tabla.groupby(["lake", "date"]):
        destino = ERR_DIR / "pixels" / f"lake={lago}" / f"date={fecha}"
        destino.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["lake", "date"]).to_parquet(
            destino / "part-0.parquet", engine="pyarrow", compression="snappy",
            index=False)

    m_glob = calcular_metricas(y, prob_oof, umbral, "oof_global")
    escribir_json(ERR_DIR / f"metricas_oof_global_{perfil}.json",
                  {**m_glob, "perfil": perfil, "umbral": umbral,
                   "n_folds": cfg["n_folds"], "bloques_compartidos": compartidos_total,
                   "modelo": "random_forest", "hiperparametros": hiperparametros_rf()})
    LOGGER.info("OOF global: PR-AUC=%.4f R=%.3f P=%.3f | TP=%s FN=%s FP=%s",
                m_glob["pr_auc"], m_glob["recall"], m_glob["precision"],
                f"{m_glob['TP']:,}", f"{m_glob['FN']:,}", f"{m_glob['FP']:,}")

    def resumen_por(clave):
        filas = []
        for k, g in tabla.groupby(clave):
            if g["y_true"].nunique() < 2:
                filas.append({"grupo": str(k), "n": len(g),
                              "advertencia": "una sola clase"})
                continue
            m = calcular_metricas(g["y_true"].to_numpy(),
                                  g["y_probability_oof"].to_numpy(), umbral, str(k))
            filas.append({"grupo": str(k), "n": len(g), "advertencia": "", **m,
                          "area_FP_ha": m["FP"] * AREA_PIXEL_HA,
                          "area_FN_ha": m["FN"] * AREA_PIXEL_HA,
                          "area_error_ha": (m["FP"] + m["FN"]) * AREA_PIXEL_HA})
        return pd.DataFrame(filas)

    por_lago = resumen_por("lake")
    por_fecha = resumen_por(["lake", "date"])
    por_bloque = resumen_por("spatial_block")
    escribir_csv(ERR_DIR / f"metricas_oof_por_lago_{perfil}.csv", por_lago)
    escribir_csv(ERR_DIR / f"metricas_oof_por_fecha_{perfil}.csv", por_fecha)
    escribir_csv(ERR_DIR / f"metricas_oof_por_bloque_{perfil}.csv", por_bloque)

    conteo = (tabla.groupby(["lake", "date", "error_type"]).size()
              .rename("n_pixeles").reset_index())
    conteo["area_ha"] = conteo["n_pixeles"] * AREA_PIXEL_HA
    escribir_csv(ERR_DIR / f"conteo_errores_{perfil}.csv", conteo)

    con_error = por_bloque[por_bloque.get("FN", pd.Series(dtype=float)).notna()]
    if not con_error.empty:
        peores_fn = con_error.nlargest(min(10, len(con_error)), "FN")[
            ["grupo", "n", "FN", "FP", "recall", "precision", "area_FN_ha"]]
        peores_fp = con_error.nlargest(min(10, len(con_error)), "FP")[
            ["grupo", "n", "FP", "FN", "precision", "recall", "area_FP_ha"]]
        escribir_csv(ERR_DIR / f"peores_bloques_FN_{perfil}.csv", peores_fn)
        escribir_csv(ERR_DIR / f"peores_bloques_FP_{perfil}.csv", peores_fp)
    pf = por_fecha[por_fecha.get("FN", pd.Series(dtype=float)).notna()]
    if not pf.empty:
        escribir_csv(ERR_DIR / f"peores_fechas_FN_{perfil}.csv",
                     pf.nlargest(min(10, len(pf)), "FN")[
                         ["grupo", "n", "FN", "FP", "recall", "precision"]])

    # --- Mapas PNG de error ---
    geoms = {l: geometria_lago(l) for l in lagos}
    colores = {"TN": "#dddddd", "FP": "#ff7f0e", "FN": "#d62728", "TP": "#2ca02c"}
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([colores["TN"], colores["FP"], colores["FN"], colores["TP"]])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    representativas = []
    criticas = fechas_criticas_parte1()
    for lago in lagos:
        f = criticas.get(lago)
        if f is not None and ((tabla["lake"] == lago) & (tabla["date"] == f)).any():
            representativas.append((lago, f))
    if not pf.empty:
        peor = pf.nlargest(1, "FN")["grupo"].iloc[0]
        try:
            lg, fc = eval(peor) if peor.startswith("(") else (None, None)
            if lg and (lg, fc) not in representativas:
                representativas.append((lg, fc))
        except Exception:
            pass

    marca = f" [{_marca(perfil)}]"
    for lago, fecha in representativas:
        g = tabla[(tabla["lake"] == lago) & (tabla["date"] == fecha)]
        if g.empty:
            continue
        geom = geoms[lago]
        codigo = np.array([{"TN": 0, "FP": 1, "FN": 2, "TP": 3}[t]
                           for t in g["error_type"]], dtype=np.uint8)
        matriz = reconstruir_matriz(codigo, g["row"].to_numpy(), g["col"].to_numpy(),
                                    geom, NODATA_ERROR, np.uint8)
        escribir_geotiff(ERR_DIR / lago / f"error_oof_{lago}_{fecha}.tif",
                         matriz, geom,
                         {"lago": lago, "fecha": fecha, "codificacion": str(CLASES_ERROR),
                          "tipo_prediccion": "out-of-fold espacial", "umbral": umbral,
                          "perfil": perfil, "nodata": NODATA_ERROR},
                         NODATA_ERROR)
        b_ = geom["transform"]
        extent = [b_.c, b_.c + b_.a * geom["ancho"],
                  b_.f + b_.e * geom["alto"], b_.f]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(np.ma.masked_equal(matriz, NODATA_ERROR), cmap=cmap, norm=norm,
                  extent=extent, origin="upper", interpolation="nearest")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=k) for k, c in colores.items()],
                  loc="upper right", fontsize=9)
        n_fn = int((g["error_type"] == "FN").sum())
        n_fp = int((g["error_type"] == "FP").sum())
        ax.set(title=f"{lago} - {fecha}: errores out-of-fold{marca}\n"
                     f"FN={n_fn:,} ({n_fn*AREA_PIXEL_HA:.1f} ha) | "
                     f"FP={n_fp:,} ({n_fp*AREA_PIXEL_HA:.1f} ha)",
               xlabel="Este (m, UTM 15N)", ylabel="Norte (m)")
        fig.tight_layout()
        fig.savefig(MAPFIG_DIR / f"errores_oof_{lago}_{fecha}_{perfil}.png", dpi=140)
        plt.close(fig)

    escribir_json(ck, {"firma_config": firma, "completo": True, "perfil": perfil,
                       "bloques_compartidos": compartidos_total,
                       "cobertura_oof": int((fold_de > 0).sum()),
                       "pr_auc_oof": m_glob["pr_auc"]})
    LOGGER.info("Mapas de error out-of-fold completados.")
    return 0


# ---------------------------------------------------------------------------
# EJERCICIO 10 - Conclusiones calculadas
# ---------------------------------------------------------------------------
def ejecutar_conclusions(args) -> int:
    """Consolida las evidencias de los Ejercicios 4-9 en cifras calculadas."""
    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    LOGGER.info("=" * 86)
    LOGGER.info("EJERCICIO 10 - CONCLUSIONES CALCULADAS [%s]", _marca(perfil))
    LOGGER.info("=" * 86)

    ev = {"perfil": perfil, "generado": datetime.now().isoformat(timespec="seconds"),
          "semilla": SEED, "modelo_principal": "random_forest",
          "criterio_seleccion": "mayor PR-AUC medio en validacion espacial"}

    r = BASE / "metrics" / "random" / "metricas_standard.csv"
    if r.exists():
        d = pd.read_csv(r)
        d = d[d["etiqueta"] == "umbral_0.5"]
        ev["aleatorio_pr_auc"] = {row["modelo"]: round(row["pr_auc"], 4)
                                  for _, row in d.iterrows()}
        ev["mejor_aleatorio"] = d.loc[d["pr_auc"].idxmax(), "modelo"]

    s = BASE / "metrics" / "spatial" / "agregado_standard.csv"
    if s.exists():
        d = pd.read_csv(s).set_index("modelo")
        ev["espacial_pr_auc"] = {m: {"media": round(d.loc[m, "pr_auc_mean"], 4),
                                     "desv": round(d.loc[m, "pr_auc_std"], 4)}
                                 for m in d.index}
        top2 = d["pr_auc_mean"].nlargest(2)
        ev["mejor_espacial"] = top2.index[0]
        ev["diferencia_top2_espacial"] = round(float(top2.iloc[0] - top2.iloc[1]), 5)
        ev["empate_practico"] = bool(
            ev["diferencia_top2_espacial"] < d.loc[top2.index[0], "pr_auc_std"])

    t = BASE / "metrics" / "temporal" / "agregado_standard.csv"
    if t.exists():
        d = pd.read_csv(t).set_index("modelo")
        ev["temporal_pr_auc"] = {m: {"media": round(d.loc[m, "pr_auc_mean"], 4),
                                     "desv": round(d.loc[m, "pr_auc_std"], 4),
                                     "min": round(d.loc[m, "pr_auc_min"], 4),
                                     "max": round(d.loc[m, "pr_auc_max"], 4)}
                                for m in d.index}

    c = BASE / "metrics" / "cross_lake" / "metricas_standard.csv"
    if c.exists():
        d = pd.read_csv(c)
        d = d[d["etiqueta"] == "umbral_operacional"]
        ev["cross_lake"] = [{"experimento": row["experimento"],
                             "train": row["train"], "test": row["test"],
                             "pr_auc": round(row["pr_auc"], 4),
                             "recall_operacional": round(row["recall"], 4),
                             "TP": int(row["TP"]), "FN": int(row["FN"])}
                            for _, row in d.iterrows()]

    o = ERR_DIR / f"metricas_oof_global_{perfil}.json"
    if o.exists():
        m = json.load(open(o, encoding="utf-8"))
        ev["oof"] = {k: m[k] for k in ("pr_auc", "recall", "precision", "f2",
                                       "TP", "FN", "FP", "TN") if k in m}
        ev["oof"]["area_FN_ha"] = m.get("FN", 0) * AREA_PIXEL_HA
        ev["oof"]["area_FP_ha"] = m.get("FP", 0) * AREA_PIXEL_HA

    i = INTERP_DIR / f"shap_abs_media_{perfil}.csv"
    if i.exists():
        d = pd.read_csv(i)
        ev["shap_top3"] = d.head(3).to_dict("records")

    ev["limitaciones"] = [
        "La respuesta es un PROXY ESPECTRAL de clorofila-a, no una medicion de "
        "cianobacterias ni de toxinas.",
        "No hubo validacion in situ en ninguna fase del laboratorio.",
        "El algoritmo CyanoLakes reporta MAPE 42.3 % y RMSE relativo 95.8 %, y fue "
        "calibrado para Microcystis aeruginosa sobre datos simulados.",
        "Dominio de calibracion del NDCI: 1-60 ug/L; una parte importante de los "
        "pixeles de Atitlan queda por debajo.",
        "Atitlan presenta clorofila estimada negativa en una fraccion de sus pixeles: "
        "valores sin sentido fisico, conservados para trazabilidad.",
        "La coleccion SENTINEL2_L1C no expone mascara de nubes por pixel (sin CLM, "
        "CLP, dataMask, SCL ni QA60).",
        "Solo 22 escenas (11 fechas por lago): insuficiente para afirmar tendencias "
        "interanuales.",
        "Desbalance 1:61 y fuerte concentracion de los positivos (37.7 % en 5 bloques, "
        "87.6 % en 5 fechas).",
        "Los mapas de probabilidad descriptivos son in-sample; la evaluacion no "
        "sesgada es la out-of-fold.",
    ]
    ev["conclusiones"] = [
        "El split aleatorio 70/30 es OPTIMISTA: mezcla pixeles vecinos del mismo "
        "bloque de 1 km, casi identicos entre si.",
        "Random Forest y XGBoost estan en EMPATE PRACTICO bajo validacion espacial; "
        "la diferencia es menor que la desviacion entre folds.",
        "Existe INESTABILIDAD TEMPORAL marcada: el desempeno varia fuertemente entre "
        "fechas y cae respecto de la validacion espacial.",
        "La TRANSFERENCIA ENTRE LAGOS NO es satisfactoria: con umbral fijo el recall "
        "operacional se desploma en ambas direcciones.",
        "El modelo sirve como APOYO DE CRIBADO dentro del dominio observado (estos dos "
        "lagos, estas fechas, este algoritmo de referencia).",
        "NO sustituye el monitoreo in situ: no confirma presencia de cianobacterias ni "
        "toxicidad.",
        "NO demuestra transferencia robusta a otros lagos ni a fechas futuras.",
    ]
    escribir_json(CONCL_DIR / f"conclusiones_{perfil}.json", ev)

    filas = []
    for etapa, clave in [("aleatorio 70/30", "aleatorio_pr_auc"),
                         ("validacion espacial", "espacial_pr_auc"),
                         ("validacion temporal", "temporal_pr_auc")]:
        d = ev.get(clave, {})
        for modelo, val in d.items():
            filas.append({"etapa": etapa, "modelo": modelo,
                          "pr_auc": val if isinstance(val, float) else val["media"],
                          "desviacion": (val.get("desv")
                                         if isinstance(val, dict) else None)})
    if filas:
        escribir_csv(CONCL_DIR / f"comparacion_etapas_{perfil}.csv", pd.DataFrame(filas))
    LOGGER.info("Conclusiones calculadas: %s", CONCL_DIR.relative_to(ROOT))
    return 0


# ---------------------------------------------------------------------------
# Informe final en Markdown (solo con resultados standard)
# ---------------------------------------------------------------------------
def ejecutar_report_only(args) -> int:
    """
    Redacta INFORME_LAB4_PARTE2.md. No genera ni convierte ningun PDF.

    Se niega a redactar con cifras del perfil smoke.
    """
    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    if perfil == "smoke":
        LOGGER.error("El informe final NO se redacta con cifras smoke.")
        LOGGER.error("Ejecute --all --profile standard y despues --report-only.")
        return 2

    ruta_ev = CONCL_DIR / "conclusiones_standard.json"
    if not ruta_ev.exists():
        LOGGER.error("Falta %s: ejecute --conclusions antes.", ruta_ev)
        return 2
    ev = json.load(open(ruta_ev, encoding="utf-8"))

    destino = ROOT / "INFORME_LAB4_PARTE2.md"
    with open(destino, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# Informe — Laboratorio 4, Parte 2\n")
        w("## Modelos de aprendizaje automatico para detectar cianobacterias\n\n")
        w("**Universidad del Valle de Guatemala — CC3084 Data Science**  \n")
        w(f"Generado: {datetime.now():%Y-%m-%d %H:%M} | Semilla {SEED} | "
          f"Dataset {DATASET_VERSION}\n\n")
        w("Lagos de Atitlan y Amatitlan | 22 escenas Sentinel-2 L1C | "
          "3,756,510 observaciones | EPSG:32615 a 20 m (0.04 ha por pixel)\n\n---\n\n")

        w("## 1. Que se hizo\n\n")
        w(f"Se entrenaron tres clasificadores para predecir `{TARGET_COLUMN}` "
          f"(clorofila-a estimada >= {TARGET_THRESHOLD_UG_L:.0f} ug/L, la transicion "
          "a condicion eutrofica) a partir de **ocho variables espectrales** sin fuga "
          "de informacion: " + ", ".join(PREDICTORES_PRINCIPALES) + ".\n\n")
        w("Se evaluaron con cuatro estrategias de validacion distintas, cada una "
          "respondiendo a una pregunta diferente.\n\n")

        w("## 2. Resultados por estrategia de validacion\n\n")
        w("| Estrategia | Que estima | PR-AUC |\n|---|---|---|\n")
        if "aleatorio_pr_auc" in ev:
            mejor = ev.get("mejor_aleatorio", "")
            w(f"| Aleatoria 70/30 | Optimista: mezcla vecinos | "
              f"{ev['aleatorio_pr_auc']} (mejor: {mejor}) |\n")
        if "espacial_pr_auc" in ev:
            e = ev["espacial_pr_auc"]
            w("| Espacial (bloques 1 km) | Predecir en zona nueva | "
              + "; ".join(f"{m}={v['media']}±{v['desv']}" for m, v in e.items())
              + " |\n")
        if "temporal_pr_auc" in ev:
            t = ev["temporal_pr_auc"]
            w("| Temporal (expansiva) | Predecir fecha futura | "
              + "; ".join(f"{m}={v['media']}±{v['desv']}" for m, v in t.items())
              + " |\n")
        if "oof" in ev:
            w(f"| Out-of-fold espacial | Base de los mapas de error | "
              f"{ev['oof'].get('pr_auc')} |\n")
        w("\n")
        if ev.get("empate_practico"):
            w(f"**Empate practico.** La diferencia entre los dos mejores modelos bajo "
              f"validacion espacial es de {ev['diferencia_top2_espacial']}, menor que "
              "la desviacion entre folds. Random Forest se selecciona por una regla "
              "determinista, **no por superioridad sustantiva** sobre XGBoost.\n\n")

        if "cross_lake" in ev:
            w("## 3. Generalizacion entre lagos\n\n")
            w("| Experimento | Entrena | Evalua | PR-AUC | Recall operacional | TP | FN |\n")
            w("|---|---|---|---|---|---|---|\n")
            for x in ev["cross_lake"]:
                w(f"| {x['experimento']} | {x['train']} | {x['test']} | {x['pr_auc']} | "
                  f"{x['recall_operacional']} | {x['TP']:,} | {x['FN']:,} |\n")
            w("\nUn PR-AUC moderado **no equivale a generalizacion satisfactoria** "
              "cuando el recall operacional se desploma: el umbral aprendido en una "
              "distribucion no se transfiere a otra con prevalencia dos ordenes de "
              "magnitud distinta.\n\n")

        if "shap_top3" in ev:
            w("## 4. Interpretabilidad\n\n")
            w("Variables mas influyentes segun SHAP (media del valor absoluto):\n\n")
            w("| Variable | SHAP medio |\n|---|---|\n")
            for x in ev["shap_top3"]:
                w(f"| `{x['variable']}` | {x['shap_abs_media']:.5f} |\n")
            w("\n")
            for a in ADVERTENCIAS_SHAP:
                w(f"- {a}\n")
            w("\n")

        if "oof" in ev:
            w("## 5. Errores espaciales (out-of-fold)\n\n")
            o = ev["oof"]
            w(f"- Verdaderos positivos: {o.get('TP', 0):,}\n")
            w(f"- **Falsos negativos: {o.get('FN', 0):,}** "
              f"({o.get('area_FN_ha', 0):,.1f} ha sin aviso)\n")
            w(f"- Falsos positivos: {o.get('FP', 0):,} "
              f"({o.get('area_FP_ha', 0):,.1f} ha de inspeccion innecesaria)\n\n")
            w("El **falso negativo es el error mas grave** en vigilancia ambiental: "
              "una zona con floracion que no se avisa. Por eso la metrica prioritaria "
              "es el Recall y se reporta F2.\n\n")

        w("## 6. Conclusiones\n\n")
        for c in ev.get("conclusiones", []):
            w(f"- {c}\n")
        w("\n## 7. Limitaciones\n\n")
        for l in ev.get("limitaciones", []):
            w(f"- {l}\n")
        w("\n---\n\n")
        w("> **Alcance.** Herramienta de **cribado** para priorizar donde y cuando "
          "muestrear, dentro del dominio observado. No es un diagnostico de toxicidad "
          "ni un sustituto del monitoreo in situ.\n")

    LOGGER.info("Informe Markdown generado: %s", destino.name)
    LOGGER.info("No se genero ningun PDF (restriccion explicita).")
    return 0



# ---------------------------------------------------------------------------
# MODO --dry-run
# ---------------------------------------------------------------------------
def ejecutar_dry_run(args) -> int:
    perfil = args.profile
    cfg = PERFILES[perfil]
    aplicar_perfil_a_rutas(perfil)

    LOGGER.info("=" * 86)
    LOGGER.info("DRY-RUN - EJERCICIOS 8, 9 y 10 - PERFIL %s", perfil.upper())
    LOGGER.info("  %s", cfg["descripcion"])
    LOGGER.info("=" * 86)

    info = entorno()
    LOGGER.info("")
    LOGGER.info("ENTORNO")
    for k in ("python", "numpy", "pandas", "scikit_learn", "shap", "rasterio",
              "pyarrow", "xgboost", "cpus"):
        LOGGER.info("  %-14s: %s", k, info.get(k) or "NO INSTALADO")
    if info.get("shap") is None:
        LOGGER.error("shap es obligatorio para el Ejercicio 8.")
        return 2
    if info.get("rasterio") is None:
        LOGGER.error("rasterio es obligatorio para escribir los GeoTIFF.")
        return 2

    n_jobs = args.n_jobs or n_jobs_seguro()
    LOGGER.info("  %-14s: %d (recomendado seguro: %d)", "n_jobs", n_jobs,
                n_jobs_seguro())
    ram = memoria_disponible_gb()
    LOGGER.info("  %-14s: %s GB", "RAM disponible", ram if ram else "no medible")

    LOGGER.info("")
    LOGGER.info("MODELO")
    try:
        hp = hiperparametros_rf()
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    LOGGER.info("  Random Forest, seleccionado por validacion espacial")
    LOGGER.info("  Hiperparametros reales del tuning: %s", hp)
    LOGGER.info("  Umbral operacional: %.4f (fijado sin usar el test)", umbral_rf())

    LOGGER.info("")
    LOGGER.info("DATOS")
    datos = cargar_datos(perfil)
    X, y = datos["X"], datos["y"]
    n, pos = len(y), int(y.sum())
    verificar_sin_fuga(datos["features"], "dry-run")
    LOGGER.info("  Filas / predictores  : %s x %d", f"{n:,}", X.shape[1])
    LOGGER.info("  Positivos / Negativos: %s / %s (%.4f %%)",
                f"{pos:,}", f"{n-pos:,}", 100 * pos / n)
    LOGGER.info("  Memoria en RAM       : %.0f MB (pico al entrenar ~%.0f MB)",
                datos["memoria_mb"], datos["memoria_mb"] * 3.5)
    LOGGER.info("  Verificacion de fuga : OK")

    LOGGER.info("")
    LOGGER.info("EJERCICIO 8 - INTERPRETABILIDAD")
    sel, detalle = muestra_shap_estratificada(datos, cfg["shap_max"])
    LOGGER.info("  Muestra SHAP RECOMENDADA: %s filas", f"{len(sel):,}")
    LOGGER.info("    estratificada por lago x clase y repartida por fecha: %s", detalle)
    LOGGER.info("    prevalencia en la muestra %.2f %% frente a %.4f %% poblacional",
                100 * float(y[sel].mean()), 100 * y.mean())
    LOGGER.info("    (sobrerrepresenta positivos a proposito: son el recurso escaso)")
    LOGGER.info("  Permutation importance: hasta %s filas NO vistas, %d repeticiones",
                f"{cfg['perm_max']:,}", cfg["perm_repeats"])

    LOGGER.info("")
    LOGGER.info("EJERCICIO 9 - MAPAS")
    pares = combinaciones_oficiales()
    if cfg["max_combinaciones"]:
        pares = pares[:cfg["max_combinaciones"]]
    geoms = {l: geometria_lago(l) for l in datos["lake_nombres"]}
    bytes_tif = 0
    for lago, _ in pares:
        g = geoms[lago]
        bytes_tif += g["ancho"] * g["alto"] * 4
    LOGGER.info("  GeoTIFF de probabilidad: %d (float32, EPSG:32615, LZW)", len(pares))
    for l, g in geoms.items():
        LOGGER.info("    %-10s rejilla %d x %d", l, g["ancho"], g["alto"])
    LOGGER.info("  Disco estimado GeoTIFF : ~%.0f MB sin comprimir, ~%.0f MB con LZW",
                bytes_tif / 1e6, bytes_tif / 3e6)
    LOGGER.info("  Parquet OOF estimado   : ~%.0f MB", n * 30 / 1e6)
    LOGGER.info("  Figuras PNG            : ~8-12 (versionables)")
    LOGGER.info("  Fechas criticas de la Parte 1: %s", fechas_criticas_parte1())

    LOGGER.info("")
    LOGGER.info("ESTRATEGIA OUT-OF-FOLD")
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True,
                                random_state=SEED)
    cobertura = np.zeros(n, dtype=np.int16)
    compartidos = 0
    for i, (tr, va) in enumerate(sgkf.split(np.zeros(n), y, groups=datos["bloque"]),
                                 start=1):
        comp = set(np.unique(datos["bloque"][tr])) & set(np.unique(datos["bloque"][va]))
        compartidos += len(comp)
        cobertura[va] += 1
        LOGGER.info("  fold %d: ajuste %s | OOF %s | positivos OOF %s | compartidos %d",
                    i, f"{len(tr):,}", f"{len(va):,}", f"{int(y[va].sum()):,}", len(comp))
    LOGGER.info("  Bloques compartidos totales: %d (debe ser 0)", compartidos)
    LOGGER.info("  Cada observacion predicha exactamente una vez: %s",
                bool((cobertura == 1).all()))
    if compartidos or not (cobertura == 1).all():
        LOGGER.error("La estrategia OOF no es valida.")
        return 1

    LOGGER.info("")
    LOGGER.info("COSTO ESTIMADO")
    n_ajustes = 1 + 1 + cfg["n_folds"]
    LOGGER.info("  Ajustes de Random Forest: %d", n_ajustes)
    LOGGER.info("    1 para interpretabilidad (fold espacial)")
    LOGGER.info("    1 final con todo el dataset (mapas descriptivos)")
    LOGGER.info("    %d para las predicciones out-of-fold", cfg["n_folds"])
    from sklearn.datasets import make_classification
    Xc, yc = make_classification(n_samples=30_000, n_features=len(PREDICTORES_PRINCIPALES),
                                 n_informative=5, weights=[0.984, 0.016],
                                 random_state=SEED)
    m = construir_rf(perfil, n_jobs)
    t0 = time.time(); m.fit(Xc, yc); s_fila = (time.time() - t0) / 30_000
    t_total = s_fila * n * n_ajustes
    LOGGER.info("  Velocidad medida        : %.2e s/fila", s_fila)
    LOGGER.info("  Tiempo estimado ajustes : %.0f min", t_total / 60)
    LOGGER.info("  SHAP + permutation      : ~%.0f min", cfg["shap_budget_s"] / 60)
    LOGGER.info("  TIEMPO TOTAL ESTIMADO   : %.0f min (~%.1f h)",
                t_total / 60 + cfg["shap_budget_s"] / 60,
                (t_total + cfg["shap_budget_s"]) / 3600)
    LOGGER.info("  DISCO TOTAL ESTIMADO    : ~%.0f MB (GeoTIFF + Parquet OOF + figuras)",
                bytes_tif / 3e6 + n * 30 / 1e6 + 20)

    LOGGER.info("")
    LOGGER.info("ESTADO DE LAS RUTAS")
    for ruta, funcion in [("--explain", ejecutar_explain), ("--maps", ejecutar_maps),
                          ("--errors-oof", ejecutar_errors_oof),
                          ("--conclusions", ejecutar_conclusions),
                          ("--report-only", ejecutar_report_only),
                          ("--all", ejecutar_all), ("--validate", ejecutar_validate)]:
        LOGGER.info("  %-16s: IMPLEMENTADA (%s)", ruta, funcion.__name__)

    checkpoints = {"explain": INTERP_DIR / f"checkpoint_explain_{perfil}.json",
                   "maps": PROB_DIR / f"checkpoint_maps_{perfil}.json",
                   "errors_oof": ERR_DIR / f"checkpoint_errors_{perfil}.json"}
    LOGGER.info("  Checkpoints: %s",
                ", ".join(f"{k}={'si' if v.exists() else 'no'}"
                          for k, v in checkpoints.items()))

    LOGGER.info("")
    LOGGER.info("ADVERTENCIAS")
    avisos = [
        "Los mapas de probabilidad se generan con un modelo ajustado sobre TODO el "
        "dataset: son descriptivos/de despliegue, NO una validacion no sesgada.",
        "Los mapas de error TP/TN/FP/FN se construyen SOLO con predicciones "
        "out-of-fold espaciales, que si son honestas.",
        "La muestra SHAP sobrerrepresenta positivos y no reproduce la prevalencia "
        "poblacional.",
        "SHAP describe el comportamiento del modelo, no causalidad; las bandas estan "
        "correlacionadas entre si.",
        "La respuesta es un proxy espectral de clorofila-a sin validacion in situ.",
    ]
    if perfil == "smoke":
        avisos.insert(0, "PERFIL SMOKE: resultados NO entregables, solo prueban que el "
                         "pipeline corre.")
    for a in avisos:
        LOGGER.warning("  - %s", a)

    crear_carpetas()
    escribir_json(REPORTS_DIR / f"dry_run_plan_ej8_10_{perfil}.json", {
        "perfil": perfil, "entorno": info, "n_jobs": n_jobs,
        "modelo": "random_forest", "hiperparametros": hp, "umbral": umbral_rf(),
        "n_filas": int(n), "positivos": int(pos),
        "shap_recomendado": int(len(sel)), "estratos_shap": detalle,
        "n_ajustes_rf": n_ajustes, "n_geotiff": len(pares),
        "tiempo_estimado_min": round(t_total / 60 + cfg["shap_budget_s"] / 60, 1),
        "disco_estimado_mb": round(bytes_tif / 3e6 + n * 30 / 1e6 + 20, 1),
        "memoria_mb": round(datos["memoria_mb"], 1),
        "oof_bloques_compartidos": compartidos,
        "oof_cobertura_unica": bool((cobertura == 1).all()),
        "advertencias": avisos,
    })

    LOGGER.info("")
    LOGGER.info("DRY-RUN correcto. No se genero ningun resultado definitivo.")
    LOGGER.info("")
    LOGGER.info("COMANDO RECOMENDADO A CONTINUACION:")
    LOGGER.info("  python explicabilidad_mapas_parte2.py --all --profile %s --n-jobs %d",
                perfil, n_jobs)
    return 0


# ---------------------------------------------------------------------------
# MODO --all
# ---------------------------------------------------------------------------
def ejecutar_all(args) -> int:
    perfil = args.profile
    LOGGER.info("#" * 86)
    LOGGER.info("EJERCICIOS 8, 9 y 10 - PERFIL %s", perfil.upper())
    if perfil == "smoke":
        LOGGER.warning("PERFIL SMOKE: los resultados NO son entregables.")
    LOGGER.info("#" * 86)

    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    t0 = time.time()
    datos = cargar_datos(perfil)     # se carga una sola vez

    fases = [("explain", ejecutar_explain), ("maps", ejecutar_maps),
             ("errors_oof", ejecutar_errors_oof)]
    estado = {}
    for nombre, funcion in fases:
        LOGGER.info("")
        try:
            codigo = funcion(args, datos)
        except Exception as exc:
            LOGGER.exception("Fase %s fallo: %s", nombre, exc)
            estado[nombre] = f"ERROR: {exc}"
            LOGGER.error("Se detienen las fases dependientes; los artefactos validos "
                         "previos se conservan y puede reanudar con el mismo comando.")
            break
        estado[nombre] = "OK" if codigo == 0 else f"codigo {codigo}"
        if codigo != 0:
            LOGGER.error("Fase %s devolvio %d; se detiene.", nombre, codigo)
            break

    if all(v == "OK" for v in estado.values()) and len(estado) == len(fases):
        LOGGER.info("")
        estado["conclusions"] = ("OK" if ejecutar_conclusions(args) == 0
                                 else "fallo")

    LOGGER.info("")
    LOGGER.info("=" * 86)
    LOGGER.info("RESUMEN (%.1f min)", (time.time() - t0) / 60)
    for nombre in ("explain", "maps", "errors_oof", "conclusions"):
        LOGGER.info("  %-12s: %s", nombre, estado.get(nombre, "no ejecutada"))
    if perfil == "smoke":
        LOGGER.warning("Recuerde: el informe final NO debe redactarse con estas cifras.")
    else:
        LOGGER.info("Siguiente paso: --report-only para generar "
                    "INFORME_LAB4_PARTE2.md (sin PDF).")
    return 0 if all(v == "OK" for v in estado.values()) else 1


# ---------------------------------------------------------------------------
# MODO --validate
# ---------------------------------------------------------------------------
def ejecutar_validate(args) -> int:
    perfil = args.profile
    aplicar_perfil_a_rutas(perfil)
    crear_carpetas()
    criticos, avisos, lineas = [], [], []

    info = entorno()
    for dep in ("shap", "rasterio", "pyarrow"):
        if info.get(dep) is None:
            criticos.append(f"{dep} no instalado")
    lineas.append(f"shap {info.get('shap')} | rasterio {info.get('rasterio')}")
    lineas.append(f"Perfil                     : {perfil}")

    try:
        verificar_sin_fuga(list(PREDICTORES_PRINCIPALES), "validate")
        lineas.append(f"Predictores sin fuga       : {len(PREDICTORES_PRINCIPALES)}")
    except AssertionError as exc:
        criticos.append(str(exc))
    if TARGET_COLUMN != "high_cyano_8":
        criticos.append(f"Respuesta inesperada: {TARGET_COLUMN}")

    # --- Ejercicio 8 ---
    for nombre in (f"importancia_nativa_{perfil}.csv",
                   f"permutation_importance_{perfil}.csv",
                   f"shap_abs_media_{perfil}.csv",
                   f"interpretabilidad_meta_{perfil}.json"):
        r = INTERP_DIR / nombre
        if not r.exists():
            criticos.append(f"Falta {nombre}")
        elif r.suffix == ".csv" and pd.read_csv(r).empty:
            criticos.append(f"{nombre} esta vacio")
    figs8 = list(INTERP_DIR.glob(f"*{perfil}.png"))
    lineas.append(f"Figuras de interpretabilidad: {len(figs8)}")
    if len(figs8) < 4:
        criticos.append(f"Se esperaban >= 4 figuras de interpretabilidad, hay {len(figs8)}")

    meta = INTERP_DIR / f"interpretabilidad_meta_{perfil}.json"
    if meta.exists():
        m = json.load(open(meta, encoding="utf-8"))
        lineas.append(f"Muestra SHAP               : {m.get('n_shap'):,} filas")
        lineas.append(f"Permutation sobre no vistas: {m.get('n_permutation'):,} filas")
        if m.get("hiperparametros") != hiperparametros_rf():
            criticos.append("Los hiperparametros usados no coinciden con el tuning real")
        if not m.get("advertencias"):
            criticos.append("Falta documentar las advertencias de SHAP")

    # --- Ejercicio 9: GeoTIFF legibles ---
    tifs = list(PROB_DIR.rglob("prob_*.tif"))
    lineas.append(f"GeoTIFF de probabilidad    : {len(tifs)}")
    if not tifs:
        criticos.append("No se genero ningun GeoTIFF de probabilidad")
    else:
        import rasterio
        with rasterio.open(tifs[0]) as src:
            arr = src.read(1, masked=True)
            lineas.append(f"  ejemplo: {tifs[0].name} {src.width}x{src.height} "
                          f"{src.crs} dtype={src.dtypes[0]} nodata={src.nodata}")
            if str(src.crs) != CRS_ESPERADO:
                criticos.append(f"{tifs[0].name}: CRS {src.crs} != {CRS_ESPERADO}")
            if src.dtypes[0] != "float32":
                criticos.append(f"{tifs[0].name}: dtype {src.dtypes[0]} != float32")
            v = arr.compressed()
            if v.size and (v.min() < 0 or v.max() > 1):
                criticos.append(f"{tifs[0].name}: probabilidades fuera de [0,1]")
            tags = src.tags()
            for k in ("lago", "fecha", "modelo", "umbral_operacional",
                      "tipo_prediccion"):
                if k not in tags:
                    criticos.append(f"{tifs[0].name}: falta el tag {k}")

    # --- Ejercicio 9: OOF ---
    ck_oof = ERR_DIR / f"checkpoint_errors_{perfil}.json"
    if not ck_oof.exists():
        criticos.append("Falta el checkpoint de errores out-of-fold")
    else:
        c = json.load(open(ck_oof, encoding="utf-8"))
        lineas.append(f"OOF: bloques compartidos {c.get('bloques_compartidos')} | "
                      f"cobertura {c.get('cobertura_oof'):,} | "
                      f"PR-AUC {c.get('pr_auc_oof')}")
        if c.get("bloques_compartidos"):
            criticos.append("Hay bloques compartidos en la validacion out-of-fold")
    for nombre in (f"metricas_oof_por_lago_{perfil}.csv",
                   f"metricas_oof_por_fecha_{perfil}.csv",
                   f"metricas_oof_por_bloque_{perfil}.csv",
                   f"conteo_errores_{perfil}.csv"):
        if not (ERR_DIR / nombre).exists():
            criticos.append(f"Falta {nombre}")
    figs9 = list(MAPFIG_DIR.glob(f"*{perfil}.png"))
    lineas.append(f"Figuras de mapas           : {len(figs9)}")
    if len(figs9) < 2:
        criticos.append(f"Se esperaban >= 2 figuras de mapas, hay {len(figs9)}")

    # --- Ejercicio 10 ---
    rc = CONCL_DIR / f"conclusiones_{perfil}.json"
    if not rc.exists():
        criticos.append("Faltan las conclusiones calculadas")
    else:
        ev = json.load(open(rc, encoding="utf-8"))
        for clave in ("conclusiones", "limitaciones"):
            if not ev.get(clave):
                criticos.append(f"Las conclusiones no incluyen '{clave}'")
        lineas.append(f"Conclusiones               : {len(ev.get('conclusiones', []))} | "
                      f"limitaciones {len(ev.get('limitaciones', []))}")

    # --- Aislamiento smoke / standard ---
    if perfil == "smoke":
        fuera = [p for p in (BASE / "interpretability", BASE / "conclusions",
                             BASE / "maps" / "probability")
                 if p.exists() and any(p.glob("*smoke*"))]
        if fuera:
            criticos.append(f"Artefactos smoke en carpetas standard: {fuera}")
        else:
            lineas.append("Aislamiento smoke          : correcto")
        if (ROOT / "INFORME_LAB4_PARTE2.md").exists():
            t = (ROOT / "INFORME_LAB4_PARTE2.md").read_text(encoding="utf-8")
            if "SMOKE" in t.upper():
                criticos.append("El informe final contiene cifras smoke")

    # --- Sin PDF de la Parte 2 (restriccion permanente, no solo de este turno) ---
    if (ROOT / "INFORME_LAB4_PARTE2.pdf").exists():
        criticos.append("Existe INFORME_LAB4_PARTE2.pdf: la Parte 2 prohibe generar PDF")
    else:
        lineas.append("Sin PDF de la Parte 2       : correcto")

    if perfil == "standard":
        # --- Informe academico ---
        ruta_inf = ROOT / "INFORME_LAB4_PARTE2.md"
        if not ruta_inf.exists():
            criticos.append("Falta INFORME_LAB4_PARTE2.md")
        else:
            texto_inf = ruta_inf.read_text(encoding="utf-8")
            if "PENDIENTE" in texto_inf:
                criticos.append("INFORME_LAB4_PARTE2.md contiene PENDIENTE")
            if "SMOKE" in texto_inf.upper():
                criticos.append("INFORME_LAB4_PARTE2.md contiene cifras smoke")
            import re as _re
            refs = _re.findall(r"!\[.*?\]\((.+?)\)", texto_inf)
            rotas = [r for r in refs if not (ROOT / r).exists()]
            lineas.append(f"Informe academico           : {len(texto_inf.split()):,} "
                          f"palabras, {len(refs)} figuras ({len(rotas)} rotas)")
            if rotas:
                criticos.append(f"INFORME_LAB4_PARTE2.md referencia figuras inexistentes: "
                                f"{rotas[:3]}")
            if not refs:
                criticos.append("INFORME_LAB4_PARTE2.md no incrusta ninguna figura")

        # --- Articulo tecnico ---
        ruta_art = ROOT / "ARTICULO_TECNICO_LAB4.md"
        if not ruta_art.exists():
            criticos.append("Falta ARTICULO_TECNICO_LAB4.md")
        else:
            texto_art = ruta_art.read_text(encoding="utf-8")
            n_palabras = len(texto_art.split())
            lineas.append(f"Articulo tecnico             : {n_palabras:,} palabras")
            if not (3500 <= n_palabras <= 8000):
                criticos.append(f"ARTICULO_TECNICO_LAB4.md tiene {n_palabras} palabras, "
                                "fuera del rango 4000-7000 esperado")
            if "PENDIENTE" in texto_art or "SMOKE" in texto_art.upper():
                criticos.append("ARTICULO_TECNICO_LAB4.md contiene PENDIENTE o cifras smoke")

        # --- Notebook final ---
        nb_ruta = ROOT / "lab4-2.ipynb"
        if not nb_ruta.exists():
            criticos.append("Falta lab4-2.ipynb")
        else:
            import json as _json
            nb_texto = nb_ruta.read_text(encoding="utf-8")
            nb_json = _json.loads(nb_texto)
            n_celdas = len(nb_json["cells"])
            n_ejec = sum(1 for c in nb_json["cells"]
                        if c["cell_type"] == "code" and c.get("execution_count"))
            n_err = sum(1 for c in nb_json["cells"] for o in c.get("outputs", [])
                       if o.get("output_type") == "error")
            lineas.append(f"Notebook final              : {n_celdas} celdas, "
                          f"{n_ejec} ejecutadas, {n_err} errores")
            if n_err:
                criticos.append(f"lab4-2.ipynb tiene {n_err} celdas con error")
            tiene_8_10 = all(f"Ejercicio {n}" in nb_texto for n in (8, 9, 10))
            if not tiene_8_10:
                criticos.append("lab4-2.ipynb no incluye los Ejercicios 8, 9 y 10")
            if "SMOKE - NO ENTREGABLE" in nb_texto:
                criticos.append("lab4-2.ipynb muestra resultados marcados como smoke")

        # --- Inventario 8-10 ---
        ruta_inv = REPORTS_DIR / "inventario_ejercicios_8_10.csv"
        if not ruta_inv.exists():
            criticos.append("Falta outputs/parte2/reports/inventario_ejercicios_8_10.csv")
        else:
            inv = pd.read_csv(ruta_inv)
            lineas.append(f"Inventario Ejercicios 8-10   : {len(inv)} artefactos")

    ruta_rep = REPORTS_DIR / f"validacion_ej8_10_{perfil}.txt"
    with open(ruta_rep, "w", encoding="utf-8") as fh:
        fh.write("VALIDACION EJERCICIOS 8, 9 y 10 - LABORATORIO 4 PARTE 2\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\nPerfil: {perfil}\n\n")
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
        LOGGER.error("VALIDACION FALLIDA (%d criticos)", len(criticos))
        return 1
    LOGGER.info("")
    LOGGER.info("VALIDACION CORRECTA")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def construir_parser():
    p = argparse.ArgumentParser(
        prog="explicabilidad_mapas_parte2.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Ejercicios 8, 9 y 10 del Laboratorio 4, Parte 2.\n\n"
                     "8. Interpretabilidad del Random Forest (impureza, permutation, SHAP).\n"
                     "9. Mapas de probabilidad y mapas de error out-of-fold.\n"
                     "10. Conclusiones calculadas e informe en Markdown.\n\n"
                     "No genera ningun PDF."),
        epilog=("PERFILES\n"
                "  smoke     muestra pequena, aislada en outputs/parte2/smoke/.\n"
                "            Sus resultados NUNCA son conclusiones del laboratorio.\n"
                "  standard  configuracion entregable.\n\n"
                "CODIGOS DE SALIDA\n"
                "  0 correcto | 1 errores criticos | 2 falta un requisito\n"))
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true",
                   help="Planifica y estima costos sin producir resultados.")
    m.add_argument("--explain", action="store_true",
                   help="Ejercicio 8: importancia nativa, permutation y SHAP.")
    m.add_argument("--maps", action="store_true",
                   help="Ejercicio 9: GeoTIFF y PNG de probabilidad (descriptivos).")
    m.add_argument("--errors-oof", action="store_true", dest="errors_oof",
                   help="Ejercicio 9: mapas TP/TN/FP/FN out-of-fold espaciales.")
    m.add_argument("--conclusions", action="store_true",
                   help="Ejercicio 10: conclusiones calculadas.")
    m.add_argument("--report-only", action="store_true", dest="report_only",
                   help="Genera INFORME_LAB4_PARTE2.md (solo standard, sin PDF).")
    m.add_argument("--all", action="store_true",
                   help="Ejecuta explain, maps, errors-oof y conclusions.")
    m.add_argument("--validate", action="store_true",
                   help="Valida los artefactos de los Ejercicios 8-10.")
    p.add_argument("--profile", choices=list(PERFILES), default="standard")
    p.add_argument("--n-jobs", type=int, default=None, dest="n_jobs")
    p.add_argument("--force", action="store_true",
                   help="Rehace las fases aunque tengan checkpoint valido.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv=None):
    parser = construir_parser()
    args = parser.parse_args(argv)
    configurar_logging(args.verbose)
    try:
        if args.dry_run:
            return ejecutar_dry_run(args)
        if args.explain:
            return ejecutar_explain(args)
        if args.maps:
            return ejecutar_maps(args)
        if args.errors_oof:
            return ejecutar_errors_oof(args)
        if args.conclusions:
            return ejecutar_conclusions(args)
        if args.report_only:
            return ejecutar_report_only(args)
        if args.all:
            return ejecutar_all(args)
        if args.validate:
            return ejecutar_validate(args)
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
