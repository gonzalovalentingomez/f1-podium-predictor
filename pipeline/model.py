"""Entrenamiento del modelo de podio (clasificación binaria: top 3 sí/no).

Arranca simple, como pide el brief: Random Forest Classifier o regresión
logística (scikit-learn) sobre las 8 features ya construidas en
`features.py` (todas menos el proxy de rookies vía Fórmula 2, que queda
pendiente). Comparar algoritmos y calibrar probabilidades es trabajo de
iteración futura, no de esta primera versión.

Pondera las filas por recencia: el reglamento 2026 (motores 50/50
batería-combustión) invalida buena parte del rendimiento histórico de
auto/equipo, así que las features de equipo/piloto (forma reciente,
ritmo, confiabilidad) deben pesar cada vez menos cuanto más vieja es la
temporada. Las features de circuito (dificultad de adelantamiento,
historial) sí se benefician de histórico multi-temporada, y esa
información ya entra completa porque se calcula sobre todo el dataset
disponible en `features.py`, antes de aplicar el peso por fila acá.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import features

COLUMNAS_FEATURES = [
    "grid",
    "gap_pole_seg",
    "forma_reciente",
    "historial_piloto_circuito",
    "historial_equipo_circuito",
    "dificultad_adelantamiento",
    "delta_clasificacion_ritmo",
    "curva_desarrollo",
    "tasa_dnf_equipo",
    "lluvia",
    "temperatura_c",
]
COLUMNA_OBJETIVO = "podio"

TEMPORADA_REGLAMENTO_NUEVO = 2026
FACTOR_DECAIMIENTO_POR_TEMPORADA = 0.6  # peso *= esto por cada año de antigüedad


def calcular_peso_por_recencia(
    tabla: pd.DataFrame,
    temporada_actual: int = TEMPORADA_REGLAMENTO_NUEVO,
    factor_decaimiento: float = FACTOR_DECAIMIENTO_POR_TEMPORADA,
) -> pd.Series:
    """Calcula el peso de entrenamiento de cada fila según su antigüedad.

    peso = factor_decaimiento ** (temporada_actual - temporada). La
    temporada más reciente pesa 1.0; cada temporada hacia atrás pesa
    menos, exponencialmente (con el default 0.6: la temporada anterior
    pesa 0.6, la de hace dos años 0.36, etc.). No se descartan del todo
    las temporadas viejas porque las features de circuito ya incorporan
    su historia de forma agregada (ver docstring del módulo).

    Args:
        tabla: Debe tener una columna `temporada`.
        temporada_actual: Temporada de referencia (peso 1.0).
        factor_decaimiento: Cuánto pesa cada año de antigüedad (entre 0 y 1).

    Returns:
        Serie de pesos, alineada al índice de `tabla`.
    """
    antiguedad = (temporada_actual - tabla["temporada"]).clip(lower=0)
    return factor_decaimiento**antiguedad


def preparar_datos(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Selecciona features y objetivo, y arma los pesos por recencia.

    Convierte `lluvia` (bool) a 0/1 para que el imputer numérico pueda
    operar sobre ella junto con el resto de las features.

    Returns:
        Tupla (X, y, pesos).
    """
    dataset = dataset.copy()
    dataset["lluvia"] = dataset["lluvia"].astype("float")

    X = dataset[COLUMNAS_FEATURES]
    y = dataset[COLUMNA_OBJETIVO].astype(int)
    pesos = calcular_peso_por_recencia(dataset)
    return X, y, pesos


def dividir_entrenamiento_evaluacion(
    dataset: pd.DataFrame, fraccion_evaluacion: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento y evaluación de forma CRONOLÓGICA, no al azar.

    El modelo se usa para predecir carreras futuras, así que evaluarlo
    con una división al azar sería optimista (mezclaría carreras
    "futuras" en el set de entrenamiento). Se ordena por temporada/ronda
    y se separa el último `fraccion_evaluacion` de las carreras como
    evaluación.

    Args:
        dataset: Tabla con columnas temporada y ronda.
        fraccion_evaluacion: Proporción de carreras (no de filas) a
            reservar para evaluación, tomando las más recientes.

    Returns:
        Tupla (entrenamiento, evaluacion), ambos subconjuntos de `dataset`.
    """
    carreras = (
        dataset[["temporada", "ronda"]]
        .drop_duplicates()
        .sort_values(["temporada", "ronda"])
        .reset_index(drop=True)
    )
    corte = int(len(carreras) * (1 - fraccion_evaluacion))

    entrenamiento = dataset.merge(carreras.iloc[:corte], on=["temporada", "ronda"], how="inner")
    evaluacion = dataset.merge(carreras.iloc[corte:], on=["temporada", "ronda"], how="inner")
    return entrenamiento, evaluacion


def construir_pipeline(algoritmo: str = "random_forest") -> Pipeline:
    """Arma el pipeline de imputación + (escalado) + clasificador.

    Args:
        algoritmo: "random_forest" (default) o "logistica". El brief
            propone ambos como punto de partida; comparar resultados
            entre los dos queda para una iteración futura.
    """
    if algoritmo == "logistica":
        pasos = [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
            ("clasificar", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    elif algoritmo == "random_forest":
        pasos = [
            ("imputar", SimpleImputer(strategy="median")),
            ("clasificar", RandomForestClassifier(
                n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
            )),
        ]
    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo!r}")

    return Pipeline(pasos)


def imprimir_importancia_features(pipeline: Pipeline, algoritmo: str) -> None:
    """Imprime qué tan importante fue cada feature para el modelo final.

    Sirve para contrastar el resultado empírico contra el criterio propio
    mencionado en el brief (ej.: ¿el modelo realmente aprendió que el
    grid importa mucho en circuitos de alta dificultad de adelantamiento?).
    """
    clasificador = pipeline.named_steps["clasificar"]
    if algoritmo == "random_forest":
        valores = clasificador.feature_importances_
    else:
        valores = clasificador.coef_[0]

    importancias = sorted(zip(COLUMNAS_FEATURES, valores), key=lambda par: abs(par[1]), reverse=True)
    print("Importancia de features:")
    for nombre, valor in importancias:
        print(f"  {nombre}: {valor:.4f}")


def entrenar_y_evaluar(
    temporadas: list[int], algoritmo: str = "random_forest", usar_cache: bool = True
) -> Pipeline:
    """Arma el dataset, entrena el modelo y evalúa contra las carreras más recientes.

    Args:
        temporadas: Temporadas a incluir (ver `features.construir_dataset_base`).
        algoritmo: Ver `construir_pipeline`.
        usar_cache: Si es False, fuerza a pedir todos los datos a las APIs.

    Returns:
        El pipeline final, reentrenado sobre TODO el dataset (entrenamiento
        + evaluación) una vez medida la performance: para el modelo que
        se va a usar para predecir no tiene sentido dejar afuera las
        carreras más recientes, que son las más informativas.
    """
    dataset = features.construir_dataset_base(temporadas, usar_cache=usar_cache)
    entrenamiento, evaluacion = dividir_entrenamiento_evaluacion(dataset)

    X_entrenamiento, y_entrenamiento, pesos_entrenamiento = preparar_datos(entrenamiento)
    X_evaluacion, y_evaluacion, _ = preparar_datos(evaluacion)

    pipeline_modelo = construir_pipeline(algoritmo)
    pipeline_modelo.fit(X_entrenamiento, y_entrenamiento, clasificar__sample_weight=pesos_entrenamiento)

    predicciones = pipeline_modelo.predict(X_evaluacion)
    probabilidades = pipeline_modelo.predict_proba(X_evaluacion)[:, 1]

    print(
        f"Evaluación sobre {len(X_evaluacion)} filas "
        f"({evaluacion[['temporada', 'ronda']].drop_duplicates().shape[0]} carreras más "
        "recientes, fuera del entrenamiento):"
    )
    print(classification_report(y_evaluacion, predicciones, target_names=["no_podio", "podio"]))
    print(f"ROC-AUC: {roc_auc_score(y_evaluacion, probabilidades):.3f}")
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print(confusion_matrix(y_evaluacion, predicciones))

    X_todo, y_todo, pesos_todo = preparar_datos(dataset)
    pipeline_final = construir_pipeline(algoritmo)
    pipeline_final.fit(X_todo, y_todo, clasificar__sample_weight=pesos_todo)
    imprimir_importancia_features(pipeline_final, algoritmo)

    return pipeline_final


if __name__ == "__main__":
    import sys

    temporadas_pedidas = [int(arg) for arg in sys.argv[1:]] or list(range(2019, 2027))
    modelo = entrenar_y_evaluar(temporadas_pedidas)

    salida = Path(__file__).parent / "data" / "modelo_podio.joblib"
    salida.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, salida)
    print(f"Modelo guardado en {salida}")
