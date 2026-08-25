"""Pruebas simples para el módulo model.py.

Mismo estilo que `tests/test_analisis.py` del F1 Stats Explorer: sin
framework de testing, funciones de verificación manual con asserts,
pensadas para correrse directamente con: python pipeline/tests/test_model.py

No pegan a ninguna API: usan datos armados a mano para verificar la
lógica de pesos, el split cronológico y que el pipeline de scikit-learn
entrena sin romperse.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

import model  # noqa: E402


def test_calcular_peso_por_recencia():
    tabla = pd.DataFrame({"temporada": [2026, 2025, 2024]})
    pesos = model.calcular_peso_por_recencia(tabla, temporada_actual=2026, factor_decaimiento=0.5)

    assert pesos.iloc[0] == 1.0
    assert pesos.iloc[1] == 0.5
    assert pesos.iloc[2] == 0.25


def test_dividir_entrenamiento_evaluacion():
    tabla = pd.DataFrame({
        "temporada": [2025, 2025, 2025, 2025, 2026, 2026, 2026, 2026, 2026, 2026],
        "ronda": [1, 1, 2, 2, 1, 1, 2, 2, 3, 3],
    })
    entrenamiento, evaluacion = model.dividir_entrenamiento_evaluacion(tabla, fraccion_evaluacion=0.2)

    # 5 carreras en total (2025-r1, 2025-r2, 2026-r1, 2026-r2, 2026-r3); 20% = la última.
    carreras_evaluacion = set(map(tuple, evaluacion[["temporada", "ronda"]].drop_duplicates().values))
    assert carreras_evaluacion == {(2026, 3)}

    carreras_entrenamiento = set(map(tuple, entrenamiento[["temporada", "ronda"]].drop_duplicates().values))
    assert (2026, 3) not in carreras_entrenamiento
    assert len(entrenamiento) + len(evaluacion) == len(tabla)


def test_preparar_datos_convierte_lluvia_a_numerico():
    dataset = pd.DataFrame({
        "temporada": [2026], "grid": [1], "gap_pole_seg": [0.0], "forma_reciente": [1.0],
        "historial_piloto_circuito": [1.0], "historial_equipo_circuito": [1.0],
        "dificultad_adelantamiento": [0.5], "delta_clasificacion_ritmo": [0.0],
        "curva_desarrollo": [0.0], "tasa_dnf_equipo": [0.0], "lluvia": [True],
        "temperatura_c": [20.0], "podio": [True],
    })
    X, y, pesos = model.preparar_datos(dataset)

    assert X["lluvia"].iloc[0] == 1.0
    assert y.iloc[0] == 1
    assert pesos.iloc[0] == 1.0


def test_entrenar_pipeline_de_punta_a_punta_con_datos_sinteticos():
    """No pega a ninguna API: entrena con un dataset armado a mano, solo
    para verificar que imputación + clasificador no rompen y que el
    modelo aprende a distinguir el caso obvio (grid 1 vs. grid 10)."""
    filas = []
    for i in range(40):
        es_podio = i % 5 == 0
        filas.append({
            "temporada": 2026,
            "grid": 1 if es_podio else 10,
            "gap_pole_seg": 0.0 if es_podio else 1.5,
            "forma_reciente": 2.0 if es_podio else 12.0,
            "historial_piloto_circuito": None,
            "historial_equipo_circuito": 3.0 if es_podio else 14.0,
            "dificultad_adelantamiento": 0.5,
            "delta_clasificacion_ritmo": 0.0,
            "curva_desarrollo": None,
            "tasa_dnf_equipo": 0.0,
            "lluvia": False,
            "temperatura_c": 22.0,
            "podio": es_podio,
        })
    dataset = pd.DataFrame(filas)

    X, y, pesos = model.preparar_datos(dataset)
    pipeline_modelo = model.construir_pipeline("random_forest")
    pipeline_modelo.fit(X, y, clasificar__sample_weight=pesos)

    predicciones = pipeline_modelo.predict(X)
    assert len(predicciones) == len(X)
    assert set(predicciones).issubset({0, 1})
    assert (predicciones == y.values).mean() > 0.9  # caso obvio, debería acertar casi siempre


if __name__ == "__main__":
    test_calcular_peso_por_recencia()
    test_dividir_entrenamiento_evaluacion()
    test_preparar_datos_convierte_lluvia_a_numerico()
    test_entrenar_pipeline_de_punta_a_punta_con_datos_sinteticos()
    print("Todas las pruebas de model.py pasaron correctamente.")
