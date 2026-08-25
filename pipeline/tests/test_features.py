"""Pruebas simples para el módulo features.py.

Mismo estilo que `tests/test_analisis.py` del F1 Stats Explorer: sin
framework de testing, funciones de verificación manual con asserts,
pensadas para correrse directamente con: python pipeline/tests/test_features.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

import features  # noqa: E402

CARRERAS_DE_PRUEBA = [
    {
        "season": "2026", "round": "14", "raceName": "Italian Grand Prix",
        "date": "2026-09-06",
        "Circuit": {"circuitId": "monza", "circuitName": "Autodromo Nazionale di Monza"},
        "Results": [
            {"position": "1", "grid": "1", "points": "25", "status": "Finished",
             "Driver": {"driverId": "norris", "givenName": "Lando", "familyName": "Norris"},
             "Constructor": {"constructorId": "mclaren", "name": "McLaren"}},
            {"position": "4", "grid": "3", "points": "12", "status": "Finished",
             "Driver": {"driverId": "verstappen", "givenName": "Max", "familyName": "Verstappen"},
             "Constructor": {"constructorId": "red_bull", "name": "Red Bull"}},
            {"position": "R", "grid": "5", "points": "0", "status": "Accident",
             "Driver": {"driverId": "leclerc", "givenName": "Charles", "familyName": "Leclerc"},
             "Constructor": {"constructorId": "ferrari", "name": "Ferrari"}},
        ],
    },
]

CARRERAS_QUALI_DE_PRUEBA = [
    {
        "season": "2026", "round": "14",
        "QualifyingResults": [
            {"Driver": {"driverId": "norris"}, "Q1": "1:20.500", "Q2": "1:19.800", "Q3": "1:19.500"},
            {"Driver": {"driverId": "verstappen"}, "Q1": "1:20.700", "Q2": "1:20.100", "Q3": "1:19.900"},
            {"Driver": {"driverId": "leclerc"}, "Q1": "1:21.000", "Q2": "1:20.400", "Q3": None},
        ],
    },
]


def test_tiempo_a_segundos():
    assert features._tiempo_a_segundos("1:19.500") == 79.5
    assert features._tiempo_a_segundos("19.500") == 19.5
    assert features._tiempo_a_segundos(None) is None
    assert features._tiempo_a_segundos("") is None


def test_construir_tabla_resultados():
    tabla = features.construir_tabla_resultados(CARRERAS_DE_PRUEBA)

    assert len(tabla) == 3
    assert tabla.loc[tabla["piloto_id"] == "norris", "podio"].iloc[0] == True  # noqa: E712
    assert tabla.loc[tabla["piloto_id"] == "verstappen", "podio"].iloc[0] == False  # noqa: E712
    fila_leclerc = tabla.loc[tabla["piloto_id"] == "leclerc"].iloc[0]
    assert pd.isna(fila_leclerc["posicion_final"])
    assert fila_leclerc["podio"] == False  # noqa: E712
    assert fila_leclerc["circuito_id"] == "monza"
    assert fila_leclerc["constructor_id"] == "ferrari"


def test_construir_tabla_clasificacion():
    tabla = features.construir_tabla_clasificacion(CARRERAS_QUALI_DE_PRUEBA)

    fila_norris = tabla.loc[tabla["piloto_id"] == "norris"].iloc[0]
    assert fila_norris["gap_pole_seg"] == 0.0

    fila_verstappen = tabla.loc[tabla["piloto_id"] == "verstappen"].iloc[0]
    assert round(fila_verstappen["gap_pole_seg"], 3) == 0.4

    # Leclerc no marcó tiempo en Q3: su mejor tiempo debe salir de Q2.
    fila_leclerc = tabla.loc[tabla["piloto_id"] == "leclerc"].iloc[0]
    assert fila_leclerc["mejor_tiempo_seg"] == 80.4


def test_agregar_forma_reciente():
    tabla = pd.DataFrame([
        {"temporada": 2026, "ronda": 1, "piloto_id": "norris", "posicion_final": 1},
        {"temporada": 2026, "ronda": 2, "piloto_id": "norris", "posicion_final": 3},
        {"temporada": 2026, "ronda": 3, "piloto_id": "norris", "posicion_final": 2},
    ])
    resultado = features.agregar_forma_reciente(tabla, ventana=4)

    fila_ronda1 = resultado.loc[resultado["ronda"] == 1].iloc[0]
    assert pd.isna(fila_ronda1["forma_reciente"])  # sin carreras previas

    fila_ronda2 = resultado.loc[resultado["ronda"] == 2].iloc[0]
    assert fila_ronda2["forma_reciente"] == 1.0  # solo la ronda 1

    fila_ronda3 = resultado.loc[resultado["ronda"] == 3].iloc[0]
    assert fila_ronda3["forma_reciente"] == 2.0  # promedio de rondas 1 y 2: (1+3)/2


def test_agregar_historial_circuito():
    tabla = pd.DataFrame([
        {"temporada": 2025, "ronda": 1, "piloto_id": "norris", "constructor_id": "mclaren",
         "circuito_id": "monza", "posicion_final": 1},
        {"temporada": 2025, "ronda": 10, "piloto_id": "norris", "constructor_id": "mclaren",
         "circuito_id": "spa", "posicion_final": 5},
        {"temporada": 2026, "ronda": 14, "piloto_id": "norris", "constructor_id": "mclaren",
         "circuito_id": "monza", "posicion_final": 2},
    ])
    resultado = features.agregar_historial_circuito(tabla)

    fila_monza_2025 = resultado.loc[
        (resultado["temporada"] == 2025) & (resultado["circuito_id"] == "monza")
    ].iloc[0]
    assert pd.isna(fila_monza_2025["historial_piloto_circuito"])  # primera vez en Monza

    fila_monza_2026 = resultado.loc[resultado["temporada"] == 2026].iloc[0]
    assert fila_monza_2026["historial_piloto_circuito"] == 1.0
    assert fila_monza_2026["historial_equipo_circuito"] == 1.0


if __name__ == "__main__":
    test_tiempo_a_segundos()
    test_construir_tabla_resultados()
    test_construir_tabla_clasificacion()
    test_agregar_forma_reciente()
    test_agregar_historial_circuito()
    print("Todas las pruebas de features.py pasaron correctamente.")
