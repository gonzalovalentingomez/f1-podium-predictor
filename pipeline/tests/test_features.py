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


if __name__ == "__main__":
    test_tiempo_a_segundos()
    test_construir_tabla_resultados()
    test_construir_tabla_clasificacion()
    print("Todas las pruebas de features.py pasaron correctamente.")
