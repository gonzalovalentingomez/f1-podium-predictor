"""Pruebas simples para el módulo predict.py.

El resto de la lógica de predict.py se validó de punta a punta contra
las APIs reales (Monza 2026, ver historial de commits); acá solo se
prueba la parte pura que es fácil de aislar sin red.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import predict  # noqa: E402

CALENDARIO_DE_PRUEBA = [
    {"round": "12", "raceName": "Dutch Grand Prix", "Circuit": {"circuitId": "zandvoort"}},
    {"round": "13", "raceName": "Italian Grand Prix", "Circuit": {"circuitId": "monza"}},
]


def test_buscar_carrera_en_calendario():
    carrera = predict._buscar_carrera_en_calendario(CALENDARIO_DE_PRUEBA, "monza")
    assert carrera["raceName"] == "Italian Grand Prix"
    assert carrera["round"] == "13"


def test_buscar_carrera_en_calendario_circuito_inexistente():
    fallo = False
    try:
        predict._buscar_carrera_en_calendario(CALENDARIO_DE_PRUEBA, "circuito_que_no_existe")
    except ValueError:
        fallo = True
    assert fallo


if __name__ == "__main__":
    test_buscar_carrera_en_calendario()
    test_buscar_carrera_en_calendario_circuito_inexistente()
    print("Todas las pruebas de predict.py pasaron correctamente.")
