"""Pruebas simples para el módulo predict.py.

El resto de la lógica de predict.py se validó de punta a punta contra
las APIs reales (Monza 2026, ver historial de commits); acá solo se
prueba la parte pura que es fácil de aislar sin red.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import predict  # noqa: E402
import weather  # noqa: E402

CALENDARIO_DE_PRUEBA = [
    {"round": "12", "raceName": "Dutch Grand Prix", "Circuit": {"circuitId": "zandvoort"}},
    {"round": "13", "raceName": "Italian Grand Prix", "Circuit": {"circuitId": "monza"}},
]

CARRERA_OBJETIVO_DE_PRUEBA = {
    "raceName": "Italian Grand Prix",
    "date": "2026-09-06",
    "Qualifying": {"date": "2026-09-05", "time": "14:00:00Z"},
    "Circuit": {
        "circuitId": "monza",
        "circuitName": "Autodromo Nazionale di Monza",
        "Location": {"lat": "45.6156", "long": "9.28111", "country": "Italy"},
    },
}

ULTIMA_CARRERA_CONOCIDA_DE_PRUEBA = {
    "Results": [
        {"number": "10", "Driver": {"driverId": "gasly", "givenName": "Pierre", "familyName": "Gasly"},
         "Constructor": {"constructorId": "alpine", "name": "Alpine F1 Team"}},
        {"number": "63", "Driver": {"driverId": "russell", "givenName": "George", "familyName": "Russell"},
         "Constructor": {"constructorId": "mercedes", "name": "Mercedes"}},
        {"number": "22", "Driver": {"driverId": "tsunoda", "givenName": "Yuki", "familyName": "Tsunoda"},
         "Constructor": {"constructorId": "rb", "name": "RB F1 Team"}},
    ]
}

# Formas reales devueltas por OpenF1 (verificadas contra la API real de
# la clasificación de Monza 2026): duration = [Q1, Q2, Q3], None en las
# sesiones que el piloto no llegó a correr.
SESIONES_OPENF1_DE_PRUEBA = [
    {"session_key": 11357, "session_name": "Qualifying", "date_start": "2026-09-05T14:00:00+00:00"},
]
SESSION_RESULT_OPENF1_DE_PRUEBA = [
    {"position": 1, "driver_number": 10, "duration": [82.612, 82.077, 81.786]},
    {"position": 2, "driver_number": 63, "duration": [82.779, 82.161, 81.846]},
    {"position": 17, "driver_number": 22, "duration": [83.755, None, None]},
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


def test_alineacion_desde_openf1():
    def _pedir_json_falso(url, parametros):
        if url.endswith("/sessions"):
            return SESIONES_OPENF1_DE_PRUEBA
        if url.endswith("/session_result"):
            return SESSION_RESULT_OPENF1_DE_PRUEBA
        raise AssertionError(f"URL inesperada: {url}")

    original = weather._pedir_json
    weather._pedir_json = _pedir_json_falso
    try:
        alineacion = predict._alineacion_desde_openf1(
            CARRERA_OBJETIVO_DE_PRUEBA, 2026, ULTIMA_CARRERA_CONOCIDA_DE_PRUEBA
        )
    finally:
        weather._pedir_json = original

    assert alineacion is not None
    assert len(alineacion) == 3

    fila_gasly = alineacion.loc[alineacion["piloto_id"] == "gasly"].iloc[0]
    assert fila_gasly["grid"] == 1
    assert fila_gasly["gap_pole_seg"] == 0.0

    fila_russell = alineacion.loc[alineacion["piloto_id"] == "russell"].iloc[0]
    assert fila_russell["grid"] == 2
    assert round(fila_russell["gap_pole_seg"], 3) == 0.06

    # Tsunoda quedó eliminado en Q1 (duration solo tiene el primer valor
    # no nulo); su mejor tiempo debe salir de ahí, no de un Q3 inexistente.
    fila_tsunoda = alineacion.loc[alineacion["piloto_id"] == "tsunoda"].iloc[0]
    assert fila_tsunoda["grid"] == 17
    assert round(fila_tsunoda["gap_pole_seg"], 3) == 1.969


def test_alineacion_desde_openf1_sin_sesion_disponible():
    original = weather._pedir_json
    weather._pedir_json = lambda url, parametros: []  # ninguna sesión encontrada todavía
    try:
        alineacion = predict._alineacion_desde_openf1(
            CARRERA_OBJETIVO_DE_PRUEBA, 2026, ULTIMA_CARRERA_CONOCIDA_DE_PRUEBA
        )
    finally:
        weather._pedir_json = original

    assert alineacion is None


if __name__ == "__main__":
    test_buscar_carrera_en_calendario()
    test_buscar_carrera_en_calendario_circuito_inexistente()
    test_alineacion_desde_openf1()
    test_alineacion_desde_openf1_sin_sesion_disponible()
    print("Todas las pruebas de predict.py pasaron correctamente.")
