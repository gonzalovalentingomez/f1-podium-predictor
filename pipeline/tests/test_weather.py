"""Pruebas simples para el módulo weather.py.

Mismo estilo que `tests/test_analisis.py` del F1 Stats Explorer: sin
framework de testing, funciones de verificación manual con asserts,
pensadas para correrse directamente con: python pipeline/tests/test_weather.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import weather  # noqa: E402


def test_hora_utc_a_entero():
    assert weather._hora_utc_a_entero("13:00:00Z") == 13
    assert weather._hora_utc_a_entero(None) is None
    assert weather._hora_utc_a_entero("") is None


def test_clima_open_meteo_usa_ventana_de_carrera_no_el_dia_completo():
    """Regresión: España 2021 tuvo 0mm durante la carrera (13-15h UTC) pero
    llovió esa noche (21-23h). El resumen diario de Open-Meteo lo contaba
    como "lluvioso"; la ventana horaria acotada a la carrera debe dar seco.
    """
    respuesta_falsa = {
        "hourly": {
            "time": [f"2021-05-09T{h:02d}:00" for h in range(24)],
            "precipitation": [0.0] * 21 + [1.2, 0.3, 0.9],  # llueve recién a las 21-23h
            "temperature_2m": [15.0] * 24,
        }
    }

    pedir_json_original = weather._pedir_json
    leer_de_cache_original = weather._leer_de_cache
    guardar_en_cache_original = weather._guardar_en_cache
    weather._pedir_json = lambda url, parametros: respuesta_falsa
    weather._leer_de_cache = lambda clave: None
    weather._guardar_en_cache = lambda clave, datos: None
    try:
        clima_durante_carrera = weather._clima_open_meteo(
            2021, 4, "2021-05-09", "13:00:00Z", 41.57, 2.26, usar_cache=False
        )
        clima_sin_horario_conocido = weather._clima_open_meteo(
            2021, 4, "2021-05-09", None, 41.57, 2.26, usar_cache=False
        )
    finally:
        weather._pedir_json = pedir_json_original
        weather._leer_de_cache = leer_de_cache_original
        weather._guardar_en_cache = guardar_en_cache_original

    assert clima_durante_carrera["lluvia"] is False
    # Sin horario conocido, cae a considerar el día completo (si llueve en
    # algún momento del día, sí lo cuenta como lluvioso).
    assert clima_sin_horario_conocido["lluvia"] is True


def test_obtener_pronostico_carrera_dentro_de_rango():
    respuesta_falsa = {
        "hourly": {
            "time": [f"2026-09-06T{h:02d}:00" for h in range(24)],
            "precipitation": [0.0] * 24,
            "precipitation_probability": [5] * 13 + [80, 85, 20] + [5] * 8,  # pico 13-15h
            "temperature_2m": [25.0] * 24,
        }
    }

    pedir_json_original = weather._pedir_json
    weather._pedir_json = lambda url, parametros: respuesta_falsa
    try:
        pronostico = weather.obtener_pronostico_carrera(
            "2026-09-06", "13:00:00Z", 45.6156, 9.2811
        )
    finally:
        weather._pedir_json = pedir_json_original

    assert pronostico["lluvia"] is True  # 85% de probabilidad dentro de la ventana 13-15h
    assert pronostico["probabilidad_lluvia_pct"] == 85
    assert pronostico["temperatura_c"] == 25.0


def test_obtener_pronostico_carrera_fuera_de_rango():
    """Fecha fuera del horizonte de pronóstico: la API responde pero sin
    valores útiles para esos índices (huecos de None)."""
    respuesta_falsa = {
        "hourly": {
            "time": ["2026-12-25T00:00"],
            "precipitation": [None],
            "precipitation_probability": [None],
            "temperature_2m": [None],
        }
    }

    pedir_json_original = weather._pedir_json
    weather._pedir_json = lambda url, parametros: respuesta_falsa
    try:
        pronostico = weather.obtener_pronostico_carrera(
            "2026-12-25", "13:00:00Z", 45.6156, 9.2811
        )
    finally:
        weather._pedir_json = pedir_json_original

    assert pronostico == {"lluvia": None, "probabilidad_lluvia_pct": None, "temperatura_c": None}


if __name__ == "__main__":
    test_hora_utc_a_entero()
    test_clima_open_meteo_usa_ventana_de_carrera_no_el_dia_completo()
    test_obtener_pronostico_carrera_dentro_de_rango()
    test_obtener_pronostico_carrera_fuera_de_rango()
    print("Todas las pruebas de weather.py pasaron correctamente.")
