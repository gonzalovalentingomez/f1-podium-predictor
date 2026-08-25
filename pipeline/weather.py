"""Clima histórico por carrera, para la feature de clima del dataset.

Combina dos fuentes gratuitas y sin API key:
- OpenF1 (https://openf1.org): clima real de sesión, cobertura desde 2023.
  Se prioriza para esos años porque son muestras tomadas durante la
  carrera misma (no solo un resumen diario).
- Open-Meteo Archive (https://open-meteo.com): clima histórico diario por
  coordenadas y fecha, con cobertura de décadas. Se usa para carreras
  anteriores a 2023 (donde OpenF1 no tiene datos) y como respaldo si
  OpenF1 no encuentra la sesión.

Importante: `obtener_clima_carrera` da clima HISTÓRICO real (para
entrenar con carreras ya disputadas). Para una carrera futura (ej. Monza
2026, todavía sin resultados en Jolpica-F1) hace falta un PRONÓSTICO en
vez de clima real: eso lo resuelve `obtener_pronostico_carrera`, vía
Open-Meteo Forecast (mismo proveedor que el archivo histórico, pero su
endpoint de pronóstico, con cobertura de ~16 días a futuro). Se usa recién
en el pipeline de predicción (cerca de la fecha de la carrera, para que
el pronóstico sea confiable), no en el armado del dataset de entrenamiento.
"""

import json
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).parent / "data_cache"
TIMEOUT_SEGUNDOS = 15

OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_PRIMER_ANIO = 2023

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
UMBRAL_PROBABILIDAD_LLUVIA_PCT = 50


def _ruta_cache(clave: str) -> Path:
    return CACHE_DIR / f"{clave}.json"


def _guardar_en_cache(clave: str, datos: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(_ruta_cache(clave), "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)


def _leer_de_cache(clave: str) -> dict | None:
    ruta = _ruta_cache(clave)
    if not ruta.exists():
        return None
    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except (json.JSONDecodeError, OSError):
        return None


def _pedir_json(url: str, parametros: dict):
    """Pedido HTTP con manejo de errores no fatal.

    El clima es una feature secundaria del modelo: si la fuente falla o
    no tiene el dato, se devuelve None para que el dataset siga
    armándose con esa columna vacía, en vez de interrumpir todo el
    proceso (a diferencia de `api_client`, donde grid/resultados son
    datos centrales y sí ameritan una excepción).
    """
    try:
        respuesta = requests.get(url, params=parametros, timeout=TIMEOUT_SEGUNDOS)
        respuesta.raise_for_status()
        return respuesta.json()
    except (requests.exceptions.RequestException, ValueError):
        return None


def _clima_openf1(temporada: int, ronda: int, fecha: str, usar_cache: bool) -> dict | None:
    """Clima de la sesión de carrera vía OpenF1 (cobertura desde 2023)."""
    clave = f"clima_openf1_{temporada}_{ronda}"
    if usar_cache:
        cacheado = _leer_de_cache(clave)
        if cacheado is not None:
            return cacheado

    sesiones = _pedir_json(f"{OPENF1_BASE_URL}/sessions", {"year": temporada, "session_name": "Race"})
    if not sesiones:
        return None

    sesion = next(
        (s for s in sesiones if str(s.get("date_start", "")).startswith(fecha)), None
    )
    if sesion is None:
        return None

    muestras = _pedir_json(f"{OPENF1_BASE_URL}/weather", {"session_key": sesion["session_key"]})
    if not muestras:
        return None

    temperaturas = [m["air_temperature"] for m in muestras if m.get("air_temperature") is not None]
    clima = {
        "lluvia": any(m.get("rainfall") for m in muestras),
        "temperatura_c": round(sum(temperaturas) / len(temperaturas), 1) if temperaturas else None,
    }
    _guardar_en_cache(clave, clima)
    return clima


def _hora_utc_a_entero(hora: str | None) -> int | None:
    """Extrae la hora UTC entera de un horario tipo "13:00:00Z" -> 13."""
    if not hora:
        return None
    try:
        return int(hora.split(":")[0])
    except (ValueError, IndexError):
        return None


def _clima_open_meteo(
    temporada: int,
    ronda: int,
    fecha: str,
    hora: str | None,
    lat: float | None,
    lon: float | None,
    usar_cache: bool,
) -> dict | None:
    """Clima vía Open-Meteo Archive, por coordenadas y horario del circuito.

    Usa el pronóstico horario (no el resumen diario) acotado a la ventana
    de la carrera (desde el horario de largada hasta 3 horas después, en
    UTC). El resumen diario sobreestima lluvia: cuenta como "lluvioso" un
    día que llovió de noche aunque la carrera haya sido en seco (caso
    real: España 2021, 0mm durante la carrera pero lluvia esa noche).

    Si no se conoce el horario de la carrera (algunas temporadas viejas
    no lo tienen en Jolpica-F1), cae a considerar el día completo.
    """
    clave = f"clima_openmeteo_{temporada}_{ronda}"
    if usar_cache:
        cacheado = _leer_de_cache(clave)
        if cacheado is not None:
            return cacheado

    if lat is None or lon is None:
        return None

    parametros = {
        "latitude": lat,
        "longitude": lon,
        "start_date": fecha,
        "end_date": fecha,
        "hourly": "precipitation,temperature_2m",
        "timezone": "UTC",
    }
    datos = _pedir_json(OPEN_METEO_ARCHIVE_URL, parametros)
    if datos is None:
        return None

    try:
        precipitaciones = datos["hourly"]["precipitation"]
        temperaturas = datos["hourly"]["temperature_2m"]
    except KeyError:
        return None

    hora_inicio = _hora_utc_a_entero(hora)
    if hora_inicio is None:
        indices_ventana = range(len(precipitaciones))
    else:
        indices_ventana = range(hora_inicio, min(hora_inicio + 3, len(precipitaciones)))

    precip_ventana = [precipitaciones[i] for i in indices_ventana if precipitaciones[i] is not None]
    temp_ventana = [temperaturas[i] for i in indices_ventana if temperaturas[i] is not None]

    clima = {
        "lluvia": bool(precip_ventana) and sum(precip_ventana) > 0,
        "temperatura_c": round(sum(temp_ventana) / len(temp_ventana), 1) if temp_ventana else None,
    }
    _guardar_en_cache(clave, clima)
    return clima


def obtener_clima_carrera(
    temporada: int,
    ronda: int,
    fecha: str,
    hora: str | None,
    lat: float | None,
    lon: float | None,
    usar_cache: bool = True,
) -> dict:
    """Clima histórico de una carrera puntual: lluvia y temperatura.

    Prioriza OpenF1 para temporadas >= 2023; si esa temporada es
    anterior, o si OpenF1 no encuentra la sesión, cae a Open-Meteo
    Archive usando las coordenadas y el horario del circuito.

    Args:
        temporada, ronda: identifican la carrera (clave de caché).
        fecha: fecha de la carrera en formato YYYY-MM-DD.
        hora: horario de largada en UTC, formato "13:00:00Z" (puede ser
            None en temporadas viejas donde Jolpica-F1 no lo registra).
        lat, lon: coordenadas del circuito (Jolpica-F1 Circuit.Location).
        usar_cache: si es False, fuerza a pedir los datos siempre a la API.

    Returns:
        Diccionario {"lluvia": bool | None, "temperatura_c": float | None}.
        Ambos valores son None si ninguna fuente pudo resolver el clima
        de esa carrera (se trata como dato faltante, no como error).
    """
    if temporada >= OPENF1_PRIMER_ANIO:
        clima = _clima_openf1(temporada, ronda, fecha, usar_cache)
        if clima is not None:
            return clima

    clima = _clima_open_meteo(temporada, ronda, fecha, hora, lat, lon, usar_cache)
    return clima if clima is not None else {"lluvia": None, "temperatura_c": None}


def obtener_pronostico_carrera(
    fecha: str, hora: str | None, lat: float | None, lon: float | None
) -> dict:
    """Pronóstico de clima para una carrera futura, vía Open-Meteo Forecast.

    A diferencia de `obtener_clima_carrera`, NO se cachea en disco: un
    pronóstico cambia día a día a medida que se acerca la fecha, así que
    guardarlo permanentemente daría un dato viejo y engañoso. Solo tiene
    cobertura confiable dentro de los próximos ~16 días; para fechas más
    lejanas, la API puede no devolver datos.

    Args:
        fecha: fecha de la carrera en formato YYYY-MM-DD.
        hora: horario de largada en UTC, formato "13:00:00Z" (si es None,
            se considera el día completo en vez de una ventana horaria).
        lat, lon: coordenadas del circuito (Jolpica-F1 Circuit.Location).

    Returns:
        Diccionario {"lluvia": bool | None, "probabilidad_lluvia_pct":
        float | None, "temperatura_c": float | None}. Todos los valores
        son None si la fecha está fuera del rango de pronóstico
        disponible o si la fuente no respondió.
    """
    if lat is None or lon is None:
        return {"lluvia": None, "probabilidad_lluvia_pct": None, "temperatura_c": None}

    parametros = {
        "latitude": lat,
        "longitude": lon,
        "start_date": fecha,
        "end_date": fecha,
        "hourly": "precipitation,precipitation_probability,temperature_2m",
        "timezone": "UTC",
    }
    datos = _pedir_json(OPEN_METEO_FORECAST_URL, parametros)
    if datos is None:
        return {"lluvia": None, "probabilidad_lluvia_pct": None, "temperatura_c": None}

    try:
        precipitaciones = datos["hourly"]["precipitation"]
        probabilidades = datos["hourly"]["precipitation_probability"]
        temperaturas = datos["hourly"]["temperature_2m"]
    except KeyError:
        return {"lluvia": None, "probabilidad_lluvia_pct": None, "temperatura_c": None}

    hora_inicio = _hora_utc_a_entero(hora)
    if hora_inicio is None:
        indices_ventana = range(len(precipitaciones))
    else:
        indices_ventana = range(hora_inicio, min(hora_inicio + 3, len(precipitaciones)))

    precip_ventana = [precipitaciones[i] for i in indices_ventana if precipitaciones[i] is not None]
    prob_ventana = [probabilidades[i] for i in indices_ventana if probabilidades[i] is not None]
    temp_ventana = [temperaturas[i] for i in indices_ventana if temperaturas[i] is not None]

    if not precip_ventana and not prob_ventana:
        # Fecha fuera del rango de pronóstico: la API devuelve la
        # estructura pero sin valores útiles para esos índices.
        return {"lluvia": None, "probabilidad_lluvia_pct": None, "temperatura_c": None}

    probabilidad_maxima = max(prob_ventana) if prob_ventana else None

    return {
        "lluvia": bool(
            (precip_ventana and sum(precip_ventana) > 0)
            or (probabilidad_maxima is not None and probabilidad_maxima >= UMBRAL_PROBABILIDAD_LLUVIA_PCT)
        ),
        "probabilidad_lluvia_pct": probabilidad_maxima,
        "temperatura_c": round(sum(temp_ventana) / len(temp_ventana), 1) if temp_ventana else None,
    }
