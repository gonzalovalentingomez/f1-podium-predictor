"""Genera la predicción de podio para una carrera puntual (ej. Monza 2026).

Usa el modelo ya entrenado (`model.py` / `modelo_podio.joblib`): NO
reentrena nada acá, solo arma la fila de features de la carrera pedida y
se la pasa al modelo para obtener una probabilidad de podio por piloto.

Reusa las funciones de `features.py`: arma la tabla histórica cruda +
una fila "stub" por cada piloto esperado en la carrera a predecir (mismo
temporada/ronda/circuito, sin resultado todavía) y corre encima las
mismas funciones de forma reciente / historial / etc. que usa el
dataset de entrenamiento. Así el shift(1) que ya usan esas funciones
excluye automáticamente la carrera a predecir de su propio cálculo, sin
duplicar la lógica de "no filtrar el futuro".

El grid y el gap a la pole solo existen después de la clasificación
(sábado). Jolpica-F1 suele tardar en publicarla; mientras tanto se
intenta con OpenF1 (que la tiene casi apenas termina la sesión) como
alternativa más rápida. Si ninguna de las dos la tiene todavía, la
predicción es "preliminar": usa la alineación de pilotos/equipos de la
última carrera disputada como aproximación, y deja grid/gap_pole_seg
vacíos (el modelo los imputa con la mediana, igual que en
entrenamiento). Conviene volver a correr esto después de que la
clasificación esté disponible en alguna de las dos fuentes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

import api_client
import features
import model
import weather

OPENF1_BASE_URL = "https://api.openf1.org/v1"


def _buscar_carrera_en_calendario(calendario: list, circuito_id: str) -> dict:
    """Busca la carrera de un circuito dado en el calendario de una temporada."""
    carrera = next(
        (c for c in calendario if c.get("Circuit", {}).get("circuitId") == circuito_id), None
    )
    if carrera is None:
        raise ValueError(f"No se encontró el circuito '{circuito_id}' en ese calendario.")
    return carrera


def _alineacion_desde_openf1(
    carrera_objetivo: dict, temporada_objetivo: int, ultima_carrera_conocida: dict
) -> pd.DataFrame | None:
    """Arma la alineación (grid real) desde OpenF1, cuando Jolpica-F1
    todavía no publicó la clasificación. OpenF1 suele tener la sesión
    disponible casi apenas termina, a diferencia de Jolpica que puede
    tardar horas o más en ingerirla.

    OpenF1 identifica pilotos por número de auto, no por driverId como
    Jolpica; se cruza contra los números de auto de la última carrera
    DISPUTADA en Jolpica (`ultima_carrera_conocida`) para recuperar
    driver_id/constructor_id y nombres legibles. Un piloto que debuta
    justo en esta carrera (sin número conocido todavía en Jolpica) queda
    afuera de la alineación: es una limitación aceptada de este cruce.

    Returns:
        DataFrame con piloto_id, piloto, constructor_id, constructor,
        grid y gap_pole_seg. None si OpenF1 tampoco tiene la sesión
        todavía, o si no se pudo cruzar ningún piloto.
    """
    fecha_quali = carrera_objetivo.get("Qualifying", {}).get("date")
    pais = carrera_objetivo.get("Circuit", {}).get("Location", {}).get("country")
    if not fecha_quali or not pais:
        return None

    sesiones = weather._pedir_json(
        f"{OPENF1_BASE_URL}/sessions",
        {"year": temporada_objetivo, "session_name": "Qualifying", "country_name": pais},
    )
    sesion = next(
        (s for s in sesiones or [] if str(s.get("date_start", "")).startswith(fecha_quali)), None
    )
    if sesion is None:
        return None

    resultados = weather._pedir_json(
        f"{OPENF1_BASE_URL}/session_result", {"session_key": sesion["session_key"]}
    )
    if not resultados:
        return None

    numero_a_piloto = {
        r["number"]: {
            "piloto_id": r["Driver"]["driverId"],
            "piloto": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
            "constructor_id": r["Constructor"]["constructorId"],
            "constructor": r["Constructor"]["name"],
        }
        for r in ultima_carrera_conocida["Results"]
    }

    def _mejor_tiempo(resultado: dict) -> float | None:
        # `duration` trae [Q1, Q2, Q3]; None en las sesiones que el
        # piloto no alcanzó a correr (eliminado antes). El último valor
        # no nulo es su mejor vuelta relevante.
        return next((d for d in reversed(resultado["duration"]) if d is not None), None)

    tiempo_pole = next(
        (_mejor_tiempo(r) for r in resultados if r["position"] == 1), None
    )

    filas = []
    for resultado in resultados:
        piloto = numero_a_piloto.get(str(resultado["driver_number"]))
        if piloto is None:
            continue
        mejor_tiempo = _mejor_tiempo(resultado)
        gap_pole_seg = (
            round(mejor_tiempo - tiempo_pole, 3)
            if mejor_tiempo is not None and tiempo_pole is not None
            else None
        )
        filas.append({**piloto, "grid": resultado["position"], "gap_pole_seg": gap_pole_seg})

    return pd.DataFrame(filas) if filas else None


def construir_fila_prediccion(
    temporadas_historicas: list[int],
    temporada_objetivo: int,
    circuito_id: str,
    usar_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Arma la tabla de features de cada piloto esperado en la carrera pedida.

    Args:
        temporadas_historicas: Temporadas de contexto para calcular forma
            reciente, historial de circuito, etc. Debe incluir
            `temporada_objetivo` (las carreras ya disputadas de esa
            temporada cuentan como historial reciente).
        temporada_objetivo: Temporada de la carrera a predecir.
        circuito_id: circuitId de Jolpica-F1 (ej. "monza").
        usar_cache: Si es False, fuerza a pedir también las temporadas
            históricas a las APIs (la temporada objetivo se pide fresca
            siempre, sin importar este parámetro; ver nota arriba).

    Returns:
        Tupla (tabla, metadata). `tabla` tiene una fila por piloto
        esperado, con las mismas columnas de features que usa `model.py`
        (más piloto/constructor legibles). `metadata` trae ronda,
        gran_premio, fecha, `clasificacion_disponible` (True si hay grid
        real de alguna fuente) y `fuente_grid` ("jolpica", "openf1" o
        "estimada", ver arriba).
    """
    if temporada_objetivo not in temporadas_historicas:
        raise ValueError("temporadas_historicas debe incluir a temporada_objetivo.")

    # La temporada objetivo se pide siempre fresca a la API, nunca de
    # caché: es justamente la que puede tener novedades (clasificación
    # recién publicada) entre una corrida de predict.py y la siguiente.
    # Las demás temporadas son historia cerrada, no cambian, así que sí
    # conviene cachearlas.
    calendario = api_client.obtener_calendario_temporada(temporada_objetivo, usar_cache=False)
    carrera_objetivo = _buscar_carrera_en_calendario(calendario, circuito_id)
    ronda_objetivo = int(carrera_objetivo["round"])
    circuito = carrera_objetivo.get("Circuit", {})
    ubicacion = circuito.get("Location", {})

    tablas_resultados, tablas_clasificacion = [], []
    for temporada in temporadas_historicas:
        usar_cache_temporada = usar_cache and temporada != temporada_objetivo
        carreras = api_client.obtener_resultados_temporada(temporada, usar_cache=usar_cache_temporada)
        tablas_resultados.append(features.construir_tabla_resultados(carreras))
        carreras_quali = api_client.obtener_clasificacion_temporada(
            temporada, usar_cache=usar_cache_temporada
        )
        tablas_clasificacion.append(features.construir_tabla_clasificacion(carreras_quali))

    resultados = pd.concat(tablas_resultados, ignore_index=True)
    clasificacion = pd.concat(tablas_clasificacion, ignore_index=True)

    hay_clasificacion_objetivo = not clasificacion[
        (clasificacion["temporada"] == temporada_objetivo) & (clasificacion["ronda"] == ronda_objetivo)
    ].empty

    if hay_clasificacion_objetivo:
        fuente_grid = "jolpica"
        carreras_quali_objetivo = api_client.obtener_clasificacion_temporada(
            temporada_objetivo, usar_cache=False
        )
        quali_cruda = next(c for c in carreras_quali_objetivo if int(c["round"]) == ronda_objetivo)
        alineacion = pd.DataFrame([
            {
                "piloto_id": r["Driver"]["driverId"],
                "piloto": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                "constructor_id": r["Constructor"]["constructorId"],
                "constructor": r["Constructor"]["name"],
                "grid": features._a_entero_o_none(r.get("position")),
            }
            for r in quali_cruda["QualifyingResults"]
        ])
    else:
        # Jolpica-F1 todavía no publicó la clasificación. La última
        # carrera DISPUTADA sirve como mejor estimación de quién corre
        # (para el cruce con OpenF1, y como último recurso si tampoco
        # OpenF1 la tiene todavía).
        carreras_previas = api_client.obtener_resultados_temporada(
            temporada_objetivo, usar_cache=False
        )
        if not carreras_previas:
            raise ValueError(
                f"No hay clasificación ni carreras disputadas en {temporada_objetivo} "
                "para aproximar la alineación de pilotos."
            )
        ultima_carrera = max(carreras_previas, key=lambda c: int(c["round"]))

        alineacion = _alineacion_desde_openf1(carrera_objetivo, temporada_objetivo, ultima_carrera)
        if alineacion is not None:
            fuente_grid = "openf1"
        else:
            # Ninguna de las dos fuentes tiene la clasificación todavía:
            # se aproxima la alineación sin grid (el modelo lo imputa
            # con la mediana, igual que hace con cualquier NaN).
            fuente_grid = "estimada"
            alineacion = pd.DataFrame([
                {
                    "piloto_id": r["Driver"]["driverId"],
                    "piloto": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                    "constructor_id": r["Constructor"]["constructorId"],
                    "constructor": r["Constructor"]["name"],
                    "grid": None,
                }
                for r in ultima_carrera["Results"]
            ])

    stub = alineacion[["piloto_id", "piloto", "constructor_id", "constructor", "grid"]].copy()
    stub["temporada"] = temporada_objetivo
    stub["ronda"] = ronda_objetivo
    stub["gran_premio"] = carrera_objetivo["raceName"]
    stub["fecha"] = carrera_objetivo["date"]
    stub["circuito_id"] = circuito.get("circuitId", "")
    stub["circuito_nombre"] = circuito.get("circuitName", "")
    stub["posicion_final"] = None
    stub["puntos"] = 0.0
    stub["estado"] = ""
    stub["podio"] = False

    combinado = pd.concat([resultados, stub], ignore_index=True)
    combinado = combinado.merge(
        clasificacion[["temporada", "ronda", "piloto_id", "gap_pole_seg"]],
        on=["temporada", "ronda", "piloto_id"],
        how="left",
    )

    if fuente_grid == "openf1":
        # El merge de arriba no puede traer el gap a la pole (Jolpica no
        # tiene esta clasificación todavía); se completa acá con el que
        # ya viene calculado en `alineacion` desde OpenF1.
        mapa_gap = alineacion.set_index("piloto_id")["gap_pole_seg"]
        es_fila_objetivo = (combinado["temporada"] == temporada_objetivo) & (
            combinado["ronda"] == ronda_objetivo
        )
        combinado.loc[es_fila_objetivo, "gap_pole_seg"] = combinado.loc[
            es_fila_objetivo, "piloto_id"
        ].map(mapa_gap)

    combinado = features.agregar_forma_reciente(combinado)
    combinado = features.agregar_historial_circuito(combinado)
    combinado = features.agregar_dificultad_adelantamiento(combinado)
    combinado = features.agregar_delta_clasificacion_ritmo(combinado)
    combinado = features.agregar_curva_desarrollo(combinado)
    combinado = features.agregar_confiabilidad_equipo(combinado)

    fila_objetivo = combinado[
        (combinado["temporada"] == temporada_objetivo) & (combinado["ronda"] == ronda_objetivo)
    ].copy()

    pronostico = weather.obtener_pronostico_carrera(
        fecha=carrera_objetivo["date"],
        hora=carrera_objetivo.get("time"),
        lat=features._a_float_o_none(ubicacion.get("lat")),
        lon=features._a_float_o_none(ubicacion.get("long")),
    )
    fila_objetivo["lluvia"] = pronostico["lluvia"]
    fila_objetivo["temperatura_c"] = pronostico["temperatura_c"]

    metadata = {
        "ronda": ronda_objetivo,
        "gran_premio": carrera_objetivo["raceName"],
        "fecha": carrera_objetivo["date"],
        "clasificacion_disponible": fuente_grid in ("jolpica", "openf1"),
        "fuente_grid": fuente_grid,  # "jolpica" | "openf1" | "estimada"
    }
    return fila_objetivo.reset_index(drop=True), metadata


def predecir_carrera(
    temporadas_historicas: list[int],
    temporada_objetivo: int,
    circuito_id: str,
    ruta_modelo: Path | None = None,
    usar_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Genera la predicción de podio para una carrera puntual.

    Args:
        temporadas_historicas, temporada_objetivo, circuito_id, usar_cache:
            ver `construir_fila_prediccion`.
        ruta_modelo: Ruta al modelo entrenado (`.joblib`); por default
            `pipeline/data/modelo_podio.joblib`, el que genera `model.py`.

    Returns:
        Tupla (predicciones, metadata). `predicciones` tiene piloto,
        constructor, grid y `probabilidad_podio`, ordenado de mayor a
        menor probabilidad. `metadata`: ver `construir_fila_prediccion`.
    """
    fila_objetivo, metadata = construir_fila_prediccion(
        temporadas_historicas, temporada_objetivo, circuito_id, usar_cache=usar_cache
    )

    ruta_modelo = ruta_modelo or Path(__file__).parent / "data" / "modelo_podio.joblib"
    pipeline_modelo = joblib.load(ruta_modelo)

    X, _, _ = model.preparar_datos(fila_objetivo)
    probabilidades = pipeline_modelo.predict_proba(X)[:, 1]

    predicciones = fila_objetivo[["piloto_id", "piloto", "constructor_id", "constructor", "grid"]].copy()
    predicciones["probabilidad_podio"] = probabilidades

    return predicciones.sort_values("probabilidad_podio", ascending=False).reset_index(drop=True), metadata


if __name__ == "__main__":
    import sys

    circuito_objetivo = sys.argv[1] if len(sys.argv) > 1 else "monza"
    temporada_objetivo = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    temporadas_historicas = list(range(2019, temporada_objetivo + 1))

    predicciones, metadata = predecir_carrera(temporadas_historicas, temporada_objetivo, circuito_objetivo)

    if metadata["fuente_grid"] == "openf1":
        print(
            "AVISO: el grid es real, pero viene de OpenF1 (Jolpica-F1 todavía no "
            "publicó esta clasificación). Conviene correr esto de nuevo más "
            "adelante para confirmar contra Jolpica-F1.\n"
        )
    elif metadata["fuente_grid"] == "estimada":
        print(
            "AVISO: todavía no hay clasificación real en ninguna fuente (ni "
            "Jolpica-F1 ni OpenF1). La predicción es preliminar (sin grid ni "
            "gap a la pole; alineación aproximada con la última carrera "
            "disputada). Conviene correr esto de nuevo más tarde.\n"
        )

    print(f"{metadata['gran_premio']} ({metadata['fecha']}):")
    print(predicciones[["piloto", "constructor", "grid", "probabilidad_podio"]].to_string(index=False))

    salida_json = {
        "circuito_id": circuito_objetivo,
        "temporada": temporada_objetivo,
        **metadata,
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "predicciones": predicciones.to_dict(orient="records"),
    }

    repo_raiz = Path(__file__).parent.parent
    nombre_archivo = f"{temporada_objetivo}-{metadata['ronda']:02d}-{circuito_objetivo}.json"

    # Copia "canónica" versionada, en la raíz del repo: registro histórico
    # de qué se predijo y cuándo, independiente de cómo esté organizada
    # la interfaz.
    carpeta_predicciones = repo_raiz / "predictions"
    carpeta_predicciones.mkdir(exist_ok=True)
    salida = carpeta_predicciones / nombre_archivo
    with open(salida, "w", encoding="utf-8") as archivo:
        json.dump(salida_json, archivo, ensure_ascii=False, indent=2)
    print(f"\nPredicción guardada en {salida}")

    # Copia dentro de web/public/: la interfaz Next.js la lee de ahí para
    # que el proyecto sea autocontenido al deployar (Vercel, con la raíz
    # del proyecto en web/, no ve archivos fuera de esa carpeta).
    carpeta_web = repo_raiz / "web" / "public" / "predictions"
    if carpeta_web.parent.exists():  # no falla si todavía no se scaffoldeó web/
        carpeta_web.mkdir(exist_ok=True)
        with open(carpeta_web / nombre_archivo, "w", encoding="utf-8") as archivo:
            json.dump(salida_json, archivo, ensure_ascii=False, indent=2)
        print(f"Copia para la interfaz guardada en {carpeta_web / nombre_archivo}")
