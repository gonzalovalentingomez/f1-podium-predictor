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
(sábado). Si todavía no hay clasificación real para la carrera pedida,
la predicción es "preliminar": usa la alineación de pilotos/equipos de
la última carrera disputada como aproximación, y deja grid/gap_pole_seg
vacíos (el modelo los imputa con la mediana, igual que en entrenamiento).
Conviene volver a correr esto después de la clasificación real.
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


def _buscar_carrera_en_calendario(calendario: list, circuito_id: str) -> dict:
    """Busca la carrera de un circuito dado en el calendario de una temporada."""
    carrera = next(
        (c for c in calendario if c.get("Circuit", {}).get("circuitId") == circuito_id), None
    )
    if carrera is None:
        raise ValueError(f"No se encontró el circuito '{circuito_id}' en ese calendario.")
    return carrera


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
        usar_cache: Si es False, fuerza a pedir todos los datos a las APIs.

    Returns:
        Tupla (tabla, metadata). `tabla` tiene una fila por piloto
        esperado, con las mismas columnas de features que usa `model.py`
        (más piloto/constructor legibles). `metadata` trae ronda,
        gran_premio, fecha y si la clasificación real ya estaba
        disponible (predicción preliminar si no).
    """
    if temporada_objetivo not in temporadas_historicas:
        raise ValueError("temporadas_historicas debe incluir a temporada_objetivo.")

    calendario = api_client.obtener_calendario_temporada(temporada_objetivo, usar_cache=usar_cache)
    carrera_objetivo = _buscar_carrera_en_calendario(calendario, circuito_id)
    ronda_objetivo = int(carrera_objetivo["round"])
    circuito = carrera_objetivo.get("Circuit", {})
    ubicacion = circuito.get("Location", {})

    tablas_resultados, tablas_clasificacion = [], []
    for temporada in temporadas_historicas:
        carreras = api_client.obtener_resultados_temporada(temporada, usar_cache=usar_cache)
        tablas_resultados.append(features.construir_tabla_resultados(carreras))
        carreras_quali = api_client.obtener_clasificacion_temporada(temporada, usar_cache=usar_cache)
        tablas_clasificacion.append(features.construir_tabla_clasificacion(carreras_quali))

    resultados = pd.concat(tablas_resultados, ignore_index=True)
    clasificacion = pd.concat(tablas_clasificacion, ignore_index=True)

    hay_clasificacion_objetivo = not clasificacion[
        (clasificacion["temporada"] == temporada_objetivo) & (clasificacion["ronda"] == ronda_objetivo)
    ].empty

    if hay_clasificacion_objetivo:
        carreras_quali_objetivo = api_client.obtener_clasificacion_temporada(
            temporada_objetivo, usar_cache=usar_cache
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
        # Sin clasificación todavía: se aproxima con la alineación de la
        # última carrera DISPUTADA (mejor estimación disponible de quién
        # corre). El grid queda vacío; el modelo lo imputa con la
        # mediana, igual que hace con cualquier NaN en entrenamiento.
        carreras_previas = api_client.obtener_resultados_temporada(
            temporada_objetivo, usar_cache=usar_cache
        )
        if not carreras_previas:
            raise ValueError(
                f"No hay clasificación ni carreras disputadas en {temporada_objetivo} "
                "para aproximar la alineación de pilotos."
            )
        ultima_carrera = max(carreras_previas, key=lambda c: int(c["round"]))
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

    stub = alineacion.copy()
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
        "clasificacion_disponible": hay_clasificacion_objetivo,
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

    if not metadata["clasificacion_disponible"]:
        print(
            "AVISO: todavía no hay clasificación real para esta carrera. La "
            "predicción es preliminar (sin grid ni gap a la pole; alineación "
            "aproximada con la última carrera disputada). Conviene correr esto "
            "de nuevo después de la clasificación real.\n"
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
