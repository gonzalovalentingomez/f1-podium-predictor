"""Construcción del dataset de features para el modelo de podio.

Combina resultados de carrera y clasificación de Jolpica-F1 (vía
`api_client`) con clima histórico (vía `weather`) en una tabla por
piloto/carrera. Por ahora incluye: grid de largada, gap de tiempo contra
la pole, resultado final (con la variable objetivo `podio`), forma
reciente del piloto, historial de piloto/equipo en el circuito,
dificultad de adelantamiento del circuito, delta clasificación/ritmo por
equipo, curva de desarrollo del auto en la temporada, confiabilidad (tasa
de DNF) del equipo y clima (lluvia y temperatura). Queda del brief el
proxy de rookies vía Fórmula 2.

Forma reciente, historial de circuito, delta clasificación/ritmo, curva
de desarrollo y confiabilidad solo usan carreras ANTERIORES a la que se
está prediciendo (nunca la carrera actual), para no filtrar información
del futuro al dataset de entrenamiento. La curva de desarrollo además se
reinicia en cada temporada (es un fenómeno dentro de una temporada). La
dificultad de adelantamiento es la excepción: es una característica
estructural del trazado (no cambia carrera a carrera), así que se calcula
una sola vez sobre todo el histórico disponible, en línea con el brief
("no cambiaron con el reglamento, se puede usar histórico multi-
temporada").
"""

from pathlib import Path

import numpy as np
import pandas as pd

import api_client
import weather


def _a_entero_o_none(valor):
    """Convierte a entero si es posible; si no (ej. 'R' de retirado), None."""
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _a_float_o_none(valor):
    """Convierte a float si es posible; si no, None."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _tiempo_a_segundos(texto: str | None) -> float | None:
    """Convierte un tiempo de clasificación ("1:23.456" o "23.456") a segundos."""
    if not texto:
        return None
    partes = texto.split(":")
    try:
        if len(partes) == 2:
            minutos, segundos = partes
            return int(minutos) * 60 + float(segundos)
        return float(partes[0])
    except ValueError:
        return None


def _mejor_tiempo(resultado_quali: dict) -> float | None:
    """Mejor tiempo (en segundos) entre Q1, Q2 y Q3 de un resultado de clasificación."""
    tiempos = [
        _tiempo_a_segundos(resultado_quali.get(sesion)) for sesion in ("Q1", "Q2", "Q3")
    ]
    tiempos_validos = [tiempo for tiempo in tiempos if tiempo is not None]
    return min(tiempos_validos) if tiempos_validos else None


def construir_tabla_clasificacion(carreras_quali: list) -> pd.DataFrame:
    """Aplana clasificación en una tabla con el gap de tiempo contra la pole.

    Args:
        carreras_quali: Lista de carreras tal como la devuelve
            `api_client.obtener_clasificacion_temporada`.

    Returns:
        DataFrame con columnas: temporada, ronda, piloto_id,
        mejor_tiempo_seg, gap_pole_seg (0 para el poleman).
    """
    filas = []
    for carrera in carreras_quali:
        for resultado in carrera.get("QualifyingResults", []):
            filas.append({
                "temporada": int(carrera["season"]),
                "ronda": int(carrera["round"]),
                "piloto_id": resultado["Driver"]["driverId"],
                "mejor_tiempo_seg": _mejor_tiempo(resultado),
            })

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla

    tiempo_pole = tabla.groupby(["temporada", "ronda"])["mejor_tiempo_seg"].transform("min")
    tabla["gap_pole_seg"] = (tabla["mejor_tiempo_seg"] - tiempo_pole).round(3)
    return tabla


def construir_tabla_resultados(carreras: list) -> pd.DataFrame:
    """Aplana resultados de carrera en una tabla, una fila por piloto/carrera.

    Args:
        carreras: Lista de carreras tal como la devuelve
            `api_client.obtener_resultados_temporada`.

    Returns:
        DataFrame con columnas: temporada, ronda, gran_premio, fecha,
        circuito_id, circuito_nombre, circuito_lat, circuito_lon, hora,
        piloto_id, piloto, constructor_id, constructor, grid,
        posicion_final, puntos, estado y la variable objetivo `podio`
        (True si terminó entre los primeros 3). Coordenadas y hora del
        circuito son un dato intermedio para cruzar clima (ver
        `agregar_clima`), no una feature del modelo en sí.
    """
    filas = []
    for carrera in carreras:
        circuito = carrera.get("Circuit", {})
        ubicacion = circuito.get("Location", {})
        for resultado in carrera.get("Results", []):
            piloto = resultado["Driver"]
            constructor = resultado["Constructor"]
            posicion_final = _a_entero_o_none(resultado.get("position"))

            filas.append({
                "temporada": int(carrera["season"]),
                "ronda": int(carrera["round"]),
                "gran_premio": carrera["raceName"],
                "fecha": carrera["date"],
                "circuito_id": circuito.get("circuitId", ""),
                "circuito_nombre": circuito.get("circuitName", ""),
                "circuito_lat": _a_float_o_none(ubicacion.get("lat")),
                "circuito_lon": _a_float_o_none(ubicacion.get("long")),
                "hora": carrera.get("time"),
                "piloto_id": piloto["driverId"],
                "piloto": f"{piloto['givenName']} {piloto['familyName']}",
                "constructor_id": constructor["constructorId"],
                "constructor": constructor["name"],
                "grid": _a_entero_o_none(resultado.get("grid")),
                "posicion_final": posicion_final,
                "puntos": float(resultado.get("points", 0)),
                "estado": resultado.get("status", ""),
                "podio": posicion_final is not None and posicion_final <= 3,
            })

    return pd.DataFrame(filas)


def agregar_forma_reciente(tabla: pd.DataFrame, ventana: int = 4) -> pd.DataFrame:
    """Agrega la forma reciente de cada piloto: promedio de posición final
    en sus últimas `ventana` carreras, sin contar la carrera actual.

    Args:
        tabla: Tabla de resultados con al menos las columnas temporada,
            ronda, piloto_id, posicion_final.
        ventana: Cantidad de carreras previas a promediar (3-4 según el
            brief; default 4).

    Returns:
        Copia de `tabla` con la columna `forma_reciente` agregada. Es NaN
        cuando el piloto todavía no disputó ninguna carrera previa (ej. su
        primera carrera en el dataset).
    """
    tabla = tabla.sort_values(["piloto_id", "temporada", "ronda"]).copy()
    tabla["forma_reciente"] = tabla.groupby("piloto_id")["posicion_final"].transform(
        lambda serie: serie.shift(1).rolling(ventana, min_periods=1).mean()
    )
    return tabla


def agregar_historial_circuito(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega el historial de piloto y equipo en el circuito de cada carrera.

    Args:
        tabla: Tabla de resultados con al menos las columnas temporada,
            ronda, piloto_id, constructor_id, circuito_id, posicion_final.

    Returns:
        Copia de `tabla` con las columnas `historial_piloto_circuito` y
        `historial_equipo_circuito`: promedio de posición final en visitas
        anteriores de ese piloto/equipo a ese mismo trazado. NaN si no hay
        visitas previas (circuito nuevo en el calendario, o piloto/equipo
        sin historial ahí).
    """
    tabla = tabla.sort_values(["temporada", "ronda"]).copy()

    tabla["historial_piloto_circuito"] = tabla.groupby(["piloto_id", "circuito_id"])[
        "posicion_final"
    ].transform(lambda serie: serie.shift(1).expanding().mean())

    tabla["historial_equipo_circuito"] = tabla.groupby(["constructor_id", "circuito_id"])[
        "posicion_final"
    ].transform(lambda serie: serie.shift(1).expanding().mean())

    return tabla


def _filas_con_resultado_valido(tabla: pd.DataFrame) -> pd.DataFrame:
    """Filtra a las filas con grid y posición final utilizables.

    Descarta abandonos (posición final nula) y salidas desde pit lane
    (grid 0, que no representa una posición de clasificación real).
    """
    tabla = tabla.dropna(subset=["grid", "posicion_final"])
    return tabla[tabla["grid"] > 0]


def calcular_dificultad_adelantamiento(tabla: pd.DataFrame) -> pd.DataFrame:
    """Calcula, por circuito, la dificultad de adelantamiento.

    Se usa como proxy la correlación de Pearson entre grid y posición
    final a lo largo de todo el histórico disponible: cercana a 1 indica
    que el grid predice casi directamente el resultado (poco margen para
    adelantar, ej. Mónaco); valores más bajos indican pistas donde el
    resultado depende menos de la largada.

    Args:
        tabla: Tabla de resultados con grid, posicion_final, circuito_id,
            temporada y ronda (puede combinar varias temporadas).

    Returns:
        DataFrame con columnas circuito_id, dificultad_adelantamiento y
        dificultad_adelantamiento_muestras (cantidad de carreras usadas
        para el cálculo, para poder desconfiar de circuitos con poco
        historial disponible).
    """
    datos_validos = _filas_con_resultado_valido(tabla)

    filas = []
    for circuito_id, grupo in datos_validos.groupby("circuito_id"):
        filas.append({
            "circuito_id": circuito_id,
            "dificultad_adelantamiento": grupo["grid"].corr(grupo["posicion_final"]),
            "dificultad_adelantamiento_muestras": grupo[["temporada", "ronda"]]
            .drop_duplicates()
            .shape[0],
        })
    return pd.DataFrame(filas)


def agregar_dificultad_adelantamiento(tabla: pd.DataFrame) -> pd.DataFrame:
    """Une la dificultad de adelantamiento de cada circuito a la tabla."""
    dificultad = calcular_dificultad_adelantamiento(tabla)
    return tabla.merge(dificultad, on="circuito_id", how="left")


def calcular_curva_desarrollo(tabla: pd.DataFrame, minimo_carreras: int = 2) -> pd.DataFrame:
    """Calcula, por equipo y carrera, la curva de desarrollo del auto.

    Se define como la pendiente de una regresión lineal simple de los
    puntos promedio del equipo por carrera en función de la ronda, usando
    solo carreras ANTERIORES de esa misma temporada (el desarrollo del
    auto es un fenómeno dentro de una temporada, no se arrastra entre
    temporadas). Pendiente positiva: el equipo viene mejorando carrera a
    carrera. Negativa: viene empeorando.

    Args:
        tabla: Tabla de resultados con puntos, constructor_id, temporada
            y ronda.
        minimo_carreras: Cantidad mínima de carreras previas en la
            temporada para poder ajustar una pendiente (2 es el mínimo
            matemático para una regresión lineal).

    Returns:
        DataFrame con columnas temporada, ronda, constructor_id y
        curva_desarrollo. NaN si el equipo todavía no tiene suficientes
        carreras previas en esa temporada.
    """
    por_carrera = (
        tabla.groupby(["temporada", "ronda", "constructor_id"])["puntos"]
        .mean()
        .reset_index()
        .sort_values(["constructor_id", "temporada", "ronda"])
    )

    def _pendientes_del_grupo(grupo: pd.DataFrame) -> pd.Series:
        pendientes = []
        for i in range(len(grupo)):
            previas = grupo.iloc[:i]
            if len(previas) < minimo_carreras:
                pendientes.append(np.nan)
            else:
                pendiente, _ = np.polyfit(previas["ronda"], previas["puntos"], 1)
                pendientes.append(pendiente)
        return pd.Series(pendientes, index=grupo.index)

    pendientes_por_grupo = [
        _pendientes_del_grupo(grupo)
        for _, grupo in por_carrera.groupby(["constructor_id", "temporada"], sort=False)
    ]
    por_carrera["curva_desarrollo"] = pd.concat(pendientes_por_grupo).reindex(por_carrera.index)

    return por_carrera[["temporada", "ronda", "constructor_id", "curva_desarrollo"]]


def agregar_curva_desarrollo(tabla: pd.DataFrame, minimo_carreras: int = 2) -> pd.DataFrame:
    """Une la curva de desarrollo del equipo a la tabla principal."""
    curva = calcular_curva_desarrollo(tabla, minimo_carreras=minimo_carreras)
    return tabla.merge(curva, on=["temporada", "ronda", "constructor_id"], how="left")


def calcular_confiabilidad_equipo(tabla: pd.DataFrame, ventana: int = 4) -> pd.DataFrame:
    """Calcula, por equipo, la tasa de abandonos (DNF) reciente.

    Se considera abandono cualquier resultado cuyo estado no sea
    "Finished" ni "+N Lap(s)" (mismo criterio de "carrera terminada" que
    `analisis.py` del F1 Stats Explorer). Primero se promedia la tasa de
    abandono de los dos autos del equipo en cada carrera, y después se
    toma el promedio móvil de esa tasa en las últimas `ventana` carreras
    previas del equipo (sin incluir la carrera actual).

    No distingue por motor/proveedor de motopropulsor porque Jolpica-F1
    no expone esa relación (solo el constructor/equipo); queda como
    limitación conocida frente al brief.

    Args:
        tabla: Tabla de resultados con estado, constructor_id, temporada
            y ronda.
        ventana: Cantidad de carreras previas del equipo a promediar.

    Returns:
        DataFrame con columnas temporada, ronda, constructor_id y
        tasa_dnf_equipo. NaN si el equipo no tiene carreras previas.
    """
    datos = tabla.copy()
    datos["abandono"] = ~datos["estado"].str.contains(
        "Finished|\\+", case=False, regex=True, na=False
    )

    por_carrera = (
        datos.groupby(["temporada", "ronda", "constructor_id"])["abandono"]
        .mean()
        .reset_index()
        .sort_values(["constructor_id", "temporada", "ronda"])
    )
    por_carrera["tasa_dnf_equipo"] = por_carrera.groupby("constructor_id")["abandono"].transform(
        lambda serie: serie.shift(1).rolling(ventana, min_periods=1).mean()
    )

    return por_carrera[["temporada", "ronda", "constructor_id", "tasa_dnf_equipo"]]


def agregar_confiabilidad_equipo(tabla: pd.DataFrame, ventana: int = 4) -> pd.DataFrame:
    """Une la tasa de abandonos reciente del equipo a la tabla principal."""
    confiabilidad = calcular_confiabilidad_equipo(tabla, ventana=ventana)
    return tabla.merge(confiabilidad, on=["temporada", "ronda", "constructor_id"], how="left")


def agregar_clima(tabla: pd.DataFrame, usar_cache: bool = True) -> pd.DataFrame:
    """Agrega el clima histórico de cada carrera: lluvia (bool) y temperatura (°C).

    Ver `weather.obtener_clima_carrera` para las fuentes usadas (OpenF1
    desde 2023, Open-Meteo Archive para años anteriores). Es clima real,
    no pronóstico: solo tiene sentido para carreras ya disputadas.

    Args:
        tabla: Tabla de resultados con temporada, ronda, fecha, hora,
            circuito_lat, circuito_lon.
        usar_cache: Si es False, fuerza a pedir los datos siempre a la API.

    Returns:
        Copia de `tabla` con las columnas `lluvia` y `temperatura_c`
        agregadas (None si ninguna fuente pudo resolver el clima de esa
        carrera), sin las columnas de hora/coordenadas, que ya cumplieron
        su función.
    """
    carreras = tabla[
        ["temporada", "ronda", "fecha", "hora", "circuito_lat", "circuito_lon"]
    ].drop_duplicates(subset=["temporada", "ronda"])

    filas_clima = []
    for _, carrera in carreras.iterrows():
        clima = weather.obtener_clima_carrera(
            temporada=int(carrera["temporada"]),
            ronda=int(carrera["ronda"]),
            fecha=carrera["fecha"],
            hora=carrera["hora"],
            lat=carrera["circuito_lat"],
            lon=carrera["circuito_lon"],
            usar_cache=usar_cache,
        )
        filas_clima.append({
            "temporada": carrera["temporada"],
            "ronda": carrera["ronda"],
            "lluvia": clima["lluvia"],
            "temperatura_c": clima["temperatura_c"],
        })

    tabla_clima = pd.DataFrame(filas_clima)
    resultado = tabla.merge(tabla_clima, on=["temporada", "ronda"], how="left")
    return resultado.drop(columns=["hora", "circuito_lat", "circuito_lon"])


def calcular_delta_clasificacion_ritmo(tabla: pd.DataFrame, ventana: int = 4) -> pd.DataFrame:
    """Calcula, por equipo, el delta entre clasificación y ritmo de carrera.

    Primero promedia (grid - posición final) de los dos autos de cada
    equipo en cada carrera, y después toma el promedio móvil de ese delta
    en las últimas `ventana` carreras previas del equipo (sin incluir la
    carrera actual). Un delta positivo indica que el equipo suele
    terminar mejor de lo que clasificó (buen ritmo de carrera relativo a
    la clasificación, ej. Alpine); uno negativo indica el patrón inverso
    (ej. Red Bull).

    Args:
        tabla: Tabla de resultados con grid, posicion_final,
            constructor_id, temporada y ronda.
        ventana: Cantidad de carreras previas del equipo a promediar.

    Returns:
        DataFrame con columnas temporada, ronda, constructor_id y
        delta_clasificacion_ritmo. NaN si el equipo no tiene carreras
        previas en el histórico disponible.
    """
    datos_validos = _filas_con_resultado_valido(tabla).copy()
    datos_validos["delta"] = datos_validos["grid"] - datos_validos["posicion_final"]

    por_carrera = (
        datos_validos.groupby(["temporada", "ronda", "constructor_id"])["delta"]
        .mean()
        .reset_index()
        .sort_values(["constructor_id", "temporada", "ronda"])
    )
    por_carrera["delta_clasificacion_ritmo"] = por_carrera.groupby("constructor_id")[
        "delta"
    ].transform(lambda serie: serie.shift(1).rolling(ventana, min_periods=1).mean())

    return por_carrera[["temporada", "ronda", "constructor_id", "delta_clasificacion_ritmo"]]


def agregar_delta_clasificacion_ritmo(tabla: pd.DataFrame, ventana: int = 4) -> pd.DataFrame:
    """Une el delta clasificación/ritmo por equipo a la tabla principal."""
    delta = calcular_delta_clasificacion_ritmo(tabla, ventana=ventana)
    return tabla.merge(delta, on=["temporada", "ronda", "constructor_id"], how="left")


def construir_dataset_base(
    temporadas: list[int],
    usar_cache: bool = True,
    ventana_forma_reciente: int = 4,
    incluir_clima: bool = True,
) -> pd.DataFrame:
    """Arma la tabla base del dataset: resultados, grid, gap a la pole,
    forma reciente, historial de circuito, dificultad de adelantamiento,
    delta clasificación/ritmo, curva de desarrollo, confiabilidad por
    equipo y clima histórico.

    Args:
        temporadas: Años a incluir (ej: [2023, 2024, 2025, 2026]).
        usar_cache: Si es False, fuerza a pedir los datos siempre a la API.
        ventana_forma_reciente: Cantidad de carreras previas a promediar
            para la forma reciente, el delta clasificación/ritmo y la
            confiabilidad del equipo (ver `agregar_forma_reciente`,
            `agregar_delta_clasificacion_ritmo` y
            `agregar_confiabilidad_equipo`).
        incluir_clima: Si es False, omite `agregar_clima` (una llamada de
            red por carrera nueva, más lenta que el resto de las
            features); útil para iterar rápido sin depender de OpenF1 /
            Open-Meteo.

    Returns:
        DataFrame combinado de todas las temporadas pedidas, ordenado por
        temporada, ronda y grid.
    """
    tablas_resultados = []
    tablas_clasificacion = []

    for temporada in temporadas:
        carreras = api_client.obtener_resultados_temporada(temporada, usar_cache=usar_cache)
        carreras_quali = api_client.obtener_clasificacion_temporada(
            temporada, usar_cache=usar_cache
        )
        tablas_resultados.append(construir_tabla_resultados(carreras))
        tablas_clasificacion.append(construir_tabla_clasificacion(carreras_quali))

    resultados = pd.concat(tablas_resultados, ignore_index=True)
    clasificacion = pd.concat(tablas_clasificacion, ignore_index=True)

    dataset = resultados.merge(
        clasificacion[["temporada", "ronda", "piloto_id", "gap_pole_seg"]],
        on=["temporada", "ronda", "piloto_id"],
        how="left",
    )
    dataset = agregar_forma_reciente(dataset, ventana=ventana_forma_reciente)
    dataset = agregar_historial_circuito(dataset)
    dataset = agregar_dificultad_adelantamiento(dataset)
    dataset = agregar_delta_clasificacion_ritmo(dataset, ventana=ventana_forma_reciente)
    dataset = agregar_curva_desarrollo(dataset)
    dataset = agregar_confiabilidad_equipo(dataset, ventana=ventana_forma_reciente)
    if incluir_clima:
        dataset = agregar_clima(dataset, usar_cache=usar_cache)

    return dataset.sort_values(["temporada", "ronda", "grid"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    temporadas_pedidas = [int(arg) for arg in sys.argv[1:]] or [2023, 2024, 2025, 2026]
    dataset = construir_dataset_base(temporadas_pedidas)

    salida = Path(__file__).parent / "data" / "dataset_base.csv"
    salida.parent.mkdir(exist_ok=True)
    dataset.to_csv(salida, index=False)

    print(f"Dataset base: {len(dataset)} filas, temporadas {temporadas_pedidas}.")
    print(f"Guardado en {salida}")
