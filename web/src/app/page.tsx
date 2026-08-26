import { promises as fs } from "fs";
import path from "path";

type Prediccion = {
  piloto_id: string;
  piloto: string;
  constructor_id: string;
  constructor: string;
  grid: number | null;
  probabilidad_podio: number;
};

type PrediccionCarrera = {
  circuito_id: string;
  temporada: number;
  ronda: number;
  gran_premio: string;
  fecha: string;
  clasificacion_disponible: boolean;
  generado_en: string;
  predicciones: Prediccion[];
};

async function obtenerUltimaPrediccion(): Promise<PrediccionCarrera | null> {
  const carpeta = path.join(process.cwd(), "..", "predictions");

  let archivos: string[];
  try {
    archivos = await fs.readdir(carpeta);
  } catch {
    return null;
  }

  // Los nombres de archivo son "{temporada}-{ronda}-{circuito}.json" con la
  // ronda con cero a la izquierda, así que orden alfabético == orden
  // cronológico: el último es la carrera más reciente predicha.
  const archivosJson = archivos.filter((archivo) => archivo.endsWith(".json")).sort();
  if (archivosJson.length === 0) return null;

  const contenido = await fs.readFile(
    path.join(carpeta, archivosJson[archivosJson.length - 1]),
    "utf-8"
  );
  return JSON.parse(contenido) as PrediccionCarrera;
}

function formatearFecha(fecha: string): string {
  return new Date(`${fecha}T00:00:00Z`).toLocaleDateString("es-AR", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}

export default async function Home() {
  const prediccion = await obtenerUltimaPrediccion();

  if (!prediccion) {
    return (
      <main className="flex-1 flex items-center justify-center p-8 text-center">
        <p className="text-neutral-500 dark:text-neutral-400">
          Todavía no hay ninguna predicción generada. Corré{" "}
          <code className="rounded bg-neutral-100 px-1.5 py-0.5 text-sm dark:bg-neutral-800">
            pipeline/predict.py
          </code>{" "}
          para generar una.
        </p>
      </main>
    );
  }

  return (
    <main className="flex-1 mx-auto w-full max-w-2xl px-4 py-10 sm:py-16">
      <header className="mb-8">
        <p className="text-sm font-semibold tracking-wide text-red-600 uppercase">
          F1 Podium Predictor
        </p>
        <h1 className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl">
          {prediccion.gran_premio}
        </h1>
        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
          {formatearFecha(prediccion.fecha)}
        </p>

        {!prediccion.clasificacion_disponible && (
          <p className="mt-4 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800 ring-1 ring-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:ring-amber-900">
            Predicción preliminar: todavía no hay clasificación real para esta
            carrera. El grid y el gap a la pole se completan recién después
            de la clasificación.
          </p>
        )}
      </header>

      <ol className="divide-y divide-neutral-200 overflow-hidden rounded-lg ring-1 ring-neutral-200 dark:divide-neutral-800 dark:ring-neutral-800">
        {prediccion.predicciones.map((piloto, indice) => (
          <li
            key={piloto.piloto_id}
            className={`flex items-center gap-4 px-4 py-3 ${
              indice < 3 ? "bg-red-50/70 dark:bg-red-950/20" : ""
            }`}
          >
            <span className="w-6 text-right text-sm tabular-nums text-neutral-400">
              {indice + 1}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium">{piloto.piloto}</p>
              <p className="truncate text-sm text-neutral-500 dark:text-neutral-400">
                {piloto.constructor}
              </p>
            </div>
            <span className="w-10 text-right text-sm tabular-nums text-neutral-400">
              {piloto.grid ?? "—"}
            </span>
            <span className="w-16 text-right font-semibold tabular-nums">
              {(piloto.probabilidad_podio * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ol>

      <p className="mt-6 text-xs text-neutral-400 dark:text-neutral-500">
        Generado el {new Date(prediccion.generado_en).toLocaleString("es-AR")}
        {" · "}Random Forest entrenado con datos de Jolpica-F1, OpenF1 y
        Open-Meteo.
      </p>
    </main>
  );
}
