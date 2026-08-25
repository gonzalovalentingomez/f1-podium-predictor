# F1 v2 — Proyecto de Modelo Predictivo

Extensión del **F1 Stats Explorer** (Streamlit + Jolpica-F1 API + pandas), capstone de Programación 1 en ISTEA, ya deployado en GitHub.

## Objetivo
Predecir si un piloto termina en el podio (top 3) en la próxima carrera. Empezar con clasificación binaria (podio sí/no) antes de complicarlo.

## Milestone inicial
Predicción del **GP de Italia (Monza), 4-6 de septiembre 2026**. Correr el modelo antes de la carrera, guardar la predicción, y comparar contra el resultado real el domingo.

## Por qué este enfoque
- F1 Insights (AWS) se investigó como caso de referencia (Tarea #1 de Minería de Datos I), pero sus métricas en vivo (Battle Forecast, Undercut Threat, desgaste de neumáticos) dependen de telemetría de 300 sensores por auto que no es pública.
- Este proyecto va por otro camino, igual de legítimo: predicción *pre-carrera* con datos históricos públicos (resultados, clasificación, vueltas, pit stops), el mismo tipo de dataset que se usa en investigación académica y modelos de casas de apuestas.
- El principio de algunos insights de AWS sí se puede aproximar: pit stops (Jolpica-F1 tiene vuelta y duración) y gaps entre autos por vuelta.

## Features a incluir

1. **Posición de clasificación (grid)** — y el gap de tiempo contra la pole, no solo la posición cruda.
2. **Forma reciente** — promedio de resultados de las últimas 3-4 carreras por piloto.
3. **Historial en el circuito específico** — resultados pasados de ese piloto/equipo en ese trazado.
4. **Dificultad de adelantamiento del circuito** — Mónaco y trazados lentos/sinuosos con rectas cortas tienen alta correlación grid-resultado final (pole casi asegura victoria). Se puede derivar empíricamente (correlación histórica grid vs. posición final por circuito) y contrastar con criterio propio.
5. **Delta clasificación vs. ritmo de carrera por equipo** — captura casos como Alpine (buen ritmo de carrera, floja clasificación) vs. Red Bull (patrón inverso). Se calcula como la diferencia promedio entre posición de clasificación y posición de carrera en las últimas carreras, por equipo.
6. **Curva de desarrollo del auto en la temporada** — tendencia de puntos/resultados del equipo carrera a carrera.
7. **Confiabilidad** — tasa de abandonos (DNF) por equipo/motor.
8. **Clima** — no está en Jolpica-F1; se obtiene de **OpenF1** (datos desde 2023, cubre toda la temporada 2026), que trae clima por sesión sin necesidad de una API genérica aparte.
9. **Pilotos rookies / poco historial en F1** — usar resultados de Fórmula 2 (y en menor medida F3) como proxy. Resuelve el problema de "cold start" para pilotos nuevos.

## Ponderación por el cambio de reglamento 2026
El nuevo reglamento (motores 50/50 batería-combustión) invalida buena parte del histórico de rendimiento de auto/equipo, aunque no todo:

- **Features de equipo/piloto** (forma reciente, ritmo, confiabilidad): priorizar fuerte los datos de 2026, peso bajo o nulo a temporadas anteriores.
- **Features de circuito** (dificultad de adelantamiento, características del trazado): no cambiaron con el reglamento — se puede usar histórico multi-temporada.

## Enfoque de modelado sugerido
- Arrancar simple: regresión logística o Random Forest Classifier (scikit-learn) para podio sí/no.
- Iterar después: comparar algoritmos, calibración de probabilidades — en línea con contenidos de Minería de Datos I / Modelos Analíticos (ISTEA, 2do cuatrimestre).

## Stack técnico
- **Python + pandas + scikit-learn** para el pipeline de datos y el modelo (regresión logística / Random Forest) — corre localmente o en un job programado, y genera un JSON con las predicciones actualizadas.
- **Next.js (React)** para la interfaz, leyendo ese JSON — aprovecha y fortalece el conocimiento previo en HTML/CSS/React, y da mucho más control visual que Streamlit.
- Proyecto **separado** del F1 Stats Explorer (repo nuevo), ya que Streamlit y Next.js no se mezclan bien en un mismo proyecto. Quedan como dos piezas de portfolio distintas y completas.
- Orden de trabajo sugerido: primero dejar el modelo funcionando y prediciendo (lo mínimo para Monza), la interfaz linda se suma después sin presión de fecha — importante dado el tiempo limitado entre cursada y vida.

## APIs: permisos, límites y fuentes de datos
- **Jolpica-F1** (histórico desde 1950): sigue siendo la mejor opción para forma reciente, historial de circuito y gap de clasificación — no es solo continuidad con el proyecto anterior. Open source, gratuita, sin API key para uso básico. Términos de uso simples: no saturar la API. Límites sin autenticación: **4 requests/segundo, 500/hora** — cachear los datos localmente (actualizar después de cada clasificación/carrera) en vez de consultar en cada carga. Es un proyecto comunitario (no corporativo), riesgo bajo pero no nulo de discontinuidad a futuro (como pasó con Ergast).
- **OpenF1** (datos desde 2023, cubre toda la temporada 2026): gratuita, sin auth. Se suma específicamente para **clima por sesión**, resolviendo ese feature sin necesidad de una API de clima genérica.

## Despliegue
- **Vercel** (gratis, integración nativa con Next.js, deploy automático desde GitHub) para el proyecto nuevo — da una URL fija (`*.vercel.app`) accesible desde cualquier dispositivo con navegador, sin instalar dependencias nunca más.
- Opcional: desplegar también el F1 Stats Explorer en Streamlit Community Cloud (gratis) para resolver el mismo problema en ese proyecto.

## Fuera de alcance por ahora
- **App nativa para celular** (App Store / Google Play): complejidad grande (firma de código, Xcode/Mac, tiendas de apps) que no aporta al objetivo del modelo. Se deja como fase futura opcional (posible camino: PWA instalable desde el navegador, sin pasar por tiendas de apps).

## Próximos pasos
1. Armar el dataset con las features de arriba, reutilizando el código del F1 Stats Explorer contra Jolpica-F1.
2. Entrenar el primer modelo (versión simple).
3. Generar la predicción para Monza antes del 6/9 y guardarla.
4. Comparar contra el resultado real y ajustar.
5. Construir la interfaz en Next.js (proyecto separado del F1 Stats Explorer), leyendo las predicciones desde el JSON generado por el pipeline.
6. Desplegar en Vercel para tener una URL fija, accesible desde cualquier dispositivo sin instalar nada.
