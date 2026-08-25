# F1 Podium Predictor

Modelo predictivo de podio en F1 (top 3 sí/no), extensión del [F1 Stats Explorer](https://github.com/gonzalovalentingomez/f1-stats-explorer). Pipeline en Python (pandas + scikit-learn) sobre datos históricos públicos de Jolpica-F1 y OpenF1, con interfaz en Next.js desplegada en Vercel.

Ver [`f1-v2-proyecto-brief.md`](./f1-v2-proyecto-brief.md) para el detalle completo del enfoque, features y roadmap.

## Estructura

- `pipeline/` — dataset, features, entrenamiento y generación de predicciones (Python).
- `web/` — interfaz Next.js que consume el JSON de predicciones (se suma más adelante).

## Milestone inicial

Predicción del GP de Italia (Monza), 4-6 de septiembre de 2026.
