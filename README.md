# Optimización de Rutas de Última Milla

Proyecto de portfolio — Fase 1 de **Insight** (consultora de datos + IA aplicada a e-commerce, logística y fulfillment operations).

## Objetivo

Analizar el dataset [Amazon Last Mile Routing Research Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/) para identificar patrones de riesgo de incumplimiento de SLA en la última milla, y construir un modelo predictivo simple de ese riesgo junto con un dashboard ejecutivo en Power BI.

Este proyecto conecta directamente con experiencia operativa real: 2.5 años como Courier y luego Dispatcher en Amazon Logistics (Berlín), gestionando KPIs de cumplimiento (SLA, delivery rate, incidentes) en 50+ rutas diarias.

## Estado actual

🚧 En progreso — Fase 1 del roadmap de Insight (deadline: 04-sep-2026).

- [ ] EDA inicial (3 gráficos exploratorios + conclusión de negocio)
- [ ] Modelo predictivo de riesgo de incumplimiento de SLA
- [ ] Dashboard ejecutivo en Power BI

La conclusión de negocio de cada entregable se agrega a este README a medida que se completa — todavía no hay resultados que reportar.

## Estructura del proyecto

```
codigo/
  notebooks/     → EDA, exploración, prototipado
  src/           → funciones reutilizables (limpieza, feature engineering) si el proyecto lo justifica
data/
  raw/           → dataset original, tal cual se descargó — nunca se edita a mano
  processed/     → datos limpios/transformados, generados por el código de codigo/
outputs/
  figures/       → gráficos exportados desde los notebooks
  models/        → modelo entrenado (una vez que exista)
```

`data/` y `outputs/` no se versionan en Git (ver `.gitignore`) porque son regenerables: cualquiera que clone este repo y corra los notebooks sobre el dataset público debería poder reproducir los mismos resultados.

## Cómo correrlo

1. Clonar el repositorio.
2. Descargar el dataset del [Amazon Last Mile Routing Research Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/) y colocarlo en `data/raw/`.
3. Instalar dependencias: `pip install -r requirements.txt`.
4. Correr los notebooks en `codigo/notebooks/` en orden.

## Autor

Federico Giglio — [LinkedIn](https://linkedin.com/in/federico-oscar-giglio) · [GitHub](https://github.com/federicooscargiglio)
