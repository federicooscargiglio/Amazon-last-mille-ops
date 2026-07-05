# Optimización de Rutas de Última Milla

Proyecto de portfolio — Fase 1 de **Insight** (consultora de datos + IA aplicada a e-commerce, logística y fulfillment operations).

## Objetivo

Analizar el dataset [Amazon Last Mile Routing Research Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/) para identificar patrones de riesgo de incumplimiento de SLA en la última milla, y construir un modelo predictivo simple de ese riesgo junto con un dashboard ejecutivo en Power BI.

Este proyecto conecta directamente con experiencia operativa real: 2.5 años como Courier y luego Dispatcher en Amazon Logistics (Berlín), gestionando KPIs de cumplimiento (SLA, delivery rate, incidentes) en 50+ rutas diarias.

## Estado actual

🚧 En progreso — Fase 1 del roadmap de Insight (deadline: 04-sep-2026).

- [x] EDA inicial (3 gráficos exploratorios + conclusión de negocio) — `codigo/notebooks/01_eda_inicial.ipynb`
- [x] Definición de la variable objetivo de riesgo de SLA (ver sección abajo)
- [ ] Modelo predictivo de riesgo de incumplimiento de SLA
- [ ] Dashboard ejecutivo en Power BI

### Conclusión de negocio — EDA inicial

El 93.4% de las paradas de este dataset no tiene una ventana horaria de entrega prometida al cliente registrada, por lo que el riesgo de incumplimiento de SLA no puede definirse como "incumplimiento de ventana horaria" para la mayoría de los casos — la variable objetivo del modelo se apoya en su lugar en `route_score`, el indicador de calidad que Amazon calcula para el 100% de las rutas (detalle abajo).

### Definición de la variable objetivo (riesgo de SLA)

**Decisión:** el target del modelo es `route_score` (`Low` / `Medium` / `High`), calculado por Amazon a nivel ruta y propagado a cada parada de esa ruta como etiqueta débil (*weak label*). Se descartó usar cumplimiento de ventana horaria como target porque no está disponible para la mayoría de los datos (ver EDA).

**Por qué a nivel parada y no a nivel ruta:** con ~898K paradas en vez de ~6K rutas, el modelo cuenta con muchísimo más volumen para aprender patrones reales, y el resultado es accionable a nivel operativo — permite identificar qué paradas puntuales son de riesgo dentro de una ruta, en lugar de solo poder marcar la ruta completa como problemática sin poder decir por qué.

**Limitación conocida, aceptada explícitamente:** al propagar el score de una ruta a cada una de sus paradas, no todas contribuyeron por igual a ese resultado — es una etiqueta con ruido. Se acepta este trade-off porque el ruido no está correlacionado con las variables predictoras y se diluye con el volumen de datos: la diferencia real entre grupos (por ejemplo, zonas con más incidencia de rutas `Low`) sigue siendo detectable aunque una porción de las etiquetas individuales sea imprecisa.

**Pendiente para la siguiente etapa (feature engineering / modelo baseline):** la clase `Low` representa solo 1.7% de las paradas — un desbalance de clases severo que va a condicionar las métricas de evaluación (accuracy no sirve con este desbalance) y posiblemente requiera balanceo o ponderación de clases.

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
