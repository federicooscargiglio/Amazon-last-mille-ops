# Optimización de Rutas de Última Milla

Proyecto de portfolio — Fase 1 de **Insight** (consultora de datos + IA aplicada a e-commerce, logística y fulfillment operations).

## Objetivo

Analizar el dataset [Amazon Last Mile Routing Research Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/) para identificar patrones de riesgo de incumplimiento de SLA en la última milla, y construir un modelo predictivo simple de ese riesgo junto con un dashboard ejecutivo en Power BI.

Este proyecto conecta directamente con experiencia operativa real: 2.5 años como Courier y luego Dispatcher en Amazon Logistics (Berlín), gestionando KPIs de cumplimiento (SLA, delivery rate, incidentes) en 50+ rutas diarias.

## Estado actual

🚧 En progreso — Fase 1 del roadmap de Insight (deadline: 04-sep-2026).

- [x] EDA inicial (3 gráficos exploratorios + conclusión de negocio) — `codigo/notebooks/01_eda_inicial.ipynb`
- [x] Definición de la variable objetivo de riesgo de SLA (ver sección abajo)
- [x] Feature engineering: ventana horaria, densidad de paquetes, distancia a la siguiente parada y zona (ver sección abajo)
- [x] Split train/test (por ruta completa, estratificado por `route_score`)
- [x] Modelo baseline: regresión logística (`codigo/notebooks/02_modelo_baseline.ipynb`)
- [x] Iteración de modelo: Random Forest (`codigo/notebooks/03_modelo_random_forest.ipynb`)
- [ ] Dashboard ejecutivo en Power BI

### Conclusión de negocio — EDA inicial

El 93.4% de las paradas de este dataset no tiene una ventana horaria de entrega prometida al cliente registrada, por lo que el riesgo de incumplimiento de SLA no puede definirse como "incumplimiento de ventana horaria" para la mayoría de los casos — la variable objetivo del modelo se apoya en su lugar en `route_score`, el indicador de calidad que Amazon calcula para el 100% de las rutas (detalle abajo).

### Definición de la variable objetivo (riesgo de SLA)

**Decisión:** el target del modelo es `route_score` (`Low` / `Medium` / `High`), calculado por Amazon a nivel ruta y propagado a cada parada de esa ruta como etiqueta débil (*weak label*). Se descartó usar cumplimiento de ventana horaria como target porque no está disponible para la mayoría de los datos (ver EDA).

**Por qué a nivel parada y no a nivel ruta:** con ~898K paradas en vez de ~6K rutas, el modelo cuenta con muchísimo más volumen para aprender patrones reales, y el resultado es accionable a nivel operativo — permite identificar qué paradas puntuales son de riesgo dentro de una ruta, en lugar de solo poder marcar la ruta completa como problemática sin poder decir por qué.

**Limitación conocida, aceptada explícitamente:** al propagar el score de una ruta a cada una de sus paradas, no todas contribuyeron por igual a ese resultado — es una etiqueta con ruido. Se acepta este trade-off porque el ruido no está correlacionado con las variables predictoras y se diluye con el volumen de datos: la diferencia real entre grupos (por ejemplo, zonas con más incidencia de rutas `Low`) sigue siendo detectable aunque una porción de las etiquetas individuales sea imprecisa.

**Pendiente para la siguiente etapa (feature engineering / modelo baseline):** la clase `Low` representa solo 1.7% de las paradas — un desbalance de clases severo que va a condicionar las métricas de evaluación (accuracy no sirve con este desbalance) y posiblemente requiera balanceo o ponderación de clases.

### Feature engineering

A partir de las columnas ya disponibles se construyeron cinco variables nuevas:

- **Ventana horaria:** `window_duration_min` (duración de la ventana prometida al cliente, en minutos) y `franja_horaria` (madrugada/mañana/tarde/noche).
- **Densidad de paquetes:** `volumen_promedio_paquete_cm3`, `paradas_por_ruta` y `paquetes_por_ruta`.
- **Distancia:** `distancia_a_siguiente_km`, calculada con la fórmula de Haversine sobre el orden real de visita de cada ruta (`actual_sequences.json`), no sobre el orden en que las filas aparecen en la tabla.
- **Zona:** `zona_riesgo_low`, la proporción histórica de paradas `Low` en cada zona — calculada **exclusivamente con el conjunto de entrenamiento** para evitar fuga de datos (ver split abajo). Las zonas no vistas en entrenamiento reciben la tasa global de `Low` como valor de referencia.

### Split train/test

**Decisión:** el split se hace sobre rutas completas (nunca paradas sueltas de la misma ruta repartidas entre ambos conjuntos) y estratificado por `route_score`, dado que la clase `Low` es solo 102 de 6,112 rutas (1.7%) — un split al azar puede dejarla mal representada en train o en test por pura casualidad estadística.

Verificado: 0 rutas se repiten entre train y test, y la proporción de `Low`/`Medium`/`High` es casi idéntica en ambos conjuntos (~1.7% de `Low` en los dos). Resultado: `data/processed/train.parquet` (719,870 paradas) y `data/processed/test.parquet` (178,545 paradas).

### Modelos: baseline vs. Random Forest (F1-05 / F1-06)

**Baseline (regresión logística):** `Low` — precision 0.02 / recall 0.16 / f1 0.04. `Medium` — 0.68 / 0.59 / 0.63. `High` — 0.56 / 0.51 / 0.53. Accuracy 0.55.

**Random Forest** (`class_weight="balanced_subsample"`, `max_depth=12`): `Low` — precision 0.02 / recall 0.18 / f1 0.04. `Medium` — 0.66 / 0.59 / 0.62. `High` — 0.56 / 0.47 / 0.51. Accuracy 0.53. ROC-AUC (ovr, macro) 0.6067.

**Conclusión:** cambiar a un modelo no lineal no mejora de forma real la detección de rutas `Low` (16% → 18% de recall, dentro del margen de ruido). Se probó además un Random Forest sin límite de profundidad para descartar que la limitación fuera de complejidad: ese modelo memoriza train (100% en las tres clases) pero cae a 0% de recall en `Low` en test — evidencia de overfitting, no de aprendizaje real. `zona_riesgo_low` concentra el 62% de la importancia de features del modelo, señal de que no hay interacciones adicionales fuertes que el modelo esté aprovechando. Ningún modelo de los dos es confiable todavía para uso operativo; el cuello de botella parece ser de features disponibles, no de algoritmo.

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
