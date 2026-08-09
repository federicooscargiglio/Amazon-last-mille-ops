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
- [~] Dashboard ejecutivo en Power BI — en progreso (ver detalle abajo)

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
- **Zona:** `zona_riesgo_low`, la tasa de `Low` por zona con **suavizado bayesiano** — calculada **exclusivamente con el conjunto de entrenamiento** para evitar fuga de datos (ver split abajo), y corregida tras una pasada de QA (ver sección "Validación" abajo) para no confiar en zonas con muy pocas rutas de historia. Las zonas no vistas en entrenamiento reciben la tasa global de `Low` como valor de referencia.

### Split train/test

**Decisión:** el split se hace sobre rutas completas (nunca paradas sueltas de la misma ruta repartidas entre ambos conjuntos) y estratificado por `route_score`, dado que la clase `Low` es solo 102 de 6,112 rutas (1.7%) — un split al azar puede dejarla mal representada en train o en test por pura casualidad estadística.

Verificado: 0 rutas se repiten entre train y test, y la proporción de `Low`/`Medium`/`High` es casi idéntica en ambos conjuntos (~1.7% de `Low` en los dos). Resultado: `data/processed/train.parquet` (719,870 paradas) y `data/processed/test.parquet` (178,545 paradas).

### Modelos: baseline vs. Random Forest (F1-05 / F1-06)

**Baseline (regresión logística):** `Low` — precision 0.02 / recall 0.18 / f1 0.04. `Medium` — 0.68 / 0.57 / 0.62. `High` — 0.55 / 0.50 / 0.52. Accuracy 0.53. ROC-AUC (ovr, macro) 0.6102.

**Random Forest** (`class_weight="balanced_subsample"`, `max_depth=12`): `Low` — precision 0.02 / recall 0.18 / f1 0.03. `Medium` — 0.66 / 0.58 / 0.62. `High` — 0.57 / 0.45 / 0.50. Accuracy 0.52. ROC-AUC (ovr, macro) 0.6029.

**Conclusión:** cambiar a un modelo no lineal no mejora la detección de rutas `Low` — el recall es prácticamente idéntico entre los dos modelos (18%), y el ROC-AUC de la regresión logística es levemente **mejor** que el de Random Forest, no peor. Se probó además un Random Forest sin límite de profundidad para descartar que la limitación fuera de complejidad: ese modelo memoriza train (100% en las tres clases) pero el recall de `Low` en test cae muy por debajo del modelo con profundidad limitada — evidencia de overfitting, no de aprendizaje real. `zona_riesgo_low` concentra ~61% de la importancia de features del Random Forest. Ningún modelo es confiable todavía para uso operativo.

**Un modelo con una sola variable (`zona_riesgo_low`, nada más) obtiene ~17% de recall en `Low`** — prácticamente lo mismo que el modelo completo con las 14 features. Esto muestra que las demás variables de F1-04 (densidad de paquetes, distancia, ventana horaria, estación) no están aportando señal adicional para esta clase; el cuello de botella no es el algoritmo ni la falta de features en sí, es la falta de señal *nueva* — y posiblemente el tamaño de muestra (solo 102 rutas `Low` en todo el dataset).

### Validación (QA) — hallazgos y una corrección propia

Antes de dar por cerrado F1-05/F1-06 se corrió una pasada de validación (`data:validate-data`) sobre la metodología y los cálculos. Hallazgos:

- **Significancia estadística, no solo intuición:** la diferencia de recall en `Low` entre el baseline original (sin suavizar, 16%) y el Random Forest (18%) se probó con un test de McNemar pareado sobre las mismas rutas `Low` de test — resultado estadísticamente significativo (p ≈ 0.00045), no ruido como se asumió en un primer momento. Aun así, la diferencia no es relevante para el negocio: 16% o 18%, en ambos casos el modelo deja pasar más de 8 de cada 10 rutas `Low` reales. **Significativo estadísticamente ≠ significativo para el negocio** — con suficientes datos, hasta diferencias mínimas pueden ser "reales" sin ser útiles.
- **Target encoding sin suavizar (corregido):** `zona_riesgo_low` original usaba la tasa cruda de `Low` por zona. El 34% de las zonas de train tenían menos de 5 rutas de historia, y varias con 1 sola ruta `Low` quedaban con tasa = 100% — ruido con forma de señal fuerte. Se corrigió con suavizado bayesiano (`zona_riesgo_low = (n_low + k×tasa_global) / (n_rutas + k)`, `k=10`).
- **Un bug propio, encontrado al re-verificar:** la primera implementación del suavizado deduplicaba por `route_id` antes de agrupar por zona, sin considerar que una ruta visita muchas zonas distintas a lo largo de sus paradas — eso descartaba casi todas las zonas de cada ruta menos una. El síntoma (zonas distintas en train cayendo de 8.720 a 3.138) hizo que un resultado inicialmente "espectacular" (recall de `Low` subiendo de 16% a 41%) resultara ser un artefacto del bug, no una mejora real. Corregido deduplicando por el par `(zone_id, route_id)`. Queda documentado en `01_eda_inicial.ipynb` (sección 14) como recordatorio: un resultado sorprendentemente bueno amerita más sospecha, no menos.

### Dashboard ejecutivo en Power BI — en progreso

Archivo: `dashboard/dashboard_riesgo_sla.pbix`. Construido de forma interactiva (Claude maneja el mouse/teclado, Fede decide cada punto de diseño y responde examen antes de cerrar cada bloque — ver "Modo de trabajo en herramientas visuales" en el CLAUDE.md del proyecto).

**Modelo tabular (cerrado):** esquema de estrella con `Hechos_Paradas` (grano = parada, ~898K filas) y dos dimensiones, `Dim_Ruta` (route_id, station_code, date, route_score, executor_capacity_cm3 — atributos constantes por ruta) y `Dim_Zona` (zone_id, zona_riesgo_low). Relaciones varios-a-uno verificadas. Un hallazgo de calidad de datos en el camino: 6.515 paradas (0.73%) sin `zone_id` — se optó por filtrarlas de `Dim_Zona` en vez de crear una categoría "sin zona", dado el volumen mínimo.

**Medidas DAX (cerradas):** `SLA Compliance Rate` (98.29%), `Riesgo Promedio` (1.68%), `Total Rutas` (6.112), `Total Paradas` (898.415), `% Incumplimiento SLA` (`= 1 - [SLA Compliance Rate]`, 1.79% agregado en la matriz de estación x franja — deriva del mismo cálculo ya validado en vez de reimplementar la lógica de cumplimiento por segunda vez).

**Estructura de 4 páginas acordada:** Resumen Ejecutivo / Riesgo por Zona / Riesgo por Estación y Franja / Estado del Modelo (esta última traduce las métricas técnicas del modelo a lenguaje de negocio, no repite números crudos de recall/ROC-AUC).

**Avance real por página:**
- Resumen Ejecutivo: **completa** — 4 tarjetas KPI + gráfico de distribución de rutas por `route_score` + gráfico de anillos (SLA Compliance Rate vs % Incumplimiento SLA) agregado en la sesión de pulido visual.
- Riesgo por Zona: matriz por `zone_id` con `Promedio de zona_riesgo_low` (ponderado por parada vía AVERAGEX/RELATED, no por zona) y `Total Paradas` — **completa**. Pendiente de una sesión anterior: gráfico "Top 10 zonas de mayor riesgo" — se intentó y se descartó (ver nota de sesión 08-ago), queda como tarea abierta.
- Riesgo por Estación y Franja: **completa** — matriz cruzada `station_code` x `franja_horaria` con `% Incumplimiento SLA` y `Total Paradas`. Volumen incluido a propósito: varias celdas de baja cantidad de paradas muestran 100% de incumplimiento — ruido estadístico de bajo volumen, no una alerta real, visible solo gracias a la columna de volumen al lado. Pendiente: aplicar formato condicional (mapa de calor) sobre la columna de incumplimiento — decidido con Fede, no llegó a construirse.
- Estado del Modelo: **sin construir todavía**. Mensaje de negocio ya acordado con Fede (deliberadamente poco alentador, no vende nada que no esté probado): el modelo detecta menos de 2 de cada 10 rutas `Low` reales (recall ~18%) y no es confiable para uso operativo; agregar más variables no mejora la detección porque casi toda la señal viene de una sola (`zona_riesgo_low`) — el cuello de botella es la falta de historial de rutas de riesgo (102 en total), no el algoritmo. Pendiente: construir la página con este mensaje sin mostrar métricas técnicas crudas.

**Nota de proceso (sesión 06-ago):** instancias duplicadas de Power BI Desktop bloqueando el archivo, un proceso zombie cerrado desde el Administrador de Tareas, e interrupción de plataforma.

**Nota de proceso (sesión 08-ago):** se repitió el mismo problema de instancias duplicadas de Power BI Desktop — en un momento hubo hasta 4 ventanas del mismo archivo abiertas simultáneamente por el robo de foco intermitente de Chrome, lo que causó la pérdida de una versión de la matriz de estación x franja (tuvo que rehacerse, no afectó el resultado final pero sí el tiempo). Fede tuvo que intervenir manualmente cerrando ventanas de más. Sigue siendo un problema no resuelto de la mecánica de trabajo interactivo, a tener en cuenta para la próxima sesión de Power BI: guardar con más frecuencia y verificar el número de ventanas abiertas antes de cada bloque de edición.

**Nota de proceso (sesión de pulido visual, 08/09-ago):** a pedido de Fede se agregaron gráficos a las 3 páginas ya completas antes de seguir con "Estado del Modelo" (decisión consciente, documentada como reordenamiento de prioridad frente al roadmap del Kanban). Se completó el gráfico de anillos de Resumen Ejecutivo. El gráfico "Top 10 zonas de riesgo" de Riesgo por Zona se intentó pero quedó roto: combinar un filtro "Top N" (por `Riesgo Promedio`) con un filtro numérico adicional (`Total Paradas >= 100`) en el mismo objeto visual dejó el gráfico sin datos — probablemente el Top N se resuelve sobre el conjunto ya filtrado y ninguna de las 10 zonas de mayor riesgo cruda supera el umbral de volumen, o hay un conflicto de evaluación de filtros de Power BI no resuelto en el tiempo disponible. Se eliminó el objeto visual roto en vez de dejarlo guardado así, y queda como tarea pendiente (posible solución a probar: usar una medida DAX explícita que calcule el ranking ya excluyendo zonas de bajo volumen, en vez de combinar dos filtros de objeto visual). El mapa de calor de Riesgo por Estación y Franja no llegó a intentarse por el tiempo insumido en el problema anterior.

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
