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
- Riesgo por Zona: matriz por `zone_id` con `Promedio de Riesgo` (ponderado por parada vía AVERAGEX/RELATED, no por zona) y `Total Paradas`, más gráfico "Top 10 Zonas de Mayor Riesgo" — **completa y pulida** (ver nota de sesión 21-ago-2026 abajo; el primer intento del gráfico se había roto y descartado en la sesión 08-ago, se reconstruyó bien en una sesión posterior).
- Riesgo por Estación y Franja: **completa y pulida** — matriz cruzada `station_code` x `franja_horaria` con `% Incumplimiento SLA` y `Total Paradas`, con mapa de calor (formato condicional rojo/verde) sobre la columna de incumplimiento. Volumen incluido a propósito: varias celdas de baja cantidad de paradas muestran 100% de incumplimiento — ruido estadístico de bajo volumen, no una alerta real, visible solo gracias a la columna de volumen al lado.
- Estado del Modelo: **completa**. Indicador tipo semáforo (círculo rojo, "NO APTO AÚN") + tres párrafos de texto con el mensaje de negocio ya acordado (deliberadamente poco alentador, no vende nada que no esté probado): el modelo detecta menos de 2 de cada 10 rutas `Low` reales (recall ~18%) y no es confiable para uso operativo; agregar más variables no mejora la detección porque casi toda la señal viene de una sola (`zona_riesgo_low`) — el cuello de botella es la falta de historial de rutas de riesgo (102 en total), no el algoritmo. Sin métricas técnicas crudas visibles, como se acordó.

**Con esto, las 4 páginas del dashboard tienen contenido real y pulido visual 100% cerrados** (ver nota de sesión 22-ago-2026 abajo para el cierre de la última página pendiente).

**Nota de proceso (sesión 06-ago):** instancias duplicadas de Power BI Desktop bloqueando el archivo, un proceso zombie cerrado desde el Administrador de Tareas, e interrupción de plataforma.

**Nota de proceso (sesión 08-ago):** se repitió el mismo problema de instancias duplicadas de Power BI Desktop — en un momento hubo hasta 4 ventanas del mismo archivo abiertas simultáneamente por el robo de foco intermitente de Chrome, lo que causó la pérdida de una versión de la matriz de estación x franja (tuvo que rehacerse, no afectó el resultado final pero sí el tiempo). Fede tuvo que intervenir manualmente cerrando ventanas de más. Sigue siendo un problema no resuelto de la mecánica de trabajo interactivo, a tener en cuenta para la próxima sesión de Power BI: guardar con más frecuencia y verificar el número de ventanas abiertas antes de cada bloque de edición.

**Nota de proceso (sesión de pulido visual, 08/09-ago):** a pedido de Fede se agregaron gráficos a las 3 páginas ya completas antes de seguir con "Estado del Modelo" (decisión consciente, documentada como reordenamiento de prioridad frente al roadmap del Kanban). Se completó el gráfico de anillos de Resumen Ejecutivo. El gráfico "Top 10 zonas de riesgo" de Riesgo por Zona se intentó pero quedó roto: combinar un filtro "Top N" (por `Riesgo Promedio`) con un filtro numérico adicional (`Total Paradas >= 100`) en el mismo objeto visual dejó el gráfico sin datos — probablemente el Top N se resuelve sobre el conjunto ya filtrado y ninguna de las 10 zonas de mayor riesgo cruda supera el umbral de volumen, o hay un conflicto de evaluación de filtros de Power BI no resuelto en el tiempo disponible. Se eliminó el objeto visual roto en vez de dejarlo guardado así, y queda como tarea pendiente (posible solución a probar: usar una medida DAX explícita que calcule el ranking ya excluyendo zonas de bajo volumen, en vez de combinar dos filtros de objeto visual). El mapa de calor de Riesgo por Estación y Franja no llegó a intentarse por el tiempo insumido en el problema anterior. En una sesión posterior (09/12-ago) se completó "Estado del Modelo" (semáforo + texto).

**Nota de proceso (sesión de retoque visual, 20-ago-2026):** se aplicó un tema custom (`dashboard/tema_logistica.json`, paleta navy/teal/amber/rojo/verde, tipografía Segoe UI) a las 4 páginas de una sola vez. Sobre ese tema, se pulieron 2 de las 4 páginas: "Estado del Modelo" (el indicador circular plano se convirtió en un badge/pill con texto encima, y el bloque de texto explicativo se agrandó/reposicionó para reducir espacio vacío) y "Resumen Ejecutivo" (título de página corregido — se cortaba arriba y después mostraba un scroll interno por tamaño de fuente insuficiente para la caja; se descubrió que la página real mide 1280x720 y no 1920x1080 como se había asumido en un momento, corrección documentada en el chat; se cerró el hueco entre las tarjetas KPI y los gráficos y se agrandaron los gráficos con el espacio ganado). Quedan pendientes de este mismo pulido visual: "Riesgo por Zona" y "Riesgo por Estación y Franja".

**Nota de proceso (sesión 21-ago-2026):** se terminó el pulido visual de "Riesgo por Zona" (tabla + gráfico alineados en espejo, 620x580, mismo patrón que las otras páginas). Se corrigieron 3 problemas puntuales de esta página: (1) columna técnica `zona_riesgo_low` renombrada a un nombre de negocio legible y puesto en formato porcentaje; (2) título de página corregido a "Riesgo por Zona" (Title Case, consistente con el resto); (3) fila en blanco en la tabla (6.515 paradas sin zona asignada) investigada y filtrada — no era error de carga: son paradas con coordenadas válidas pero fuera de la cobertura geográfica de `Dim_Zona` (aparecen lat/lng de otras regiones de EE.UU., no solo la zona principal cubierta). Queda pendiente para más adelante, no para Fase 1: si esas paradas corresponden a rutas reales de trabajo, extender la grilla de zonas para cubrirlas. También hubo un cuelgue real de Power BI Desktop (no problema de automatización) que requirió reinicio manual de la app — sin pérdida de trabajo relevante, el guardado previo cubría lo hecho.

**Nota de proceso (sesión 22-ago-2026):** se cerró el pulido visual de "Riesgo por Estación y Franja", la última página pendiente — con esto las 4 páginas del dashboard quedan 100% cerradas (contenido + pulido). Se hizo: (1) matriz reposicionada a pantalla completa (1200x620, mismo patrón que las otras 3 páginas — antes ocupaba solo una fracción chica del canvas); (2) tamaño de letra subido de 10pt a 14pt en valores, encabezados de columna y encabezados de fila (el texto quedaba chico dentro de una caja grande, ahora usa el espacio); (3) título de página agregado ("Riesgo por Estación y Franja", 18pt, mismo estilo que las otras 3 páginas) — sin repetir el bug de renderizado de la sesión anterior. Hallazgo aparte: el mapa de calor sobre `% Incumplimiento SLA`, que el README y el Kanban tenían marcado como "decidido, no construido", **ya estaba aplicado** al abrir la página esta sesión — se construyó en algún momento entre sesiones sin quedar documentado. Corregido en la sección de avance por página arriba.

## Handoff para la próxima conversación (actualizado 22-ago-2026)

Si esto se retoma en una conversación nueva de Claude, este bloque más el `CLAUDE.md` del proyecto deberían alcanzar para arrancar sin fricción — no hace falta releer todo el historial de chat.

**Estado real ahora mismo:**
- Las 4 páginas del dashboard (`dashboard/dashboard_riesgo_sla.pbix`) están 100% cerradas: contenido y pulido visual completos en Resumen Ejecutivo, Riesgo por Zona, Riesgo por Estación y Franja, Estado del Modelo (ver notas de sesión 20-ago, 21-ago y 22-ago arriba). No queda pulido de dashboard pendiente.
- **Tarjeta "Pulir dashboard" del Kanban: cerrada (22-ago-2026).** Era la tarjeta prioritaria de la semana definida en el check-in del 21-ago; se completó en la sesión del 22-ago (ver nota de proceso arriba). Próxima prioridad a definir con Fede: ítems de Backlog post-dashboard — "Examen pendiente" (modelo tabular / medidas DAX, ver bullet abajo), "README + narrativa personal" (~1.5h), "Publicar en GitHub + post LinkedIn" (~1h), o el "Checkpoint Fase 1" del 04-sep.
- **Dos puntos detectados por Fede mismo (revisión directa de `git log` + timestamps, 20-ago-2026) — resueltos en la sesión del 21-ago-2026:**
  1. **Gap de 13 días sin commits, 23-jul a 05-ago — explicado.** Según Fede (check-in del 21-ago): sí hubo trabajo en esos días, pero fue diseño/boceto del dashboard Power BI (qué páginas, qué visuales, cómo estructurar Resumen Ejecutivo vs. detalle por zona) — trabajo de planificación que no generó commits porque no había código ni archivo `.pbix` todavía para versionar. El primer commit del dashboard (`e91ad5c`, 05-ago) es donde ese diseño se empieza a construir en la herramienta. Lección para el Kanban: una tarjeta de "diseño" sin artefacto versionable todavía debería dejar rastro igual (nota en el README o el Kanban del día), no solo el commit del resultado final 13 días después.
  2. **El modelo entrenado no estaba exportado — ahora sí.** Se reentrenó el baseline (regresión logística, mismo pipeline exacto de `02_modelo_baseline.ipynb`) sobre `train.parquet` y se exportó como `outputs/models/modelo_baseline_riesgo_sla.joblib` (bundle con modelo + scaler + columnas + fallback de imputación) más `outputs/models/README_modelo.md` con instrucciones de uso. Nota: `outputs/models/` sigue sin versionarse en Git por diseño (ver `.gitignore` y sección "Estructura del proyecto") — el artefacto es regenerable corriendo el script/notebook, se entregó a Fede directamente en la sesión, no vive en el repo.
- **Examen pendiente:** las tarjetas de modelo tabular y medidas DAX del Kanban no deberían moverse a "Hecho" sin que Fede pueda explicar sin ayuda las relaciones y medidas creadas (regla del CLAUDE.md, sección "Modo de trabajo en herramientas visuales"). Se hicieron chequeos informales durante el trabajo (AVERAGEX/RELATED, CALCULATE cross-table, "varios a uno") con respuestas correctas de Fede, pero no un examen formal de cierre — vale la pena confirmarlo antes de mover esas tarjetas si todavía no se movieron.
- **Check-in semanal realizado (21-ago-2026):** (1) completado y verificado — tema visual global, "Estado del Modelo", "Resumen Ejecutivo" y "Riesgo por Zona" pulidos; (2) bloqueado/pendiente — solo queda "Riesgo por Estación y Franja" (pulido + mapa de calor), nada trabado por un problema técnico sin resolver; (3) horas: **más de 7hs**, por encima del compromiso de 5-7hs/semana, por problemas de conexión a internet — marcado como fricción real a resolver si se repite, no como anécdota.
- Próximo corte duro: **Checkpoint Fase 1, 04-sep-2026** (tarjeta ya en el Backlog del Kanban).

**Problema técnico recurrente a tener en cuenta:** Power BI Desktop tiende a abrir instancias duplicadas del mismo archivo cuando se usa `open_application` repetidamente, y hubo varios episodios de robo de foco (Chrome, apps ajenas como juegos) que interrumpieron el control de pantalla. Guardar (Ctrl+S) con mucha frecuencia — al menos después de cada cambio de campo o filtro, no solo al final de un bloque — y verificar cuántas ventanas de Power BI hay abiertas antes de asumir cuál es la activa.

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
