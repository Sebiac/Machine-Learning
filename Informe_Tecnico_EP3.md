# Informe técnico EP3 – Spotify Tracks (Caso C)

**Asignatura:** MLY1101 – Machine Learning · **Grupo 5:** Diego Marambio, Sebastián Álvarez, Álvaro Sepúlveda y Erick Rosales **Notebook que reproduce este informe:** `notebooks/Informe_Tecnico_EP3.ipynb` (las secciones de EDA, sesgos y K-Means están en `notebooks/Informe_Tecnico_EP2_Referencia.ipynb`).

## 1\. Descripción del problema de negocio

Los equipos de **A\&R** de un sello deben decidir en qué canciones invertir promoción entre miles de lanzamientos. La pregunta es: **¿pueden las características de una canción (audio, género y metadatos) ayudar a priorizar cuáles promover?** La variable objetivo es `popularity` (0–100), un índice **propietario de Spotify** que depende de la cantidad y recencia de reproducciones; no mide calidad artística ni demuestra causalidad.

## 2\. Objetivos del proyecto

- Cuantificar cuánto de la popularidad explican audio y género, comparando modelos supervisados (regresión y clasificación) frente a baselines.  
- Segmentar el catálogo con una técnica no supervisada (K-Means, EP2).  
- **EP3:** optimizar hiperparámetros (IE9), aplicar y evaluar ensambles (IE10), medir la generalización con validación cruzada (IE11) y elegir y justificar la solución óptima (IE12).  
- Traducir los resultados en recomendaciones honestas para A\&R.

## 3\. Definición de KPIs

Un **hit** es una canción con `popularity ≥ 70`. Todos los KPIs se miden sobre canciones **no vistas** en el entrenamiento.

| KPI | Meta | Resultado EP3 | ¿Se cumple? |
| :---- | :---- | :---- | :---- |
| MAE (regresor final) | ≤ 12 puntos | **10,12** | Sí |
| Mejora de MAE vs. media global | ≥ 25% | 40,8% | Sí |
| Mejora de MAE vs. **baseline por género** (nuevo) | \> 0%, con IC 95% por encima de 0 | **16,0%** (IC 1,76–2,09 pts) | Sí |
| R² (regresor final) | ≥ 0,30 | 0,366 | Sí |
| Precisión@5% (regresor final) | ≥ 15% | 12,8% | **No** |
| Precisión@5% (clasificador final) | ≥ 15% | **30,0%** (IC 27,1–33,1) | Sí |
| Lift vs. azar (clasificador final) | ≥ 3× | 8,8× | Sí |

La Precisión@5% es la proporción de hits entre el 5% de canciones mejor puntuadas (892 canciones en la prueba). El azar da 3,4%.

## 4\. Fuentes de datos

- **Archivo:** `data/Spotify_Tracks_Dataset.csv` (114.000 filas × 21 columnas): atributos de audio equivalentes a los de la API de Spotify, metadatos y género.  
- **Origen:** *Spotify Tracks Dataset* de `maharshipandya`: [https\://www\.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (espejo: [https\://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)). **Publicado el 14-06-2023**, por lo que la popularidad es una foto de esa fecha o anterior.  
- **Repositorio del proyecto:** [https\://github.com/Sebiac/Machine-Learning](https://github.com/Sebiac/Machine-Learning).

## 5\. Preparación y análisis exploratorio (resumen; detalle en el notebook de EP2)

- **Calidad:** el dataset es limpio en lo básico (1 fila con nulos, 450 duplicados exactos, 157 canciones con `tempo = 0`, \~1% de compases imposibles). Lo importante: **23.809 `track_id` repetidos** (la misma canción con varios géneros) y **14% de las filas con popularidad 0** (10,4% de las canciones únicas).  
- **Preparación:** se consolidó una fila por `track_id` (de 114.000 a **89.583 canciones únicas**), con el género en formato *multi-hot* (114 columnas; el 18,2% de las canciones tiene más de un género). Otro 9,5% son versiones de una canción con distinto `track_id`, por lo que la partición se hace **por grupos (artista \+ nombre)**: 71.727 canciones de entrenamiento, 17.856 de prueba y 0 grupos compartidos.  
- **Hallazgos del EDA:** solo el **3,5%** de las canciones son hits (por eso se evalúa con métricas de ranking y no con exactitud); ninguna variable de audio supera \|r\| \= 0,13 con la popularidad; los ceros se concentran en ciertos géneros (jazz 68%, *rock* 53%…), lo que sugiere un artefacto de la fuente; el género explica mucho más que el audio.

## 6\. Metodología (CRISP-DM)

| Fase | Qué se hizo | Dónde |
| :---- | :---- | :---- |
| Comprensión del negocio | Problema de A\&R, objetivos SMART y KPIs | Notebook EP2, secciones 1 y 2 |
| Comprensión de los datos | Calidad, duplicados, ceros, distribución | EP2, secciones 3 y 5 |
| Preparación | Una fila por canción, *multi-hot*, partición por grupos sin fuga, *pipeline* | EP2, secciones 4 y 7; EP3, sección 2 |
| Modelado | Ridge, Random Forest, HGB, K-Means (EP2); **búsqueda ampliada, ensambles y clasificador de hits (EP3)** | EP2, secciones 8 y 9; EP3, secciones 4 a 6 |
| Evaluación | Validación cruzada agrupada, prueba con intervalos de confianza, comparación con baselines | EP3, secciones 7 y 8 |
| Despliegue | Modelos guardados y función de uso | EP3, sección 9 |

## 7\. Desarrollo de los modelos

**Supervisados.** Regresión de `popularity` con Ridge, Random Forest e HistGradientBoosting (HGB) frente a tres referencias: la media global, el **baseline por género** (promedio de las medias de popularidad de los géneros de la canción) y un HGB **solo con género**. Para priorizar canciones se entrena además un **clasificador HGB** de hits. El preprocesamiento (escalado y *one-hot*) va dentro de un `Pipeline`, así que se ajusta solo con los datos de entrenamiento de cada pliegue.

**No supervisado (EP2).** K-Means con k \= 4, elegido por interpretabilidad (la silueta es plana entre k \= 3 y 6; máxima en k \= 2 con 0,244). Estabilidad entre semillas ARI \= 0,83 y silueta de 0,158 (bordes difusos). Segmentos: *acústico/ambiental* (21% de las canciones, 1,7% de hits), *intenso rock/metal* (34%, 2,7%), *hablado/comedia* (1%, 0,5%) y *bailable y positivo* (44%, 5,0%). Los segmentos explican menos del 1% de la varianza de la popularidad (η² \= 0,008): sirven para comparar lanzamientos y organizar el catálogo, no para predecir éxito.

## 8\. Optimización de hiperparámetros y ensambles

### 8.1 Optimización (IE9)

Todas las búsquedas usan el **100%** del entrenamiento, 3 pliegues agrupados y MAE (PR-AUC en el clasificador). La EP2 dejó **5 de 5 hiperparámetros en el borde de su grilla**; la EP3 hace una búsqueda aleatoria más amplia (incluida la función de pérdida) y un refinamiento posterior del HGB.

| Modelo | MAE CV EP2 | MAE CV EP3 | Mejora | Hiperparámetros elegidos |
| :---- | :---- | :---- | :---- | :---- |
| Ridge | 11,922 | 11,909 | −0,1% | `alpha = 0,01` (irrelevante) |
| Random Forest | 14,111 | 11,381 | −19,3% | sin límite de profundidad, `min_samples_leaf = 3`, `max_features = 0,5` |
| **HGB** | 11,373 | **10,347** | **−9,0%** | `loss = absolute_error`, `learning_rate = 0,12`, 61 hojas, `min_samples_leaf = 70`, `l2 = 0,07`, 200 iteraciones, `max_features = 0,8` |

El hiperparámetro dominante es la **función de pérdida**: las combinaciones con `absolute_error` rinden en promedio 10,54 de MAE, contra 11,65 con `squared_error`. Un refinamiento "de grueso a fino" no mejoró (10,374 contra 10,347), por lo que se mantiene la etapa 1\. El costo: `absolute_error` apunta a la mediana, y el R² baja de 0,414 a 0,366. Se descarta la hipótesis de la EP2 de que el Random Forest rendía mal por la dispersión de los géneros: estaba subajustado por sus límites.

### 8.2 Ensambles (IE10)

Se implementaron *bagging* (Random Forest), *boosting* (HGB), *voting* y *stacking* (con un meta-modelo que minimiza el MAE). Los errores de los tres modelos base están correlacionados en **0,91–0,94**, lo que limita lo que un ensamble puede aportar.

| Modelo (prueba) | MAE | IC 95% | R² | Diferencia de MAE contra el HGB |
| :---- | :---- | :---- | :---- | :---- |
| HGB | **10,118** | 9,85–10,42 | 0,366 | — |
| *Stacking* (meta MAE) | 10,179 | 9,92–10,47 | 0,403 | Peor en 0,06 (IC −0,12 a −0,004) |
| *Voting* | 10,692 | 10,44–10,96 | 0,413 | Peor en 0,57 (IC −0,67 a −0,48) |

**Ningún ensamble mejora el MAE** y el *stacking* cuesta 24 veces más en tiempo de ajuste (260 s contra 11 s). En RMSE tampoco hay mejora significativa frente al mejor modelo base en esa métrica.

## 9\. Evaluación de resultados

### 9.1 Generalización (IE11)

Validación cruzada agrupada de 5 pliegues sobre el entrenamiento completo, brecha entrenamiento–prueba, curva de aprendizaje y validación agrupada por artista.

| Modelo | MAE CV (± sd) | MAE entrenamiento | MAE prueba | Brecha prueba − entrenamiento |
| :---- | :---- | :---- | :---- | :---- |
| Baseline por género | 12,303 ± 0,042 | 12,279 | 12,049 | −0,23 |
| HGB solo género | 10,542 ± 0,139 | 10,414 | 10,268 | −0,15 |
| Ridge | 11,914 ± 0,066 | 11,866 | 11,712 | −0,15 |
| Random Forest | 11,219 ± 0,091 | 7,303 | 10,909 | **3,61** |
| **HGB** | **10,339 ± 0,101** | 9,413 | **10,118** | 0,71 |
| *Voting* | 10,929 ± 0,088 | 9,336 | 10,692 | 1,36 |
| *Stacking* (meta MAE) | 10,403 ± 0,129 | 8,240 | 10,179 | 1,94 |

- El HGB generaliza bien: la validación cruzada y la prueba coinciden y el MAE varía solo 0,27 entre pliegues. El Random Forest sobreajusta (brecha 3,6).  
- **Curva de aprendizaje:** el MAE de validación baja de 15,08 a 10,35 al pasar de 11.954 a 47.818 canciones y no se aplana, pero parte de esa mejora puede ser reconocer artistas.  
- **Artistas nuevos:** el 89,8% de las canciones de prueba tienen un artista que está también en el entrenamiento. Al agrupar la validación por artista, el MAE del HGB sube de 10,34 a **10,72** y su ventaja sobre el HGB solo con género cae de 1,9% a 1,2%. Las cifras son por tanto **algo optimistas** para artistas desconocidos.&nbsp;

### 9.2 KPIs de negocio (prueba, 5% superior \= 892 canciones)

| Estrategia | Precisión@5% | IC 95% | Lift vs. azar | Lift intragénero |
| :---- | :---- | :---- | :---- | :---- |
| Azar | 4,2% | 3,0–5,7 | 1,2× | 1,0× |
| Baseline por género | 8,3% | 6,7–10,3 | 2,4× | 1,6× |
| Regresor HGB de la EP2 | 14,7% | 12,5–17,2 | 4,3× | 2,2× |
| Regresor final EP3 | 12,8% | 10,7–15,1 | 3,8× | 2,2× |
| Clasificador de la EP2 | 29,7% | 26,8–32,8 | 8,7× | 3,0× |
| **Clasificador final EP3** | **30,0%** | 27,1–33,1 | **8,8×** | **3,2×** |

El clasificador captura el 44% de los hits del catálogo promoviendo solo el 5%. El regresor final ordena peor que el de la EP2 (es un modelo de mediana), por lo que **no debe usarse para priorizar**.

### 9.3 Selección de la solución óptima (IE12)

Regla aplicada **antes** de mirar la prueba: entre los modelos con MAE de validación cruzada a menos de 1 error estándar del mejor (umbral 10,384), se elige el más simple.

| Modelo | MAE CV | ¿Dentro del umbral? | Ajuste (s) | Predicción de 10.000 canciones (s) |
| :---- | :---- | :---- | :---- | :---- |
| **HGB** | **10,339** | **Sí** | 11 | 0,15 |
| *Stacking* (meta MAE) | 10,403 | No | 260 | 0,40 |
| *Voting* | 10,929 | No | 90 | 0,42 |
| Random Forest | 11,219 | No | 75 | 0,23 |
| Ridge | 11,914 | No | 0,2 | 0,01 |

**Solución elegida:** (1) **clasificador HGB ajustado** para priorizar, (2) **regresor HGB con pérdida de error absoluto** para estimar el nivel de popularidad y (3) los **4 segmentos de K-Means** para comparar un lanzamiento con canciones de su perfil. El modelo mejora 40,8% el MAE contra la media global y 16,0% contra el baseline por género, pero solo **1,5%** (IC 0,009–0,295 puntos) contra un HGB que solo conoce el género: **el sonido explica poco**.

## 10\. Conclusiones, limitaciones y recomendaciones

**Conclusiones.**

1. Optimizar mejora el rendimiento (HGB: 11,05 → 10,12 de MAE en prueba; Random Forest: 13,89 → 10,91), sobre todo por la función de pérdida y por quitar límites al bosque.  
2. Los ensambles no mejoran al mejor modelo individual: errores correlacionados y un modelo base dominante.  
3. El modelo generaliza a canciones nuevas (brecha 0,7), con peor desempeño en artistas nuevos.  
4. El género explica la mayor parte de la señal; el audio agrega \~1,5%.

**Limitaciones.** Circularidad de `popularity` (métrica de la plataforma); cuota artificial de 1.000 canciones por género; datos publicados en 2023; ceros sospechosos; ausencia de variables de artista, promoción, *playlists* y fecha; partición por canción y no por artista; búsqueda de hiperparámetros con presupuesto acotado; intervalos por *bootstrap* de canciones; cifras dependientes de la versión de las librerías.

**Sesgos y ética.** Sesgo de la fuente (el modelo reproduce lo que Spotify promueve), de selección (cuota por género), de datos faltantes (ceros concentrados en ciertos géneros) y de popularidad previa (sirve para "lo que ya funciona", no para descubrir talento). Los nombres de artistas son datos personales (Ley 19.628 y Ley 21.719 en Chile; GDPR como referencia) y el modelo **no los usa como variable**. El modelo debe ser **apoyo a la decisión humana**, nunca un mecanismo para excluir artistas o géneros, y conviene revisar la diversidad de lo que se promueve para no homogeneizar la oferta.

**Recomendaciones para el negocio.**

1. Usar el **clasificador** como filtro de priorización (≈ 30% de hits en el 5% superior contra 3,4% al azar) y comparar canciones **dentro de su género o segmento**.  
2. No interpretar la popularidad predicha como valor esperado: el regresor se equivoca en ±10 puntos y subestima a los hits.  
3. No invertir en "sonar como un hit": para mejorar de verdad hacen falta audiencia del artista, promoción, *playlists* y fecha de lanzamiento.  
4. Esperar peor desempeño con artistas desconocidos y validar por artista en la próxima iteración.  
5. Contrastar `popularity` con datos propios de la empresa antes de pasar a producción y aclarar qué significa un 0\.