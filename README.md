# 🎵 Spotify Tracks – Proyecto de Machine Learning (EP2 + EP3)

**Asignatura:** MLY1101 – Machine Learning
**Caso:** C – Inteligencia musical y predicción de popularidad
**Grupo:** 5 – Diego Marambio, Sebastián Álvarez, Álvaro Sepúlveda y Erick Rosales
**Repositorio:** <https://github.com/Sebiac/Machine-Learning>

Este proyecto conserva el trabajo de la EP2 (EDA, modelos, K-Means, sesgos) y agrega la EP3: **optimización de hiperparámetros, ensambles, validación cruzada agrupada y selección justificada de la solución final**. El informe completo está en [`Informe_Tecnico_EP3.md`](Informe_Tecnico_EP3.md).

## Problema de negocio

¿Pueden las características de una canción (audio y género) ayudar a un equipo de A&R a priorizar qué temas evaluar para promoción? La variable objetivo es `popularity` (0–100), un índice propio de Spotify; no mide calidad artística ni demuestra causalidad. Un *hit* es una canción con `popularity ≥ 70`.

## Resultados de la EP3 (una sola ejecución del notebook)

| | |
|---|---|
| Datos | 89.583 canciones únicas, 114 géneros; 71.727 de entrenamiento y 17.856 de prueba; 0 grupos compartidos |
| **Regresor elegido** | HistGradientBoosting con pérdida de error absoluto: **MAE 10,12** en prueba (IC 95%: 9,85–10,42); validación cruzada 10,34 ± 0,10 |
| **Clasificador elegido** | HistGradientBoosting ajustado: **Precisión@5% 30,0%** (IC 27,1–33,1; azar 3,4%), lift 8,8×, PR-AUC 0,304 |
| Ensambles | *Voting* (MAE 10,69) y *stacking* (10,18) **no mejoran** al HGB solo (10,12) |
| Aporte del audio | Sobre el baseline por género, el modelo mejora 16,0%; sobre un HGB que solo usa género, 1,5% |
| Generalización | Brecha entrenamiento–prueba de 0,7; con artistas nuevos el MAE sube a 10,72 |

## Qué incluye la EP3

- `RandomizedSearchCV` para Random Forest y HGB (incluida la función de pérdida), grilla fina para Ridge y un refinamiento posterior del HGB, todo sobre el 100% del entrenamiento.
- `GroupKFold` (3 pliegues para buscar, 5 para comparar) con grupos artista + nombre de canción.
- Ensambles: Random Forest (*bagging*), HGB (*boosting*), `VotingRegressor` y `StackingRegressor` con un meta-modelo que minimiza el MAE.
- Generalización: brecha entrenamiento–prueba, curva de aprendizaje, validación agrupada por artista e intervalos de confianza por *bootstrap* de grupos.
- Selección con regla de 1 error estándar, comparación contra el baseline por género y clasificador de hits.
- Preprocesamiento dentro de `Pipeline` para evitar fuga de información.

## Estructura

```text
Machine-Learning/
├── data/Spotify_Tracks_Dataset.csv
├── notebooks/
│   ├── Informe_Tecnico_EP3.ipynb              # notebook de la EP3 (ejecutado)
│   └── Informe_Tecnico_EP2_Referencia.ipynb   # EDA, sesgos, K-Means y modelos de la EP2
├── src/modelos_ep3.py                         # baseline por género y meta-modelo de stacking
├── images/                                    # gráficos (01–12 EP2, 13–17 EP3)
├── models/                                    # modelos entrenados y métricas
├── Informe_Tecnico_EP3.md
├── requirements.txt
└── README.md
```

## Cómo ejecutar

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/Informe_Tecnico_EP3.ipynb
```

Ejecutar **Restart & Run All**. El notebook detecta solo la ruta del CSV, ya sea que se abra desde `notebooks/` o desde la raíz. Tarda ≈ 32 minutos en un solo núcleo; con `MODO_RAPIDO = True` (primera celda de código) corre en pocos minutos con datos submuestreados, solo para comprobar que el flujo funciona. En Google Colab, clonar el repositorio completo para conservar las rutas relativas.

## Archivos generados por la EP3

- `models/modelo_final_regresor.joblib` y `models/modelo_final_clasificador.joblib` (*pipelines* completos, se cargan con `joblib.load`)
- `models/modelo_final_metricas.json` (hiperparámetros y métricas) y `models/resultados_ep3.csv` (comparación de modelos)
- `images/13_ep3_comparacion_mae.png` a `images/17_ep3_curva_aprendizaje.png`

## Limitaciones

El dataset tiene una distribución artificialmente balanceada por género, ceros de popularidad de interpretación incierta y fecha de publicación 2023; no contiene promoción, alcance del artista, *playlists* ni fecha de lanzamiento. Las particiones agrupan por canción, no por artista, por lo que las cifras son algo optimistas para artistas nuevos. El modelo es un apoyo para priorizar el análisis humano, nunca para excluir artistas o géneros automáticamente.
