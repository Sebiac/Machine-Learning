"""Modelos auxiliares de la EP3 (Caso C - Spotify Tracks).

Viven en un módulo aparte (y no dentro del notebook) para que los modelos guardados con ``joblib``
se puedan volver a cargar desde cualquier script: basta con tener la carpeta raíz del proyecto en ``sys.path``.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


class BaseGlobal:
    """Baseline mínimo: predice siempre la media global de popularidad del entrenamiento."""

    def fit(self, X, y):
        self.media_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.media_)


class BaselineGenero:
    """Baseline que solo conoce el género.

    Para cada género calcula la popularidad media en el entrenamiento. Como una canción puede tener varios
    géneros (multi-hot), su predicción es el promedio de las medias de sus géneros.
    """

    def __init__(self, gen_cols):
        self.gen_cols = list(gen_cols)

    def fit(self, X, y):
        M = X[self.gen_cols].to_numpy(float)
        y = np.asarray(y, float)
        self.media_genero_ = (M * y[:, None]).sum(0) / np.maximum(M.sum(0), 1)
        self.media_ = float(y.mean())
        return self

    def predict(self, X):
        M = X[self.gen_cols].to_numpy(float)
        n = M.sum(1)
        return np.where(n > 0, (M * self.media_genero_).sum(1) / np.maximum(n, 1), self.media_)


class PesosMinimoMAE(RegressorMixin, BaseEstimator):
    """Meta-modelo de *stacking* orientado al MAE (la métrica del KPI técnico).

    Combina las predicciones de los modelos base con pesos no negativos y un intercepto, elegidos para
    minimizar el error absoluto medio. Son solo k + 1 parámetros, así que se optimiza en segundos.
    """

    def fit(self, Z, y):
        Z = np.asarray(Z, float)
        y = np.asarray(y, float)
        k = Z.shape[1]

        def mae(p):
            return np.mean(np.abs(y - (Z @ np.abs(p[:k]) + p[k])))

        r = minimize(mae, np.r_[np.full(k, 1.0 / k), 0.0], method="Powell")
        self.coef_ = np.abs(r.x[:k])
        self.intercept_ = float(r.x[k])
        return self

    def predict(self, Z):
        return np.asarray(Z, float) @ self.coef_ + self.intercept_
