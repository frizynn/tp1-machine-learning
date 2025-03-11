import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..clustering.base import ClusteringModel


class KMeans(ClusteringModel):
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, random_state=42):
        """
        Inicializa el K-Means con:
          - n_clusters: número de clusters.
          - max_iter: número máximo de iteraciones.
          - tol: tolerancia para la convergencia.
          - random_state: semilla para la aleatoriedad.
        """
        super().__init__(n_clusters)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None 

    def fit(self, X):
        """
        Ajusta el modelo K-Means a los datos X (array de shape [n_muestras, n_features]).
        """
        np.random.seed(self.random_state)

        random_indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
        centroids = X[random_indices, :]

        for i in range(self.max_iter):

            labels = np.array([np.argmin(np.sum((x - centroids) ** 2, axis=1)) for x in X])
            
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(self.n_clusters)
            ])
            
            shift = np.sum((centroids - new_centroids) ** 2)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels

    def predict(self, X):
        """
        Dado un conjunto de datos X, asigna cada muestra al cluster cuyo centroide
        esté más cercano.
        """
        if self.centroids_ is None:
            raise ValueError("El modelo no ha sido ajustado aún. Ejecuta fit() primero.")
        return np.array([np.argmin(np.sum((x - self.centroids_) ** 2, axis=1)) for x in X])

