import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..clustering.base import ClusteringModel


class KMeans(ClusteringModel):
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, random_state=42):
        """
        Initialize the K-Means with:
          - n_clusters: number of clusters.
          - max_iter: maximum number of iterations.
          - tol: tolerance for convergence.
          - random_state: seed for randomness.
        """
        super().__init__(n_clusters)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None

    def fit(self, X):
        """
        Train the K-Means model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dataset to fit.

        Returns:
        --------
        self : KMeans
            The fitted model.
        """

        np.random.seed(self.random_state)

        random_indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
        centroids = X[random_indices, :]

        for i in range(self.max_iter):

            labels = np.array(
                [np.argmin(np.sum((x - centroids) ** 2, axis=1)) for x in X]
            )

            new_centroids = np.array(
                [
                    X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                    for j in range(self.n_clusters)
                ]
            )

            shift = np.sum((centroids - new_centroids) ** 2)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels

    def predict(self, X):
        """
        Predicts cluster labels for a dataset X.
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset to predict.

        Returns:
        --------
        labels : array-like of shape (n_samples,)
            Cluster labels for each sample.
        """

        if self.centroids_ is None:
            raise ValueError("Model has not been fitted yet. Run fit() first.")
        return np.array(
            [np.argmin(np.sum((x - self.centroids_) ** 2, axis=1)) for x in X]
        )

    @property
    def cluster_centers_(self):
        """Alias for centroids_ to maintain compatibility with sklearn"""
        return self.centroids_
