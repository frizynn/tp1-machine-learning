
class ClusteringModel:
    def __init__(self, n_clusters=2):
        """
        Inicializa el modelo de clustering con el número deseado de clusters.
        """
        self.n_clusters = n_clusters
        self.labels_ = None  

    def fit(self, X):
        """
        Ajusta el modelo a los datos X.
        Este método debe implementarse en las subclases.
        """
        raise NotImplementedError("El método fit() debe implementarse en la subclase.")

    def predict(self, X):
        """
        Predice la etiqueta de cluster para cada muestra en X.
        Este método debe implementarse en las subclases.
        """
        raise NotImplementedError("El método predict() debe implementarse en la subclase.")

    def fit_predict(self, X):
        """
        Combina fit() y predict() para ajustar el modelo y retornar las etiquetas.
        """
        self.fit(X)
        return self.labels_