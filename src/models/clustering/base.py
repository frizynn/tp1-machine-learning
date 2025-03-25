class ClusteringModel:
    def __init__(self, n_clusters=2):
        """
        Initialize the clustering model with the desired number of clusters.

        Parameters:
        -----------
        n_clusters : int, default=2
            Number of clusters.
        """
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        """
        Fit the model to the data X.
        This method must be implemented in subclasses.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dataset to fit.
        """
        raise NotImplementedError(
            "The fit() method must be implemented in the subclass."
        )

    def predict(self, X):
        """
        Predict the cluster label for each sample in X.
        This method must be implemented in subclasses.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dataset to predict.
        """
        raise NotImplementedError(
            "The predict() method must be implemented in the subclass."
        )

    def fit_predict(self, X):
        """
        Combines fit() and predict() to fit the model and return the labels.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dataset to fit and predict.
        """
        self.fit(X)
        return self.labels_
