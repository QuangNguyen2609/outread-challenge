from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

class Evaluator:
    def __init__(self) -> None:
        """
        Initialization for Evaluator

        """
        pass
    
    def evaluate_silhouette_score(self, vectors: np.ndarray, labels: np.ndarray):
        """
        Evaluate the silhouette score for the clustering.

        Returns:
        - Silhouette score (float).
        """
        silhouette_avg = silhouette_score(vectors, labels)
        return silhouette_avg

    def evaluate_davies_bouldin_score(self, vectors: np.ndarray, labels: np.ndarray):
        """
        Evaluate the Davies-Bouldin score for the clustering.

        Returns:
        - Davies-Bouldin score (float).
        """
        davies_bouldin_avg = davies_bouldin_score(vectors, labels)
        return davies_bouldin_avg

    def evaluate_clustering(self, vectors: np.ndarray, labels: np.ndarray):
        """
        Evaluate clustering performance using multiple metrics.

        Returns:
        - Tuple of silhouette score (float) and Davies-Bouldin score (float).
        """
        silhouette_avg = silhouette_score(vectors, labels)
        davies_bouldin_avg = davies_bouldin_score(vectors, labels)
        return silhouette_avg, davies_bouldin_avg