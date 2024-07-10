from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

class AbstractCluster(ABC):
    def __init__(self, vectors: np.ndarray, output_path: str, verbose: bool) -> None:
        """
        Initialize the KMeansCluster with the given parameters.

        Parameters:
        - vectors: Array of input vectors for clustering.
        - output_path: Path to save the clustering results.
        - verbose: Boolean flag to control verbosity of the output.
        """

        self.vectors = vectors
        self.output_path = output_path
        self.verbose = verbose

    @abstractmethod
    def cluster_texts(self, *args, **kwargs) -> Tuple:
        pass

    def silhouette_analysis(self, range_n_clusters: range = range(2, 10)) -> int:
        """
        Perform silhouette analysis to find optimal number of clusters.

        Parameters:
        - range_n_clusters: Range of number of clusters to evaluate (default: range(2, 10)).

        Returns:
        - Optimal number of clusters based on silhouette score.
        """

        print("Running Silhouette Analysis to find optimal number of clusters...")
        silhouette_avg_scores = []
        for n_clusters in range_n_clusters:
            model, labels = self.cluster_texts(n_clusters=n_clusters)
            silhouette_avg = silhouette_score(self.vectors, labels)
            silhouette_avg_scores.append(silhouette_avg)
            if self.verbose:
                print(f"For n_clusters = {n_clusters}, the silhouette score = {silhouette_avg}")
        print("Visualizing clustering result...")
        plt.figure()
        plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Average Silhouette Score')
        plt.title('Silhouette Analysis For Optimal Number of Clusters')
        plt.savefig(os.path.join(self.output_path, "silhouette_analysis.png"))
        plt.close()

        optimal_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
        return optimal_clusters
