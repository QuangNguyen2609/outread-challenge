from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

class ClusterAnalyzer:
    def __init__(self, vectors: np.ndarray, labels: np.ndarray, output_path: str, init_method: str) -> None:
        """
        Initialization for ClusterAnalyzer

        Parameters:
        - vectors: Array of input vectors for clustering.
        - labels: Array of cluster labels.
        - output_path: Path to save the silhouette plot.
        - init_method: Method to initialize centroid for KMeans

        """

        self.vectors = vectors
        self.labels = labels
        self.output_path = output_path
        self.init_method = init_method

    def silhouette_analysis(self, range_n_clusters: range = range(2, 10)) -> int:
        """
        Perform silhouette analysis to find optimal number of clusters.

        Parameters:
        - vectors: Array of input vectors for clustering.
        - output_path: Path to save the silhouette plot.
        - range_n_clusters: Range of number of clusters to evaluate (default: range(2, 10)).

        Returns:
        - Optimal number of clusters based on silhouette score.
        """

        silhouette_avg_scores = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.vectors)
            silhouette_avg = silhouette_score(self.vectors, cluster_labels)
            silhouette_avg_scores.append(silhouette_avg)
            print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")

        plt.figure()
        plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Average Silhouette Score')
        plt.title('Silhouette Analysis For Optimal Number of Clusters')
        plt.savefig(os.path.join(self.output_path, "silhouette_analysis.png"))
        plt.close()

        optimal_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
        return optimal_clusters

    def cluster_texts_kmeans(self, n_clusters: int, random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
        """
        Cluster texts using KMeans algorithm.

        Parameters:
        - n_clusters: Number of clusters to create.
        - random_state: set state for reproducibility

        Returns:
        - Tuple of trained KMeans model and array of cluster labels.
        """

        kmeans = KMeans(n_clusters=n_clusters, init=self.init_method, random_state=random_state)
        kmeans.fit(vectors)
        labels = kmeans.labels_
        return kmeans, labels
