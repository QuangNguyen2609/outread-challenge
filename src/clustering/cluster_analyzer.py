from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from .abstract_cluster import AbstractCluster

# class ClusterAnalyzer:
#     def __init__(self, output_path: str, init_method: str, seed: int, verbose: bool) -> None:
#         """
#         Initialization for ClusterAnalyzer

#         Parameters:
#         - vectors: Array of input vectors for clustering.
#         - labels: Array of cluster labels.
#         - output_path: Path to save the silhouette plot.
#         - init_method: Method to initialize centroid for KMeans

#         """

#         self.output_path = output_path
#         self.init_method = init_method
#         self.seed = seed
#         self.verbose = verbose

#     def silhouette_analysis(self, vectors: np.ndarray, range_n_clusters: range) -> int:
#         """
#         Perform silhouette analysis to find optimal number of clusters.

#         Parameters:
#         - vectors: Array of input vectors for clustering.
#         - output_path: Path to save the silhouette plot.
#         - range_n_clusters: Range of number of clusters to evaluate (default: range(2, 10+1)).

#         Returns:
#         - Optimal number of clusters based on silhouette score.
#         """

#         silhouette_avg_scores = []
#         for n_clusters in range_n_clusters:
#             kmeans = KMeans(n_clusters=n_clusters, init=self.init_method, random_state=self.seed)
#             cluster_labels = kmeans.fit_predict(vectors)
#             silhouette_avg = silhouette_score(vectors, cluster_labels)
#             silhouette_avg_scores.append(silhouette_avg)
#             if self.verbose:
#                 print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")

#         plt.figure()
#         plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
#         plt.xlabel('Number of clusters')
#         plt.ylabel('Average Silhouette Score')
#         plt.title('Silhouette Analysis For Optimal Number of Clusters')
#         plt.savefig(os.path.join(self.output_path, "silhouette_analysis.png"))
#         plt.close()

#         optimal_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
#         return optimal_clusters

#     def cluster_texts_kmeans(self, vectors: np.ndarray, n_clusters: int, seed: int = 42) -> Tuple[KMeans, np.ndarray]:
#         """
#         Cluster texts using KMeans algorithm.

#         Parameters:
#         - n_clusters: Number of clusters to create.
#         - random_state: set state for reproducibility

#         Returns:
#         - Tuple of trained KMeans model and array of cluster labels.
#         """

#         kmeans = KMeans(n_clusters=n_clusters, init=self.init_method, random_state=seed)
#         kmeans.fit(vectors)
#         labels = kmeans.labels_
#         return kmeans, labels


class KMeansCluster(AbstractCluster):
    def __init__(self, vectors: np.ndarray, output_path: str, verbose: bool, init_method: str, seed: int):
        """
        Initialize the KMeansCluster with the given parameters.

        Parameters:
        - vectors: Array of input vectors for clustering.
        - output_path: Path to save the clustering results.
        - verbose: Boolean flag to control verbosity of the output.
        - init_method: Method to initialize centroids for KMeans.
        - seed: Seed for random number generator to ensure reproducibility.
        """

        super().__init__(vectors, output_path, verbose)
        self.init_method = init_method
        self.seed = seed

    def cluster_texts(self, n_clusters: int) -> Tuple[KMeans, np.ndarray]:
        """
        Cluster texts using KMeans algorithm.

        Parameters:
        - n_clusters: Number of clusters to create.

        Returns:
        - Tuple containing the trained KMeans model and the array of cluster labels.
        """

        kmeans = KMeans(n_clusters=n_clusters, init=self.init_method, random_state=self.seed)
        labels = kmeans.fit_predict(self.vectors)
        return kmeans, labels

class DBSCANCluster(AbstractCluster):
    def __init__(self, vectors: np.ndarray, output_path: str, verbose: bool, eps: float = 0.5, min_samples: int = 5) -> None:
        """
        Initialize the DBSCANCluster with the given parameters.

        Parameters:
        - vectors: Array of input vectors for clustering.
        - output_path: Path to save the clustering results.
        - eps: Maximum distance between two samples for them to be considered as in the same neighborhood.
        - min_samples: Number of samples in a neighborhood for a point to be considered as a core point.
        """

        super().__init__(vectors, output_path, verbose)
        self.eps = eps
        self.min_samples = min_samples

    def cluster_texts(self) -> Tuple[DBSCAN, np.ndarray]:
        """
        Cluster texts using DBSCAN algorithm.

        Returns:
        - Tuple containing the trained DBSCAN model and the array of cluster labels.
        """

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(self.vectors)
        return dbscan, labels

class HierarchicalCluster(AbstractCluster):
    def __init__(self, vectors: np.ndarray, output_path: str, verbose: bool, linkage: str = 'ward'):
        """
        Initialize the HierarchicalCluster with the given parameters.

        Parameters:
        - vectors: Array of input vectors for clustering.
        - output_path: Path to save the clustering results.
        - linkage: Which linkage criterion to use. Options are 'ward', 'complete', 'average', 'single'.
        """

        super().__init__(vectors, output_path, verbose)
        self.linkage = linkage

    def cluster_texts(self, n_clusters: int, ) -> Tuple[AgglomerativeClustering, np.ndarray]:
        """
        Cluster texts using Hierarchical clustering algorithm.

        Parameters:
        - n_clusters: Number of clusters to create.

        Returns:
        - Tuple containing the trained AgglomerativeClustering model and the array of cluster labels.
        """

        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=self.linkage)
        labels = hierarchical.fit_predict(self.vectors)
        return hierarchical, labels
