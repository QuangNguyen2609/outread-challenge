from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.offline as pyo

class ClusterVisualizer:
    def __init__(self, vectors: np.ndarray, labels: np.ndarray, output_path: str, visualize_method: str) -> None:
        """
        Initialization for ClusterAnalyzer

        Parameters:
        - vectors: Array of input vectors for clustering.
        - labels: Array of cluster labels.
        - output_path: Path to save the silhouette plot.
        - visualize_method: Dimensionality reduction method to visualize

        """

        self.vectors = vectors
        self.labels = labels
        self.output_path = output_path
        self.init_method = init_method
        self.visualize_method = visualize_method
        self.colors = ['rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 2, 0.8)', 'rgba(0, 255, 200, 0.8)', 'rgba(0, 128, 255, 0.8)', 
                       'rgba(255, 0, 0, 0.8)', 'rgba(255, 255, 0, 0.8)', 'rgba(0, 255, 255, 0.8)', 'rgba(128, 0, 255, 0.8)', 
                       'rgba(255, 128, 0, 0.8)']

    def visualize_clusters(self) -> None:
        """
        Visualizes clusters using PCA or T-SNE.

        """
        
        if self.visualize_method == "pca":
            dim_reduction_model = PCA(n_components=2)
            reduced_vectors = dim_reduction_model.fit_transform(self.vectors)
        else:
            dim_reduction_model = TSNE(n_components=2, random_state=42)
            reduced_vectors = dim_reduction_model.fit_transform(self.vectors)

        df = pd.DataFrame({
            'PC1_2d': reduced_vectors[:, 0],
            'PC2_2d': reduced_vectors[:, 1],
            'label': self.labels,
            'title': self.titles
        })

        clusters = df['label'].unique()
        data = []

        for i, cluster in enumerate(clusters):
            cluster_data = df[df['label'] == cluster]
            trace = go.Scatter(
                x=cluster_data["PC1_2d"],
                y=cluster_data["PC2_2d"],
                mode="markers",
                name=f"Cluster {cluster}",
                marker=dict(color=self.colors[i % len(self.colors)]),
                text=cluster_data["title"]
            )
            data.append(trace)

        title = "Visualizing Clusters in Two Dimensions Using " + self.visualize_method.upper()
        layout = dict(title=title,
                      xaxis=dict(title='PC1' if visualize_method == 'pca' else 't-SNE1', ticklen=5, zeroline=False),
                      yaxis=dict(title='PC2' if visualize_method == 'pca' else 't-SNE2', ticklen=5, zeroline=False)
                     )

        fig = dict(data=data, layout=layout)
        pyo.plot(fig, filename=os.path.join(self.output_path, 'cluster_visualization.html'))
