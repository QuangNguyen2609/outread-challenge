import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
from typing import List

class ClusterVisualizer:
    def __init__(self, output_path: str, visualize_method: str, max_clusters: int) -> None:
        """
        Initialization for ClusterAnalyzer

        Parameters:
        - output_path: Path to save the silhouette plot.
        - visualize_method: Dimensionality reduction method to visualize
        - init_method: Method to initialize centroid for KMeans
        - max_clusters: Maximum number of cluster we can reach

        """

        self.output_path = output_path
        self.visualize_method = visualize_method
        self.max_clusters = max_clusters
        self.colors = self.generate_colors()
        
    def generate_colors(self) -> List[str]:
        """
        Generate a list of colors based on the maximum number of clusters using a distinct colormap.

        Returns:
        - List of RGBA color strings.
        """
        # Use the 'tab20' colormap
        colormap = plt.get_cmap('tab20')
        colors = [colormap(i % 20) for i in range(self.max_clusters)]
        
        # Convert RGBA to 'rgba(r, g, b, a)' string format
        colors_rgba = [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})' for r, g, b, a in colors]
        
        return colors_rgba

    def visualize_clusters(self, vectors: np.ndarray, labels: np.ndarray, titles: List[str]) -> None:
        """
        Visualizes clusters using PCA or T-SNE.

        Parameters:
        - vectors: The 2D array of vectors to visualize, after PCA transformation.
        - labels: Array of cluster labels corresponding to each vector.
        - titles: List of titles corresponding to each vector (for hover text in the plot).
        """
        
        if self.visualize_method == "pca":
            dim_reduction_model = PCA(n_components=2)
            reduced_vectors = dim_reduction_model.fit_transform(vectors)
        else:
            dim_reduction_model = TSNE(n_components=2, random_state=42)
            reduced_vectors = dim_reduction_model.fit_transform(vectors)

        df = pd.DataFrame({
            'PC1_2d': reduced_vectors[:, 0],
            'PC2_2d': reduced_vectors[:, 1],
            'label': labels,
            'title': titles
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

        title = f"Visualizing Clusters in Two Dimensions Using {self.visualize_method.upper()}"
        layout = dict(title=title,
                      xaxis=dict(title='PC1' if self.visualize_method == 'pca' else 't-SNE1', 
                                 ticklen=5, zeroline=False),
                      yaxis=dict(title='PC2' if self.visualize_method == 'pca' else 't-SNE2',
                                  ticklen=5, zeroline=False)
                     )

        fig = dict(data=data, layout=layout)
        pyo.plot(fig, filename=os.path.join(self.output_path, 'cluster_visualization.html'))
