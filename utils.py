import pandas as pd 
import numpy as np 
from typing import List, Generator

def chunks(lst: list, n: int) -> Generator:
        """
        Yield successive n-sized chunks from lst.

        Parameters:
        - lst: List to split into chunks.
        - n: Size of each chunk.

        Returns:
        - Generator yielding lists of size n from lst.
        """

        for i in range(0, len(lst), n):
            yield lst[i:i + n]

def generate_summary_report(labels: np.ndarray) -> pd.Series:
    """
    Generate a summary report of the clustering results.

    Parameters:
    - labels (np.ndarray): Array of cluster labels.

    Returns:
    - Series containing the counts of papers in each cluster.
    """

    cluster_counts = pd.Series(labels).value_counts()
    print(f"\nNumber of clusters: {len(cluster_counts)}")
    print("Number of papers in each cluster:")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count}")
    return cluster_counts
