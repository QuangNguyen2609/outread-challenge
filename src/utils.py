import pandas as pd 
import numpy as np 

 def chunks(lst: list, n: int) -> List:
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
    cluster_counts = pd.Series(labels).value_counts()
    print(f"Number of clusters: {len(cluster_counts)}")
    print("Number of papers in each cluster:" cluster_counts)
    return cluster_counts
