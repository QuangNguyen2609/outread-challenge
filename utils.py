import pandas as pd 
import numpy as np 
import os
from typing import List, Generator
from sklearn.feature_extraction.text import TfidfVectorizer

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

def generate_summary_report(texts: List[str], labels: np.ndarray) -> pd.DataFrame:
    """
    Generate a summary report including the number of clusters, the number of papers in each cluster, and key terms/topics for each cluster.

    Parameters:
    - texts: List of original texts corresponding to the vectors.
    - labels: Array of cluster labels.

    Returns:
    - DataFrame containing cluster summary.
    """
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    summary = {'Cluster': [], 'Number of Papers': [], 'Key Terms': []}

    for cluster_num in cluster_counts.index:
        cluster_texts = [text for text, label in zip(texts, labels) if label == cluster_num]
        tfidf_vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)
        mean_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        terms_scores = zip(tfidf_vectorizer.get_feature_names_out(), mean_tfidf_scores)
        sorted_terms_scores = sorted(terms_scores, key=lambda x: x[1], reverse=True)[:5]
        top_terms = [term for term, score in sorted_terms_scores]
        summary['Cluster'].append(cluster_num)
        summary['Number of Papers'].append(cluster_counts[cluster_num])
        summary['Key Terms'].append(", ".join(top_terms))

    summary_df = pd.DataFrame(summary)
    print(summary_df)
    return summary_df


def save_clustering_results(labels: np.ndarray, file_paths: List[str], output_path: str) -> None:
    """
    Saves clustering results to a CSV file.

    Parameters:
    - labels (np.ndarray): Array of cluster labels assigned by the clustering algorithm.
    - file_paths (List[str]): List of file paths corresponding to each data point.

    """
    results = pd.DataFrame({'File': file_paths, 'Cluster': labels})
    results.to_csv(os.path.join(output_path, 'clustering_results.csv'), index=False)