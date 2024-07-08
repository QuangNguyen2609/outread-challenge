import os
import multiprocessing
from time import time
from typing import List, Tuple
import pandas as pd
import numpy as np
from time import time
from preprocessing import PDFExtractor, TextPreprocessor
from embedding import TextVectorizer
from clustering import KMeansCluster
from evaluation import Evaluator
from visualization import ClusterVisualizer
from utils import chunks, generate_summary_report

class GreenEnergyClustering:
    def __init__(self, input_path: str, output_path: str, embedding_type: str, embedding_dim: int, window_size: int, parallel: bool, num_workers: int, visualize_method: str) -> None:
        """
        Initializes the GreenEnergyClustering instance.

        Parameters:
        - input_path: str - Path to the directory containing input PDF files with abstracts to be clustered.
        - output_path: str - Path to the directory where clustering results and visualizations will be saved.
        - embedding_type: str - Type of text embedding technique to use for vectorization ('tfidf' or other methods like 'word2vec').
        - embedding_dim: int - Dimensionality of the embeddings (e.g., vector size for word embeddings like word2vec).
        - window_size: int - Window size parameter for certain embedding methods (e.g., word2vec). Not currently used.
        - parallel: bool - Flag indicating whether to enable parallel processing for abstract extraction and vectorization.
        - num_workers: int - Number of worker processes to use for parallel processing. Applies when parallel=True.
        """

        self.input_path = input_path
        self.output_path = output_path
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.parallel = parallel
        self.num_workers = num_workers
        self.visualize_method = visualize_method
        self.pdf_extractor = PDFExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.text_vectorizer = TextVectorizer()
        self.kmeans_cluster = KMeansCluster()
        self.evaluator = Evaluator()
        self.cluster_visualizer = ClusterVisualizer()

    def extract_abstracts_serial(self, files: List[str]) -> Tuple[List[str], List[str]]:
        """
        Extract abstracts from PDF files and preprocess them in serial.

        Parameters:
        - files: List of file names to process.

        Returns:
        - Tuple of updated train_data and file_paths lists.
        """

        train_data = []
        file_paths = []
        for file in files:
            file_path = os.path.join(self.input_path, file)
            pdf_text = self.pdf_extractor.read_pdf(file_path)
            abstract_text = self.pdf_extractor.extract_abstract_from_pdf_text(pdf_text)
            if abstract_text == "No abstract found.":
                paragraphs = re.split(r'\n\s*\n', pdf_text)
                fallback_abstract = " ".join(paragraphs[:2])
                fallback_abstract = self.pdf_extractor.filter_title_and_author(fallback_abstract)
                abstract_text = fallback_abstract.strip()
            preprocessed_abstracts = self.text_preprocessor.preprocess_text(abstract_text)
            train_data.append(preprocessed_abstracts)
            file_paths.append(file_path)
        
        return train_data, file_paths

    def extract_abstracts_parallel(self, files_chunk: List[List[str]]) -> Tuple[List[str], List[str]]:
        """
        Extract abstracts from PDF files and preprocess them in parallel.

        Parameters:
        - files_chunk: List of file names to process.

        Returns:
        - Tuple of updated train_data and file_paths lists.
        """

        manager = multiprocessing.Manager()
        train_data = manager.list()
        file_paths = manager.list()

        start_time = time()
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.starmap(self.extract_abstracts_serial, [(chunk, train_data, file_paths) for chunk in files_chunk])

        print(f"Time taken for extraction: {time() - start_time:.2f} seconds")
        return list(train_data), list(file_paths)

    def extract_abstracts(self) -> Tuple[List[str], List[str]]:
        """
        Extract abstracts from PDF files and preprocess them.

        Returns:
        - Tuple of updated train_data and file_paths lists.
        """

        files = os.listdir(self.input_path)

        # Split files into chunks corresponding to num_workers
        files_chunk = chunks(files, self.num_workers)

        print("Extracting abstracts from PDF files...")
        if self.parallel:
            train_data, file_paths = self.extract_abstracts_parallel(files_chunk)
        else:
            train_data, file_paths = self.extract_abstracts_serial(files)

        print(f"Number of abstracts extracted: {len(train_data)}")
        print()

        return train_data, file_paths

    def vectorize_texts(self, train_data: List[str]) -> np.ndarray:
        """
        Vectorize data into embedding vectors.

        Parameters:
        - train_data: List of texts to vectorize.

        Returns:
        - Tuple of vectorizer and embedding vectors.
        """

        if self.embedding_type == "tfidf":
            vectors, vectorizer = self.text_vectorizer.vectorize_texts_tfidf(train_data)
        else:
            vectors, model = self.text_vectorizer.vectorize_texts_word2vec(train_data, self.window_size, self.embedding_dim)
        return vectors

    def run_clustering(self, vectors: np.ndarray, file_paths: List[str]):
        """
        Executes the clustering process.

        Parameters:
        - vectors (np.ndarray): Array of input vectors for clustering.
        - file_paths (List[str]): List of file paths corresponding to each vector.

        """

        optimal_clusters = self.kmeans_cluster.silhouette_analysis(vectors, self.output_path)
        kmeans, labels = self.kmeans_cluster.cluster_texts_kmeans(vectors, optimal_clusters)
        self.kmeans_cluster.generate_summary_report(labels)
        silhouette_avg, davies_bouldin_avg = self.evaluator.evaluate_clustering(vectors, labels)
        print(f"Silhouette Score: {silhouette_avg}, Davies-Bouldin Score: {davies_bouldin_avg}")
        self.save_clustering_results(labels, file_paths)
        self.cluster_visualizer.visualize_clusters(vectors, labels, file_paths, self.output_path, self.visualize_method)
        generate_summary_report(labels)

    def save_clustering_results(self, labels: np.ndarray, file_paths: List[str]) -> None:
        """
        Saves clustering results to a CSV file.

        Parameters:
        - labels (np.ndarray): Array of cluster labels assigned by the clustering algorithm.
        - file_paths (List[str]): List of file paths corresponding to each data point.

        """

        results = pd.DataFrame({'File': file_paths, 'Cluster': labels})
        results.to_csv(os.path.join(self.output_path, 'clustering_results.csv'), index=False)

    def run(self):
        """
        Executes the full pipeline for clustering Green Energy abstracts.

        Steps:
        1. Extracts abstracts from PDF files.
        2. Vectorizes the extracted abstracts.
        3. Performs clustering on the vectorized data.

        """
        
        os.makedirs(self.output_path, exist_ok=True)
        train_data, file_paths = self.extract_abstracts()
        vectors = self.vectorize_texts(train_data)
        self.run_clustering(vectors, file_paths)

