import os
import re
import multiprocessing
from time import time
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from src.preprocessing import PDFProcessor, TextPreprocessor
from src.embedding import TFIDFVectorizer, Word2VecVectorizer
from src.clustering import KMeansCluster, DBSCANCluster, HierarchicalCluster
from src.evaluation import Evaluator
from src.visualization import ClusterVisualizer
from utils import chunks, generate_summary_report, save_clustering_results

class Pipeline:
    def __init__(self, input_path: str, output_path: str, embedding_type: str,
                 embedding_dim: int, window_size: int, parallel: bool, num_workers: int, 
                 visualize_method: str, init_method: str, max_clusters: range, seed: int, sg: int,
                 verbose: bool, clustering_method: str, eps: float, min_samples: int, linkage: str) -> None:
        """
        Initialize the Pipeline instance.

        Parameters:
        - input_path: str - Path to the directory containing input PDF files with abstracts to be clustered.
        - output_path: str - Path to the directory where clustering results and visualizations will be saved.
        - embedding_type: str - Type of text embedding technique to use for vectorization ('tfidf' or other methods like 'word2vec').
        - embedding_dim: int - Dimensionality of the embeddings (e.g., vector size for word embeddings like word2vec).
        - window_size: int - Window size parameter for certain embedding methods (e.g., word2vec). Not currently used.
        - parallel: bool - Flag indicating whether to enable parallel processing for abstract extraction and vectorization.
        - num_workers: int - Number of worker processes to use for parallel processing. Applies when parallel=True.
        - visualize_method: str - Dimensionality reduction method for visualizing the clusters (e.g., 'pca', 'tsne').
        - init_method: str - Method to initialize centroids for KMeans clustering.
        - max_clusters: int - Maximum number of clusters to evaluate.
        - seed: int - Random seed for reproducibility.
        - verbose: bool - Flag indicating whether to print verbose output during the process is running.
        - sg: int - Training algorithm (0 for CBOW, 1 for Skip-gram) for Word2Vec.
        - eps: Maximum distance between two samples for them to be considered as in the same neighborhood (for DBSCAN).
        - min_samples: Number of samples in a neighborhood for a point to be considered as a core point (for DBSCAN).
        - linkage: Which linkage criterion to use. Options are 'ward', 'complete', 'average', 'single' (for Hierachical).
        """

        self.input_path = input_path
        self.output_path = output_path
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.parallel = parallel
        self.num_workers = num_workers
        self.visualize_method = visualize_method
        self.init_method = init_method
        self.seed = seed
        self.verbose = verbose
        self.n_clusters_range = range(2, max_clusters + 1)
        self.clustering_method = clustering_method
        self.eps = eps 
        self.min_samples = min_samples
        self.linkage = linkage
        self.sg = sg
        self.pdf_extractor = PDFProcessor(input_path)
        self.text_preprocessor = TextPreprocessor()
        self.evaluator = Evaluator()
        self.cluster_visualizer = ClusterVisualizer(output_path, visualize_method)

    def extract_abstracts_serial(self, files: List[str],  train_data: List[str], file_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Extract abstracts from PDF files and preprocess them in serial.

        Parameters:
        - files: List of file names to process.

        Returns:
        - Tuple of updated train_data and file_paths lists.
        """

        for file in tqdm(files):
            file_path = os.path.join(self.input_path, file)
            pdf_text = self.pdf_extractor.read_pdf(file_path)
            abstract_text = self.pdf_extractor.extract_abstract_from_pdf_text(pdf_text)
            if len(abstract_text.split()) <= 1:
                # remove outlier case
                continue
            if abstract_text == "No abstract found.":
                paragraphs = re.split(r'\n\s*\n', pdf_text)
                fallback_abstract = " ".join(paragraphs[:3])
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

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.starmap(self.extract_abstracts_serial, 
                         [(chunk, train_data, file_paths) for chunk in files_chunk])

        return list(train_data), list(file_paths)

    def extract_abstracts(self) -> Tuple[List[str], List[str]]:
        """
        Extract abstracts from PDF files and preprocess them.

        Returns:
        - Tuple of updated train_data and file_paths lists.
        """

        files = os.listdir(self.input_path)

        # Split files into chunks corresponding to num_workers
        files_chunk = list(chunks(files, len(files) // self.num_workers))
        if self.verbose:
            print("Number of file chunks: ", len(files_chunk)) 

        print("Extracting abstracts from PDF files...")
        start_time = time()
        if self.parallel:
            train_data, file_paths = self.extract_abstracts_parallel(files_chunk)
        else:
            train_data, file_paths = [], []
            train_data, file_paths = self.extract_abstracts_serial(files, train_data, file_paths)
        if self.verbose:
            print(f"Time taken for extraction: {time() - start_time:.2f} seconds")
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

        if self.embedding_type == "word2vec":
            vectorizer = Word2VecVectorizer(self.embedding_dim, self.window_size, 
                                            self.sg, self.seed)
        else:
            vectorizer = TFIDFVectorizer()
        vectors, vectorizer = vectorizer.vectorize_texts(train_data)
        return vectors, vectorizer

    def run_clustering(self, train_data: list[str], vectors: np.ndarray, 
                       file_paths: List[str], n_clusters_range: range) -> None:
        """
        Executes the clustering process.

        Parameters:
        - vectors (np.ndarray): Array of input vectors for clustering.
        - file_paths (List[str]): List of file paths corresponding to each vector.

        """

        print(f"{self.clustering_method} has been chosen as clustering method...")
        optimal_clusters = None
        if self.clustering_method == "kmeans":
            clusterer = KMeansCluster(vectors, self.output_path, 
                                      self.verbose, self.init_method, self.seed)
            optimal_clusters = clusterer.silhouette_analysis()
            model, labels = clusterer.cluster_texts(optimal_clusters)
        elif self.clustering_method == "dbscan":
            clusterer = DBSCANCluster(vectors, self.output_path, 
                                      self.verbose, self.eps, self.min_samples)
            model, labels = clusterer.cluster_texts()
        elif self.clustering_method == "hierarchical":
            clusterer = HierarchicalCluster(vectors, self.output_path,
                                            self.verbose, self.linkage)
            optimal_clusters = clusterer.silhouette_analysis()
            model, labels = clusterer.cluster_texts(n_clusters=optimal_clusters)
        else:
            raise ValueError("Unsupported clustering method")

        print("Evaluating clustering results...")
        silhouette_avg, davies_bouldin_avg = self.evaluator.evaluate_clustering(vectors, labels)
        if optimal_clusters:
            print(f"\_The optimal number of clusters is: {optimal_clusters}")
        print(f"\_Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {davies_bouldin_avg}")
        save_clustering_results(labels, file_paths, self.output_path)
        self.cluster_visualizer.visualize_clusters(vectors, labels, file_paths)
        generate_summary_report(train_data, labels)
        print("Clustering process completed successfully!")

    def run(self) -> None:
        """
        Executes the full pipeline for clustering Green Energy abstracts.

        Steps:
        1. Extracts abstracts from PDF files.
        2. Vectorizes the extracted abstracts.
        3. Performs clustering on the vectorized data.

        """

        os.makedirs(self.output_path, exist_ok=True)
        train_data, file_paths = self.extract_abstracts()
        vectors, vectorizer = self.vectorize_texts(train_data)
        self.run_clustering(train_data, vectors, file_paths, self.n_clusters_range)

