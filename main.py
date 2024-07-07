import os
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import multiprocessing
from time import time
from utils import *

def arg_parser():
    parser = argparse.ArgumentParser(description="Clustering of Green Energy Dataset")
    parser.add_argument("--data_path", type=str, default="Green_Energy_Dataset", help="Path to the Green Energy Dataset")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save the clustering results")
    parser.add_argument("--embedding_type", type=str, default="word2vec", help="Type of embedding to use (word2vec or tfidf)")
    parser.add_argument("--embedding_dim", type=int, default=150, help="Dimension of the word embeddings")
    parser.add_argument("--window", type=int, default=30, help="Window size for Word2Vec")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum count for Word2Vec")
    return parser.parse_args()

def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    files = os.listdir("Green_Energy_Dataset")
    files_chunk = list(chunks(files, len(files) // 30))  # Split files into 30 chunks

    manager = multiprocessing.Manager()
    train_data = manager.list()
    file_paths = manager.list()

    print("Extracting abstracts from PDF files...")
    start_time = time()
    with multiprocessing.Pool(processes=30) as pool:
        pool.starmap(extract_abstract, [(chunk, train_data, file_paths) for chunk in files_chunk])
    print(f"Time taken for extraction: {time() - start_time:.2f} seconds")
    train_data = list(train_data)
    file_paths = list(file_paths)
        
    print("Number of abstracts extracted:", len(train_data), "\n")

    # Word2Vec Vectorization
    print("Vectorizing texts using Word2Vec...")
    word2vec_vectors, word2vec_model = vectorize_texts_word2vec(train_data)
    print("Word2Vec Vectors:", word2vec_vectors.shape, "\n")

    # Silhouette Analysis
    print("Performing Silhouette Analysis...")
    optimal_clusters = silhouette_analysis(word2vec_vectors, args.output_path)
    print("The optimal number of clusters =", optimal_clusters)

    # Cluster the texts
    print("Running KMeans Clustering...\n")
    kmeans, labels = cluster_texts_kmeans(word2vec_vectors, optimal_clusters)
    
    print("Evaluating clustering performance...")
    # Evaluate clustering
    silhouette_avg, davies_bouldin_avg = evaluate_clustering(word2vec_vectors, labels)
    print(f'Silhouette Score: {silhouette_avg}')
    print(f'Davies-Bouldin Index: {davies_bouldin_avg}\n')

    # Visualize clusters
    print("Visualizing clusters...")
    titles = [os.path.basename(path) for path in file_paths]
    visualize_clusters_pca(word2vec_vectors, labels, titles, args.output_path)

    # Save clustering results
    print("Saving clustering results...")
    save_clustering_results(labels, file_paths, args.output_path)

    # Generate summary report
    print("Generating summary report...\n")
    generate_summary_report(labels)

    print("Process completed.")
if __name__ == "__main__":
    args = arg_parser()
    main(args)