import argparse
from src import GreenEnergyClustering

def arg_parser():
    parser = argparse.ArgumentParser(description="Clustering of Green Energy Dataset")
    parser.add_argument("--input_path", type=str, default="Green_Energy_Dataset", help="Path to the Green Energy Dataset")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save the clustering results")
    parser.add_argument("--embedding_type", type=str, default="word2vec", help="Type of embedding to use (word2vec or tfidf)")
    parser.add_argument("--embedding_dim", type=int, default=150, help="Dimension of the word embeddings")
    parser.add_argument("--window_size", type=int, default=30, help="Window size for Word2Vec")
    parser.add_argument("--parallel", action='store_true',  help="Enable parallel processing (default: False)")
    parse.add_argument("--num_worker", type=int, default=1, help="Number of processes for")
    parse.add_argument("--visualize_method", type=str, default='pca', help='Dimensionality reduction method for clustering visualzation')
    return parser.parse_args()

# def main(args):
#     os.makedirs(args.output_path, exist_ok=True)

#     files = os.listdir(args.input_path)
    
#     # Split files into chunks that corresponding to num_worker
#     files_chunk = list(chunks(files, len(files) // args.num_worker)) 

#     print("Extracting abstracts from PDF files...")
#     start_time = time()
#     if args.parallel:
#         manager = multiprocessing.Manager()
#         train_data = manager.list()
#         file_paths = manager.list()
#         with multiprocessing.Pool(processes=30) as pool:
#             pool.starmap(extract_abstract, [(chunk, train_data, file_paths) for chunk in files_chunk])
#     else:
#         train_data = []
#         file_paths = []
#         train_data, file_paths = extract_abstract(files, train_data, file_paths)

#     print(f"Time taken for extraction: {time() - start_time:.2f} seconds")
#     train_data = list(train_data)
#     file_paths = list(file_paths)
        
#     print("Number of abstracts extracted:", len(train_data), "\n")

#     # Word2Vec Vectorization
#     if args.embedding_type == "word2vec"
#         print("Vectorizing texts using Word2Vec...")
#         embedding_vectors, embedding_model = vectorize_texts_word2vec(train_data, args.window_size, args.embedding_dim)
#         print("Word2Vec Vectors:", embedding_vectors.shape, "\n")
#     else:
#         print("Vectorizing texts using TF-IDF...")
#         embedding_vectors, embedding_model = vectorize_texts_tfidf(train_data)
#         print("TF-IDF Vectors:", embedding_vectors.shape, "\n")

#     # Silhouette Analysis
#     print("Performing Silhouette Analysis...")
#     optimal_clusters = silhouette_analysis(embedding_vectors, args.output_path)
#     print("The optimal number of clusters =", optimal_clusters)

#     # Cluster the texts
#     print("Running KMeans Clustering...\n")
#     kmeans, labels = cluster_texts_kmeans(embedding_vectors, optimal_clusters)
    
#     print("Evaluating clustering performance...")
#     # Evaluate clustering
#     silhouette_avg, davies_bouldin_avg = evaluate_clustering(embedding_vectors, labels)
#     print(f'Silhouette Score: {silhouette_avg}')
#     print(f'Davies-Bouldin Index: {davies_bouldin_avg}\n')

#     # Visualize clusters
#     print("Visualizing clusters...")
#     titles = [os.path.basename(path) for path in file_paths]
#     visualize_clusters(embedding_vectors, labels, titles, args.output_path, args.visualize_method)

#     # Save clustering results
#     print("Saving clustering results...")
#     save_clustering_results(labels, file_paths, args.output_path)

#     # Generate summary report
#     print("Generating summary report...\n")
#     generate_summary_report(labels)

#     print("Process completed.")

def main(args):
    clustering = GreenEnergyClustering(
        input_path=args.input_path,
        output_path=args.output_path,
        embedding_type=args.embedding_type,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        parallel=args.parallel,
        num_workers=args.num_workers
    )
    clustering.run()

if __name__ == "__main__":
    args = arg_parser()
    main(args)