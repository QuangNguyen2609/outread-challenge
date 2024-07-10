import argparse
from src import Pipeline

def arg_parser():
    parser = argparse.ArgumentParser(description="Clustering of Green Energy Dataset")
    parser.add_argument("--input_path", type=str, default="Green_Energy_Dataset",
                         help="Path to the Green Energy Dataset")
    parser.add_argument("--output_path", type=str, default="output",
                         help="Path to save the pipeline results")
    parser.add_argument("--embedding_type", type=str, default="word2vec", 
                        help="Type of embedding to use (word2vec or tfidf)")
    parser.add_argument("--embedding_dim", type=int, default=150, 
                        help="Dimension of the word embeddings")
    parser.add_argument("--window_size", type=int, default=30,
                         help="Window size for Word2Vec")
    parser.add_argument("--parallel", action='store_true', 
                         help="Enable parallel processing (default: False)")
    parser.add_argument("--num_worker", type=int, default=1,
                         help="Number of processes for")
    parser.add_argument("--visualize_method", type=str, default='pca', 
                        help='Dimensionality reduction method for pipeline visualzation')
    parser.add_argument("--init_method", type=str, default='k-means++',
                         help='Method to initialize centroids for KMeans pipeline')
    parser.add_argument("--max_clusters", type=int, default=10,
                         help='Maximum number of clusters to evaluate')
    parser.add_argument("--seed", type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument("--verbose", action='store_true', 
                        help='Print verbose output during the process')
    parser.add_argument("--clustering_method", type=str, default='kmeans', 
                        help="Clustering method to use (e.g., 'kmeans', 'dbscan', 'hierarchical')")
    parser.add_argument("--eps", type=float, default=0.5,
                         help="Epsilon value for DBSCAN pipeline")
    parser.add_argument("--min_samples", type=int, default=5,
                         help="Minimum number of samples for DBSCAN pipeline")
    parser.add_argument("--linkage", type=str, default='ward',
                         help="Linkage criterion for hierarchical pipeline (e.g., 'ward', 'complete', 'average', 'single')")
    parser.add_argument("--sg", type=int, default=0, 
                        help="Training algorithm for Word2Vec (0 for CBOW, 1 for Skip-gram)")

    return parser.parse_args()

def main(args):
    pipeline = Pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        embedding_type=args.embedding_type,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        parallel=args.parallel,
        num_workers=args.num_worker,
        visualize_method=args.visualize_method,
        init_method=args.init_method,
        max_clusters=args.max_clusters,
        seed=args.seed,
        verbose=args.verbose,
        clustering_method=args.clustering_method,
        eps=args.eps,
        min_samples=args.min_samples,
        linkage=args.linkage,
        sg=args.sg
    )
    pipeline.run()

if __name__ == "__main__":
    args = arg_parser()
    main(args)