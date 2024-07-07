import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import multiprocessing
from time import time
from utils import *

def main():
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
    range_n_clusters = range(2, 10)
    silhouette_avg_scores = []

    for n_clusters in range_n_clusters:
        # Initialize KMeans with the current number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(word2vec_vectors)
        # Compute the silhouette score for the current number of clusters
        silhouette_avg = silhouette_score(word2vec_vectors, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")
    # Plot the silhouette scores
    plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis For Optimal Number of Clusters')
    plt.savefig('silhouette_analysis.png')

    optimal_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
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
    visualize_clusters_pca(word2vec_vectors, labels, titles)

    # Save clustering results
    print("Saving clustering results...")
    save_clustering_results(labels, file_paths)

    # Generate summary report
    print("Generating summary report...\n")
    generate_summary_report(labels)

    print("Process completed.")
if __name__ == "__main__":
    main()