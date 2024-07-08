from sklearn.metrics import silhouette_score, davies_bouldin_score

class Evaluator:
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
    
    def evaluate_silhouette_score(self):
        """
        Evaluate the silhouette score for the clustering.

        Returns:
        - Silhouette score (float).
        """
        silhouette_avg = silhouette_score(self.vectors, self.labels)
        return silhouette_avg

    def evaluate_davies_bouldin_score(self):
        """
        Evaluate the Davies-Bouldin score for the clustering.

        Returns:
        - Davies-Bouldin score (float).
        """
        davies_bouldin_avg = davies_bouldin_score(self.vectors, self.labels)
        return davies_bouldin_avg

    def evaluate_clustering(self):
        """
        Evaluate clustering performance using multiple metrics.

        Returns:
        - Tuple of silhouette score (float) and Davies-Bouldin score (float).
        """
        silhouette_avg = silhouette_score(self.vectors, self.labels)
        davies_bouldin_avg = davies_bouldin_score(self.vectors, self.labels)
        return silhouette_avg, davies_bouldin_avg

# # Example usage:
# if __name__ == "__main__":
#     # Example vectors and labels
#     vectors = ...  # Replace with actual vectors
#     labels = ...   # Replace with actual labels

#     evaluator = Evaluator(vectors, labels)
#     silhouette_score = evaluator.evaluate_silhouette_score()
#     davies_bouldin_score = evaluator.evaluate_davies_bouldin_score()

#     print(f"Silhouette Score: {silhouette_score}")
#     print(f"Davies-Bouldin Score: {davies_bouldin_score}")
