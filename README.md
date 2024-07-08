# outread-challenge

## Overview
This project aims to cluster a set of research papers based on the similarity of their abstracts using natural language processing (NLP) techniques and unsupervised machine learning algorithms. The clustering is performed to group together research papers that share common themes or topics, facilitating easier exploration and understanding of large collections of academic literature.

## Requirements
- Python 3.x (recommended version: 3.7+)
- Libraries: PyPDF2, nltk, gensim, scikit-learn, spacy, pandas, numpy, matplotlib, plotly

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/QuangNguyen2609/outread-challenge.git
   cd outread-challenge
2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
3. Download NLTK and spacy resources:

    ```bash
    python -m nltk.downloader stopwords punkt wordnet
    python -m spacy download en
## Usage
### Running the Clustering Script
To run the research paper clustering script, execute the following command from the project root directory:
```
python main.py --input_path dataset/ --output_path results/ --embedding_type word2vec --embedding_dim 150 --window_size 30 --parallel --num_worker 30 --visualize_method pca
```
Parameters:

+ **--input_path**: Path to the input dataset containing the research papers in PDF format.
+ **--output_path**: Directory path where the clustering results and visualizations will be saved.
+ **--embedding_type**: Type of embedding to use for text vectorization. Options are "word2vec" or "tfidf".
+ **--embedding_dim**: Dimension of the word embeddings.
+ **--window_size**: Window size for the Word2Vec model.
+ -**-parallel**: Enable parallel processing for faster execution.
+ **--num_worker**: Number of processes to use for parallel processing the pdf files
+ **--visualize_method**: Dimensionality reduction method to visualize clustering result
### Output

After running the script, the following outputs will be generated in the specified output directory (results/ in this example):

+ `clustering_results.csv` : CSV file containing the clustering results, showing which papers belong to each cluster.

+ `summary_report.txt` : Text file summarizing the clustering results, including the number of clusters, number of papers in each cluster, and key terms/topics associated with each cluster.

+ `cluster_visualization.html` : HTML file containing a visual representation of the clustering results using PCA.

## Approach
### Data Preprocessing
  +  **PDF Text Extraction**: The script extracts abstracts from each research paper PDF using the PyPDF2 library. 

  + **Text Preprocessing**: Preprocesses the extracted text by removing stop words, lemmatizing words, and handling special characters using NLTK and spaCy libraries.

### Text Vectorization

+ **TF-IDF Vectorization**: Converts preprocessed text data into TF-IDF vectors using the TfidfVectorizer from scikit-learn. TF-IDF captures the importance of words in each document relative to the entire dataset.

+ **Word2Vec Embeddings**: Utilizes Word2Vec embeddings from the gensim library to represent text data as dense vectors. This captures semantic relationships between words and allows for measuring similarity between research paper abstracts based on their content.

## Clustering
**K-means Clustering**: Implements K-means clustering to group research papers into clusters based on Embedding vectors. Determines the optimal number of clusters using silhouette analysis. There are two options for initialization are `random` and `k-means++`.

## Evaluation
Evaluates clustering performance using `silhouette score` and `Davies-Bouldin Index` to measure how similar each paper is to its own cluster compared to other clusters.

## Visualization
PCA Visualization: Visualizes clustering results in a 2D space using Principal Component Analysis (PCA). Displays research paper titles in a scatter plot to provide context for each cluster.

## Files Included
+ `main.py`: Main Python script for clustering research papers.
+ `utils.py`: Utility functions for running main algorithm
+ `requirements.txt` : List of Python dependencies required to run the project.
+ `README.md`: This file, providing an overview, usage instructions, and explanation of the project.
