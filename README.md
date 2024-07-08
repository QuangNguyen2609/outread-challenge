# outread-challenge

## Overview
This project aims to cluster a set of research papers based on the similarity of their abstracts using natural language processing (NLP) techniques and unsupervised machine learning algorithms. The clustering is performed to group together research papers that share common themes or topics, facilitating easier exploration and understanding of large collections of academic literature.

## Requirements
- Python 3.x (recommended version: 3.10+)
- Libraries: PyPDF2, nltk, gensim, scikit-learn, spacy, pandas, numpy, matplotlib, plotly

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/QuangNguyen2609/outread-challenge.git
   cd outread-challenge
2. Create conda env & Install the required Python libraries:
    ```bash
    conda create -n outread python=3.10
    conda activate outread
    pip install -r requirements.txt
3. Download NLTK and spacy resources:

    ```bash
    python -m nltk.downloader stopwords punkt wordnet
    python -m spacy download en
## Project Structure
The project is structured to maintain clarity and organization:
 + main.py: Main script to run the pipeline
 + src: Contains all source code files.
    + utils.py: Utility functions used across modules.
    + pipeline.py: Orchestrates the Green Energy Clustering pipeline.
    + Functional Modules
        + cluster_analyzer.py: Implements clustering algorithms and analysis.
        + text_vectorizer.py: Provides methods for text vectorization.
        + evaluator.py:  Evaluates clustering performance.
        + pdf_extractor.py: Extracts text and abstracts from PDF files.
        + text_preprocessor.py: Cleans and preprocesses text data.
        + cluster_visualizer.py: Visualizes clustering results.

    ```
    ├── README.md
    ├── utils.py
    ├── main.py
    ├── src
    │   ├── utils.py
    │   ├── pipeline.py
    │   ├── clustering
    │   │   └── cluster_analyzer.py
    │   ├── embedding
    │   │   └── text_vectorizer.py
    │   ├── evaluation
    │   │   └── evaluator.py
    │   ├── preprocessing
    │   │   ├── pdf_extractor.py
    │   │   └── text_preprocessor.py
    │   └── visualization
    │       └── cluster_visualizer.py
## Usage
### Running the Clustering Script
To run the research paper clustering script, execute the following command from the project root directory:
```
python main.py --input_path Green_Energy_Dataset/ --output_path results/ --embedding_type word2vec --embedding_dim 150 --window_size 30 --parallel --num_worker 20 --visualize_method pca --init_method k-means++ --max_clusters 9 --seed 50 --verbose
```
Parameters:

+ **--input_path**: Path to the input dataset containing the research papers in PDF format.
+ **--output_path**: Directory path where the clustering results and visualizations will be saved.
+ **--embedding_type**: Type of embedding to use for text vectorization. Options are "word2vec" or "tfidf".
+ **--embedding_dim**: Dimension of the word embeddings.
+ **--window_size**: Window size for the Word2Vec model.
+ -**-parallel**: Enable parallel processing for faster execution.
+ **--num_worker**: Number of processes to use for parallel processing the pdf files
+ **--visualize_method**: Dimensionality reduction method to visualize clustering result. Options can include "pca", "tsne", etc.
+ **--init_method**: Initialization method for KMeans clustering. Options can include "k-means++", "random", etc.
+ **--max_clusters**: Maximum number of clusters to evaluate for optimal clustering.
+ **--seed**: Random seed for reproducibility.
+ **--verbose**: Enable verbose output for more detailed logging.
## Output

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

### Clustering
**K-means Clustering**: Implements K-means clustering to group research papers into clusters based on Embedding vectors. Determines the optimal number of clusters using silhouette analysis. There are two options for initialization are `random` and `k-means++`.

### Evaluation
Evaluates clustering performance using `silhouette score` and `Davies-Bouldin Index` to measure how similar each paper is to its own cluster compared to other clusters.

### Visualization
PCA Visualization: Visualizes clustering results in a 2D space using Principal Component Analysis (PCA). Displays research paper titles in a scatter plot to provide context for each cluster.

