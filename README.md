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

+ Running Kmeans
    ```
    python main.py --input_path Green_Energy_Dataset/ --output_path results/ --clustering_method kmeans --embedding_type word2vec --embedding_dim 150 --window_size 10 --parallel --num_worker 20 --visualize_method pca --init_method k-means++ --max_clusters 9 --seed 50 --verbose
    ```

    Parameters:

    + **--input_path**: Path to the input dataset containing the research papers in PDF format.
    + **--output_path**: Directory path where the clustering results and visualizations will be saved.
    + **--clustering_method**: Clustering algorithm to use for our pipeline.
    + **--embedding_type**: Type of embedding to use for text vectorization. Options are "word2vec" or "tfidf".
    + **--embedding_dim**: Dimension of the word embeddings.
    + **--window_size**: Window size for the Word2Vec model.
    + -**-parallel**: Enable parallel processing for faster execution.
    + **--num_worker**: Number of processes to use for parallel processing the pdf files
    + **--visualize_method**: Dimensionality reduction method to visualize clustering result. Options can include "pca", "tsne".
    + **--init_method**: Initialization method for KMeans clustering. Options can include "k-means++", "random", etc.
    + **--max_clusters**: Maximum number of clusters to evaluate for optimal clustering.
    + **--seed**: Random seed for reproducibility.
    + **--verbose**: Enable verbose output for more detailed logging.

+ Running Hierarchical clustering
    ```
    python main.py --input_path Green_Energy_Dataset/ --output_path results/ --clustering_method hierarchical --embedding_type word2vec --embedding_dim 150 --window_size 10 --parallel --num_worker 20 --visualize_method pca --linkage ward --max_clusters 9 --seed 50 --verbose
    ```

    Parameters:

    + **--linkage**: Linkage criterion for hierarchical clustering pipeline (ward, complete, average, single).

+ Running DBSCAN
    ```
    python main.py --input_path Green_Energy_Dataset/ --output_path results/ --clustering_method dbscan --embedding_type word2vec --embedding_dim 150 --window_size 10 --parallel --num_worker 20 --visualize_method pca --eps 0.5 --min_samples 5 --seed 50 --verbose
    ```
    Parameters:
    + **--eps**: Epsilon value for DBSCAN pipeline.
    + **--min_samples**: Minimum number of samples for DBSCAN pipeline.

## Output

After running the script, the following outputs will be generated in the specified output directory (results/ in this example):

+ `clustering_results.csv` : CSV file containing the clustering results, showing which papers belong to each cluster.

+ `silhouette_analysis.png` : 

+ `cluster_visualization.html` : HTML file containing a visual representation of the clustering results using PCA.

## Approach
### Data Preprocessing
  +  **PDF Text Extraction**: The script extracts abstracts from each research paper PDF using the PyPDF2 library. 

  + **Text Preprocessing**: Preprocesses the extracted text by removing stop words, lemmatizing words, and handling special characters using NLTK and spaCy libraries.

### Text Vectorization

+ **TF-IDF Vectorization**: Converts preprocessed text data into TF-IDF vectors using the TfidfVectorizer from scikit-learn. TF-IDF captures the importance of words in each document relative to the entire dataset.

+ **Word2Vec Embeddings**: Utilizes Word2Vec embeddings from the gensim library to represent text data as dense vectors. This captures semantic relationships between words and allows for measuring similarity between research paper abstracts based on their content.

### Clustering

+ **K-means Clustering**: Implements K-means clustering to group research papers into clusters based on embedding vectors. Determines the optimal number of clusters using silhouette analysis. There are two options for initialization: `random` and `k-means++`.

+ **DBSCAN Clustering**: Implements DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to group research papers into clusters based on embedding vectors. It automatically finds the number of clusters based on density and can handle noise points. Key parameters include `eps` (maximum distance between two samples to be considered as in the same neighborhood) and `min_samples` (number of samples in a neighborhood for a point to be considered as a core point).
+ **Hierarchical Clustering**: Implements hierarchical clustering to group research papers into clusters based on embedding vectors. It builds a hierarchy of clusters and allows for different `linkage` methods such as `ward`, `complete`, `average`, and `single`. The number of clusters can be determined by specifying the desired number of clusters.


### Evaluation
Evaluates clustering performance using `silhouette score` and `Davies-Bouldin Index` to measure how similar each paper is to its own cluster compared to other clusters.

### Visualization
PCA Visualization: Visualizes clustering results in a 2D space using Principal Component Analysis (PCA). Displays research paper titles in a scatter plot to provide context for each cluster.

## Experiments

### Outlier
+ When using DBSCAN, i detect this outlier that when i read text using Py2PDF from this pdf (huang-et-al-2022-bacterial-growth-induced-tobramycin-smart-release-self-healing-hydrogel-for-pseudomonas-aeruginosa.pdf), the words are sticking to each other which make it impossible to analyze and might affect the clustering result so i remove it from the extracted abstract lists.
    ```
    ABSTRACT:
    Burnsareacommonhealthproblemworldwideandare
    highlysusceptibletobacterialinfectionsthataredifficulttohandlewith
    ordinarywounddressings.Therefore,burnwoundrepairisextremely
    challenginginclinicalpractice.Herein,aseriesofself-healinghydrogels
    (QCS/OD/TOB/PPY@PDA) withgoodelectricalconductivity and
    antioxidantactivitywerepreparedonthebasisofquaternizedchitosan
    (QCS),oxidizeddextran(OD),tobramycin(TOB),andpolydopamine-
    coatedpolypyrrolenanowires(PPY@PDANWs).TheseSchiffbasecross-
    linksbetweentheaminoglycoside antibioticTOBandODenableTOBto
    beslowlyreleasedandresponsivetopH.Interestingly, theacidic
    substancesduringthebacteriagrowthprocesscaninducetheon-demand
    releaseofTOB,avoidingtheabuseofantibiotics
    ```

### Performance of different embedding method

- As clear as we can see, Word2Vec outperforms TF-IDF as an Embedding model to vectorize abstract texts. There are several reasons why this happens:
1. Semantic Similarity
    - **Word2Vec**: Captures semantic relationships between words. Words with similar meanings are placed closer together in the vector space, allowing for a more nuanced understanding of the content.
    - **TF-IDF**: Focuses on the frequency of terms, which can miss semantic relationships between words. It treats each word independently without capturing the context or meaning.
2. Contextual Understanding
    - **Word2Vec**: Takes into account the context in which words appear, allowing it to generate embeddings that capture the meaning of words in different contexts. This is particularly useful in research papers where context can significantly change the meaning of terms.
    - **TF-IDF**: Does not consider the context, only the frequency and rarity of words. This can lead to less meaningful clusters if words have multiple meanings or if important context is lost.

3. Handling Synonyms and Polysemy
    - **Word2Vec**: Can handle synonyms effectively since similar words have similar vectors. It also deals with polysemy (words with multiple meanings) by placing words in a context-specific manner.
    - **TF-IDF**: Treats each word as unique, so it cannot recognize synonyms or handle polysemy well. Words with similar meanings will be treated as completely different, leading to poorer clustering performance.

4. Small Dataset Advantage
    - **Word2Vec**: Can still generate meaningful embeddings even with a relatively small dataset, as it captures word relationships effectively. The learned embeddings generalize well to new data.
    - **TF-IDF**: May struggle with small datasets because the term frequencies and inverse document frequencies might not be stable or meaningful with limited data. This can lead to poor clustering performance.

    | Embedding method | Silhouette Score | Davies-Bouldin Index |
    |---------------------|-------------|------------------|
    | Word2Vec            | 0.56        |     0.54         |
    | TF-IDF              | 0.05        |     3.05         |

### Performance of different clustering method
+  KMeans performs well with the highest Silhouette Score, indicating well-separated and cohesive clusters. The low Davies-Bouldin Index supports this, showing compact and distinct clusters.
+ Hierarchical clustering also performs strongly, with a Silhouette Score close to KMean. It has the lowest Davies-Bouldin Index, indicating the most compact and well-separated clusters
+ DBSCAN shows lower performance with a lower Silhouette Score and higher Davies-Bouldin Index, indicating less distinct and more dispersed clusters.

    | Embedding method | Silhouette Score | Davies-Bouldin Index |
    |------------------|------------------|----------------------|
    | KMeans           | 0.59             |     0.46             |
    | Hierarchical     | 0.58             |     0.41             |
    | DBSCAN           | 0.43             |     1.08             |

### Effect of Embedding Dimension and Window Size:


+ Increasing the embedding dimension from 100 to 150 consistently improves both the silhouette score and the Davies-Bouldin index, suggesting better clustering quality.
+ Increasing the window size from 10 to 30 (with embedding dimension 100) also results in better clustering performance, with higher silhouette scores and lower Davies-Bouldin index values.

    | Embedding Dimension | Window Size | Silhouette Score | Davies-Bouldin Index |
    |---------------------|-------------|------------------|----------------------|
    | 100                 | 10          |     0.56         | 0.54                 |
    | 150                 | 10          |     0.59         | 0.46                 |
    | 100                 | 30          |     0.57         | 0.48                 |
