from PyPDF2 import PdfReader
import re 
import os
import numpy as np
import pandas as pd
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import spacy
import multiprocessing
from time import time

nlp = spacy.load("en_core_web_sm") # python -m spacy download en

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_pdf(file_path):
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    # print(f"Number of pages in {file_path} = {num_pages}")
    text = ""
    # text = reader.pages[0].extract_text()
    for i, page in enumerate(reader.pages):
        if i < 3:
            text += page.extract_text() + "\n"  # Adding a newline as a separator between pages
    return text

def find_abstract_or_first_paragraph(text):
    # Attempt to find variations of "abstract"
    abstract_match = re.findall(r"(?i)(abstract)(?:-|:)?\s*(.*?)(introduction|1\|)?", text, re.DOTALL)
    if abstract_match:
        # Return the first match's content if an abstract is found
        return abstract_match[0][1].strip()
    else:
        # If no abstract is found, return the first non-empty paragraph
        paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
        if paragraphs:
            return paragraphs[0]
        else:
            return "No abstract or content found."

def filter_title_and_author(text):
    # Use spaCy to identify named entities and filter out sentences containing them
    doc = nlp(text)
    filtered_sentences = []
    for sent in doc.sents:
        if not any(ent.label_ == "PERSON" for ent in sent.ents):
            filtered_sentences.append(sent.text)
    return " ".join(filtered_sentences)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a single string
    return ' '.join(words)

def vectorize_texts_tfidf(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

def vectorize_texts_word2vec(texts):
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(tokenized_texts, vector_size=150, window=30, min_count=1, workers=8)
    vectors = []
    for tokens in tokenized_texts:
        vector = np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0)
        vectors.append(vector)
    return np.array(vectors), model

def vectorize_texts_doc2vec(texts):
    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
    model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=8, epochs=40)
    vectors = [model.dv[str(i)] for i in range(len(texts))]
    return np.array(vectors), model

def extract_abstract_from_pdf_text(pdf_text):
    abstract_text = ""
    # First attempt to find abstract using the presence of "Abstract" and "Introduction"
    abstract = re.findall(r"(?i)(abstract)(.*?)(introduction)", pdf_text, re.DOTALL)
    if len(abstract) > 0:
        # Assuming the first match is the desired abstract
        abstract_text = abstract[0][1].strip()
    else:
        # If no match, try another regex pattern that looks for "Abstract" followed by text, ending before a double newline or specific capitalization pattern
        abstract_match = re.search(r"(?i)abstract\s*:?\s*(.*?)(?:\n\n|\n(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n)|$)", pdf_text, re.DOTALL) 
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
        else:
            abstract_text = "No abstract found."
    return abstract_text

def cluster_texts_kmeans(vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(vectors)
    labels = kmeans.labels_
    return kmeans, labels

def evaluate_clustering(vectors, labels):
    silhouette_avg = silhouette_score(vectors, labels)
    davies_bouldin_avg = davies_bouldin_score(vectors, labels)
    return silhouette_avg, davies_bouldin_avg

# def visualize_clusters(vectors, labels, titles):
#     pca = PCA(n_components=2)
#     reduced_vectors = pca.fit_transform(vectors)
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=labels, palette='viridis')
#     for i, title in enumerate(titles):
#         plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], title, fontsize=9)
#     plt.title("Clustering Visualization")
#     plt.savefig('clustering_visualization.png')

# def visualize_clusters_tsne(vectors, labels, titles):
#     tsne = TSNE(n_components=2, random_state=42)
#     reduced_vectors = tsne.fit_transform(vectors)
#     df = pd.DataFrame({
#         'x': reduced_vectors[:, 0],
#         'y': reduced_vectors[:, 1],
#         'label': labels,
#         'title': titles
#     })
#     fig = px.scatter(df, x='x', y='y', color='label', text='title',
#                      title='Visualizing Clusters in Two Dimensions Using t-SNE')
#     fig.show()
#     # save fig
#     fig.write_image("clustering_visualization_tsne.png")

def visualize_clusters_pca(vectors, labels, titles):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    df = pd.DataFrame({
        'PC1_2d': reduced_vectors[:, 0],
        'PC2_2d': reduced_vectors[:, 1],
        'label': labels,
        'title': titles
    })

    clusters = df['label'].unique()
    data = []

    colors = ['rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 2, 0.8)', 'rgba(0, 255, 200, 0.8)', 'rgba(0, 128, 255, 0.8)', 'rgba(255, 0, 0, 0.8)']

    for i, cluster in enumerate(clusters):
        cluster_data = df[df['label'] == cluster]
        trace = go.Scatter(
            x=cluster_data["PC1_2d"],
            y=cluster_data["PC2_2d"],
            mode="markers",
            name=f"Cluster {cluster}",
            marker=dict(color=colors[i % len(colors)]),
            text=cluster_data["title"]
        )
        data.append(trace)

    title = "Visualizing Clusters in Two Dimensions Using PCA"
    layout = dict(title=title,
                  xaxis=dict(title='PC1', ticklen=5, zeroline=False),
                  yaxis=dict(title='PC2', ticklen=5, zeroline=False)
                 )

    fig = dict(data=data, layout=layout)
    pyo.plot(fig, filename='clustering_visualization_pca.html')


def save_clustering_results(labels, file_paths):
    results = pd.DataFrame({'File': file_paths, 'Cluster': labels})
    results.to_csv('clustering_results.csv', index=False)

def generate_summary_report(labels):
    cluster_counts = pd.Series(labels).value_counts()
    print(f"Number of clusters: {len(cluster_counts)}")
    print("Number of papers in each cluster:")
    print(cluster_counts)
    return cluster_counts

def extract_abstract(files, train_data, file_paths):
    for file in files:
        if file.endswith(".pdf"):
            file_path = os.path.join("Green_Energy_Dataset", file)
            pdf_text = read_pdf(file_path)
            abstract_text = extract_abstract_from_pdf_text(pdf_text)
            if abstract_text == "No abstract found.":
                # continue # skip for now
                paragraphs = re.split(r'\n\s*\n', pdf_text)
                fallback_abstract = " ".join(paragraphs[:2])  # Adjust number of paragraphs as needed
                fallback_abstract = filter_title_and_author(fallback_abstract)
                abstract_text = fallback_abstract.strip()
            preprocessed_abstracts = preprocess_text(abstract_text)
            train_data.append(preprocessed_abstracts)
            file_paths.append(file_path)
    return train_data, file_paths


if __name__ == "__main__":
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
    print("\n")
    # Plot the silhouette scores
    plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis For Optimal Number of Clusters')
    plt.savefig('silhouette_analysis.png')


    print("Running KMeans Clustering...\n")
    optimal_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
    print("The optimal number of clusters = ", optimal_clusters)
    # Cluster the texts
    kmeans, labels = cluster_texts_kmeans(word2vec_vectors, optimal_clusters)
    
    print("Evaluating clustering performance...\n")
    # Evaluate clustering
    silhouette_avg, davies_bouldin_avg = evaluate_clustering(word2vec_vectors, labels)
    print(f'Silhouette Score: {silhouette_avg}')
    print(f'Davies-Bouldin Index: {davies_bouldin_avg}')

    # Visualize clusters
    print("Visualizing clusters...\n")
    titles = [os.path.basename(path) for path in file_paths]
    visualize_clusters_pca(word2vec_vectors, labels, titles)

    # Save clustering results
    print("Saving clustering results...\n")
    save_clustering_results(labels, file_paths)

    # Generate summary report
    print("Generating summary report...\n")
    generate_summary_report(labels)