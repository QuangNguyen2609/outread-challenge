from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from .abstract_vectorizer import AbstractTextVectorizer
import numpy as np

class TFIDFVectorizer(AbstractTextVectorizer):
    def __init__(self) -> None:
        """
        Initialization for TFIDFVectorizer

        Parameters:
        - embedding_dim (int): Dimensionality of the embedding vectors (not used for TFIDF).
        """
        super().__init__()

    def vectorize_texts(self, texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Vectorize texts using TF-IDF representation.

        Parameters:
        - texts: List of texts to vectorize.

        Returns:
        - Tuple of vectors (TF-IDF representation) and fitted TF-IDF vectorizer.
        """
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts).toarray()
        self.vectorizer = vectorizer
        return np.array(vectors), vectorizer


class Word2VecVectorizer(AbstractTextVectorizer):
    def __init__(self, embedding_dim: int, window_size: int, sg: int, seed: int) -> None:
        """
        Initialization for Word2VecVectorizer

        Parameters:
        - embedding_dim (int): Dimensionality of the embedding vectors.
        - window_size (int): Window size for Word2Vec model.
        - sg (int): Training algorithm (0 for CBOW, 1 for Skip-gram).
        - seed (int): Seed for reproducibility.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.sg = sg
        self.seed = seed

    def vectorize_texts(self, texts: List[str]) -> Tuple[np.ndarray, Word2Vec]:
        """
        Vectorize texts using Word2Vec model.

        Parameters:
        - texts: List of texts to vectorize.

        Returns:
        - Tuple of vectors (Word2Vec representation) and trained Word2Vec model.
        """
        tokenized_texts = [text.split() for text in texts]
        vectorizer = Word2Vec(tokenized_texts, vector_size=self.embedding_dim, window=self.window_size, seed=self.seed)
        vectors = []
        for tokens in tokenized_texts:
            if len(tokens) <= 1:
                continue
            vector = np.mean([vectorizer.wv[token] for token in tokens if token in vectorizer.wv], axis=0)
            vectors.append(vector)
        self.vectorizer = vectorizer

        return np.array(vectors), vectorizer
