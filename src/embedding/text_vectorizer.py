from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
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
    def __init__(self, embedding_dim: int, window_size: int) -> None:
        """
        Initialization for Word2VecVectorizer

        Parameters:
        - embedding_dim (int): Dimensionality of the embedding vectors.
        - window_size (int): Window size for Word2Vec model.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size

    def vectorize_texts(self, texts: List[str]) -> Tuple[np.ndarray, Word2Vec]:
        """
        Vectorize texts using Word2Vec model.

        Parameters:
        - texts: List of texts to vectorize.

        Returns:
        - Tuple of vectors (Word2Vec representation) and trained Word2Vec model.
        """
        tokenized_texts = [text.split() for text in texts]
        vectorizer = Word2Vec(tokenized_texts, vector_size=self.embedding_dim, window=self.window_size, min_count=1, workers=8)
        vectors = []
        for tokens in tokenized_texts:
            vector = np.mean([vectorizer.wv[token] for token in tokens if token in vectorizer.wv], axis=0)
            vectors.append(vector)
        self.vectorizer = vectorizer
        return np.array(vectors), vectorizer



# class TextVectorizer:
#     def __init__(self, embedding_type: str, embedding_dim: int, window_size: int):
#         """
#         Initialization for TextVectorizer

#         Parameters:
#         - embedding_type (str): Type of embedding to use ('tfidf' or 'word2vec').
#         - embedding_dim (int): Dimensionality of the embedding vectors.
#         - window_size (int): Window size for Word2Vec model (applicable only if embedding_type is 'word2vec').

#         """

#         self.embedding_type = embedding_type
#         self.embedding_dim = embedding_dim
#         self.window_size = window_size
#         self.vectorizer = None

#     def vectorize_texts(self, texts: List[str]) -> Tuple[np.ndarray, any]:
#         """
#         Vectorize texts using the specified embedding technique.

#         Parameters:
#         - texts: List of texts to vectorize.

#         Returns:
#         - Tuple of vectors (embedding representation) and fitted model/vectorizer.
#         """
#         if self.embedding_type == "tfidf":
#             return self._vectorize_texts_tfidf(texts)
#         elif self.embedding_type == "word2vec":
#             return self._vectorize_texts_word2vec(texts)
#         else:
#             raise ValueError(f"Unsupported embedding_type: {self.embedding_type}")

#     def _vectorize_texts_tfidf(self, texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
#         """
#         Vectorize texts using TF-IDF representation.

#         Parameters:
#         - texts: List of texts to vectorize.

#         Returns:
#         - Tuple of vectors (TF-IDF representation) and fitted TF-IDF vectorizer.
#         """
#         vectorizer = TfidfVectorizer()
#         vectors = vectorizer.fit_transform(texts)
#         self.vectorizer = vectorizer  # Save the vectorizer for future use
#         return np.array(vectors), vectorizer

#     def _vectorize_texts_word2vec(self, texts: List[str]) -> Tuple[np.ndarray, Word2Vec]:
#         """
#         Vectorize texts using Word2Vec model.

#         Parameters:
#         - texts: List of texts to vectorize.

#         Returns:
#         - Tuple of vectors (Word2Vec representation) and trained Word2Vec model.
#         """
#         tokenized_texts = [text.split() for text in texts]
#         vectorizer = Word2Vec(tokenized_texts, vector_size=self.embedding_dim, window=self.window_size, min_count=1, workers=8)
#         vectors = []
#         for tokens in tokenized_texts:
#             vector = np.mean([vectorizer.wv[token] for token in tokens if token in vectorizer.wv], axis=0)
#             vectors.append(vector)
#         self.vectorizer = vectorizer
#         return np.array(vectors), vectorizer
