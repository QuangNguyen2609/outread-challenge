from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

class TextVectorizer:
    def __init__(self):
        pass

    def vectorize_texts_tfidf(self, texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)
        return vectors, vectorizer

    def vectorize_texts_word2vec(self, texts: List[str], window_size: int, embedding_dim: int) -> Tuple[np.ndarray, Word2Vec]:
        tokenized_texts = [text.split() for text in texts]
        model = Word2Vec(tokenized_texts, vector_size=embedding_dim, window=window_size, min_count=1, workers=8)
        vectors = []
        for tokens in tokenized_texts:
            vector = np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0)
            vectors.append(vector)
        return np.array(vectors), model

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

class TextVectorizer:
    def __init__(self, embedding_type: str, embedding_dim: int, window_size: int):
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.vectorizer = None

    def vectorize_texts(self, texts: List[str]) -> Tuple[np.ndarray, any]:
        """
        Vectorize texts using the specified embedding technique.

        Parameters:
        - texts: List of texts to vectorize.

        Returns:
        - Tuple of vectors (embedding representation) and fitted model/vectorizer.
        """
        if self.embedding_type == "tfidf":
            return self._vectorize_texts_tfidf(texts)
        elif self.embedding_type == "word2vec":
            return self._vectorize_texts_word2vec(texts)
        else:
            raise ValueError(f"Unsupported embedding_type: {self.embedding_type}")

    def _vectorize_texts_tfidf(self, texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Vectorize texts using TF-IDF representation.

        Parameters:
        - texts: List of texts to vectorize.

        Returns:
        - Tuple of vectors (TF-IDF representation) and fitted TF-IDF vectorizer.
        """
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)
        self.vectorizer = vectorizer  # Save the vectorizer for future use
        return np.array(vectors), vectorizer

    def _vectorize_texts_word2vec(self, texts: List[str]) -> Tuple[np.ndarray, Word2Vec]:
        """
        Vectorize texts using Word2Vec model.

        Parameters:
        - texts: List of texts to vectorize.

        Returns:
        - Tuple of vectors (Word2Vec representation) and trained Word2Vec model.
        """
        tokenized_texts = [text.split() for text in texts]
        model = Word2Vec(tokenized_texts, vector_size=self.embedding_dim, window=self.window_size, min_count=1, workers=8)
        vectors = []
        for tokens in tokenized_texts:
            vector = np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0)
            vectors.append(vector)
        self.vectorizer = model  # Save the Word2Vec model for future use
        return np.array(vectors), model
