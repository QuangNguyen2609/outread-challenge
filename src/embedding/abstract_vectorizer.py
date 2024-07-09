from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class AbstractTextVectorizer(ABC):
    def __init__(self) -> None:
        """
        Initialization for AbstractTextVectorizer
        """
        self.vectorizer = None

    @abstractmethod
    def vectorize_texts(self, texts: List[str]) -> Tuple[np.ndarray, any]:
        """
        Vectorize texts using the specified embedding technique.

        Parameters:
        - texts: List of texts to vectorize.

        Returns:
        - Tuple of vectors (embedding representation) and fitted model/vectorizer.
        """
        pass
