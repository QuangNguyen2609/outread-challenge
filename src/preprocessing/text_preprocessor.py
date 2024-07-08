import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        """
        Initialization for TextPreprocessor

        """

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text to normalize and standardize our input.

        Parameters:
        - text: Input text.

        Returns:
        - Preprocessed text as a string.
        """
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
