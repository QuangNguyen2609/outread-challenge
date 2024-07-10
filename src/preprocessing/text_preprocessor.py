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

        Steps:
        1. Convert text to lowercase.
        2. Remove all non-alphabetic characters.
        3. Tokenize the text into words.
        4. Remove stop words.
        5. Lemmatize each word.
        6. Join the words back into a single string.
        
        Parameters:
        - text: Input text.

        Returns:
        - Preprocessed text as a string.
        """

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
