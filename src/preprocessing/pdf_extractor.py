import re
import spacy
from PyPDF2 import PdfReader

nlp = spacy.load("en_core_web_sm")

class PDFProcessor:
    def __init__(self, input_path: str) -> None:
        """
        Initialization for PDFProcessor

        """

        self.input_path = input_path

    def read_pdf(self) -> str:
        """
        Read text content from a PDF file.

        Returns:
        - String containing text extracted from the PDF.
        """

        reader = PdfReader(self.input_path)
        num_pages = len(reader.pages)
        text = ""
        for i, page in enumerate(reader.pages):
            if i < 3:
                text += page.extract_text() + "\n"  # Adding a newline as a separator between pages
        return text


    def extract_abstract_from_pdf_text(self, text: str) -> str:
        """
        Extract abstract from PDF text.

        Parameters:
        - text: Text extracted from a PDF.

        Returns:
        - Extracted abstract as a string.
        """

        abstract_text = ""
        # First attempt to find abstract using the presence of "Abstract" and "Introduction"
        abstract = re.findall(r"(?i)(abstract)(.*?)(introduction)", text, re.DOTALL)
        if len(abstract) > 0:
            # Assuming the first match is the desired abstract
            abstract_text = abstract[0][1].strip()
        else:
            # If no match, try another regex pattern that looks for "Abstract" followed by text, ending before a double newline or specific capitalization pattern
            abstract_match = re.search(
                                r"(?i)abstract\s*:?\s*(.*?)(?:\n\n|\n(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n)|$)", 
                                text, 
                                re.DOTALL
                            ) 
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
            else:
                abstract_text = "No abstract found."
        return abstract_text


    def filter_title_and_author(self, text: str) -> str:
        """
        Filter out sentences containing named entities (persons).

        Parameters:
        - text: Text extracted from a PDF.

        Returns:
        - Filtered text as a string.
        """

        doc = nlp(text)
        filtered_sentences = []
        for sent in doc.sents:
            if not any(ent.label_ == "PERSON" for ent in sent.ents):
                filtered_sentences.append(sent.text)
        return " ".join(filtered_sentences)