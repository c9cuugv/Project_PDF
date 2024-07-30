# Invoice Similarity Comparison

This project demonstrates a simple system to automatically categorize and find the most similar invoice from a database of existing invoices. The project uses PDF text extraction and text similarity measurement techniques to identify the most similar invoice.

## Features

- **PDF Text Extraction**: Extracts text from PDF invoices using `fitz` (PyMuPDF) library.
- **Text Preprocessing**: Cleans and tokenizes text, removing punctuation, numbers, and stopwords.
- **TF-IDF Vectorization**: Converts the preprocessed text into TF-IDF vectors.
- **Cosine Similarity Calculation**: Measures the similarity between a new invoice and existing ones to find the most similar match.

## Requirements

- Python 3.x
- fitz (PyMuPDF)
- nltk
- scikit-learn

## Installation

1. Clone the repository or download the source code.
2. Install the required Python packages:

```bash
pip install pymupdf nltk scikit-learn
```

3. Download the necessary NLTK data for tokenization and stopwords:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1. **Prepare Training Data**: Place your existing PDF invoices in the same directory as the script or adjust the `invoice_files` list with paths to your PDFs.

2. **Run the Script**: Use the provided code to extract and preprocess text from the invoices, vectorize them, and then compare a new test invoice with the existing ones.

Example code snippet:

```python
# Paths to existing invoices
invoice_files = ['invoice1.pdf', 'invoice2.pdf', 'invoice3.pdf', 'invoice4.pdf']

# Extract and preprocess text from existing invoices
training_data = prepare_training_data(invoice_files)

# Train a TF-IDF vectorizer on the existing invoices
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(training_data)

# Test with a new invoice
test_invoice_file = 'test_invoice.pdf'
most_similar_invoice, similarity_score = compare_test_invoice(test_invoice_file)

print(f"The most similar invoice to the test one is: {most_similar_invoice}")
print(f"Similarity Score: {similarity_score}")
```

## Functions

- **`extract_text_from_pdf(file_path)`**: Extracts text from the given PDF file.
- **`preprocess_text(text)`**: Preprocesses the text by converting it to lowercase, removing punctuation and numbers, tokenizing, and removing stopwords.
- **`prepare_training_data(files)`**: Processes the text of all PDFs in the provided list and returns a list of preprocessed documents.
- **`compare_test_invoice(test_invoice_file)`**: Compares the text from the test PDF with the existing invoices and identifies the most similar one based on cosine similarity.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

This project uses the following libraries:

- [fitz (PyMuPDF)](https://pypi.org/project/PyMuPDF/)
- [nltk](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

---

This README provides a basic overview and instructions for setting up and running the project. You may want to customize it further based on additional features, specific configurations, or particular instructions relevant to your project.