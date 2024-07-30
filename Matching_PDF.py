import fitz  # PyMuPDF
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\W\d]', ' ', text)  # Remove non-word characters and digits
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Paths to existing invoices
invoice_files = ['D:/aff/evi/New project/Project_PDF/document similarity/train/2024.03.15_0954.pdf', "D:/aff/evi/New project/Project_PDF/document similarity/train/2024.03.15_1145.pdf", 'D:/aff/evi/New project/Project_PDF/document similarity/train/Faller_8.PDF', 'D:/aff/evi/New project/Project_PDF/document similarity/train/invoice_77073.pdf','D:/aff/evi/New project/Project_PDF/document similarity/train/invoice_102856.pdf']

# Prepare training data
def prepare_training_data(files):
    documents = []
    for file in files:
        text = extract_text_from_pdf(file)
        processed_text = preprocess_text(text)
        documents.append(processed_text)
    return documents

# Extract and preprocess text from existing invoices
training_data = prepare_training_data(invoice_files)

# Train a TF-IDF vectorizer on the existing invoices
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(training_data)

# Function to compare a test PDF with existing invoices
def compare_test_invoice(test_invoice_file):
    # Extract and preprocess text from the test PDF
    test_invoice_text = extract_text_from_pdf(test_invoice_file)
    test_invoice_processed = preprocess_text(test_invoice_text)
    
    # Transform the test invoice into a TF-IDF vector
    test_invoice_vector = vectorizer.transform([test_invoice_processed])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(test_invoice_vector, tfidf_matrix).flatten()
    
    # Identify the most similar invoice
    most_similar_index = similarities.argmax()
    most_similar_invoice = invoice_files[most_similar_index]
    similarity_score = similarities[most_similar_index]
    
    return most_similar_invoice, similarity_score, similarities

# Test with a new invoice
test_invoice_file = input('Enter the Path of file: ')
most_similar_invoice, similarity_score, all_similarities = compare_test_invoice(test_invoice_file)

print(f"The most similar invoice to the test one is: {most_similar_invoice}")
print(f"Similarity Score: {similarity_score}")
