import gensim.downloader as api
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from PyPDF2 import PdfReader
import docx
import os

# Load word2vec model
model = api.load('word2vec-google-news-300')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import docx
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
import os

# Load Universal Sentence Encoder
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def read_word(file_path):
    # Function to read text from a Word document
    doc = docx.Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file_path):
    # Function to read text from a PDF document
    pdf_reader = PdfReader(file_path)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def calculate_sentence_similarity(sentence1, sentence2):
    # Function to calculate sentence similarity using Universal Sentence Encoder
    embeddings = model([sentence1, sentence2])
    similarity = np.inner(embeddings[0], embeddings[1])
    return similarity

def find_plagiarized_sentences(text1, text2, threshold=0.8):
    # Function to find plagiarized sentences between two texts
    plagiarized_sentences = []

    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)

    for sentence1 in sentences1:
        for sentence2 in sentences2:
            similarity = calculate_sentence_similarity(sentence1, sentence2)
            if similarity > threshold:
                plagiarized_sentences.append(sentence2)

    return plagiarized_sentences

# Example usage:
script_path = os.path.dirname(os.path.realpath(__file__))
documentA_path = os.path.join(script_path, "C:\\Users\HP\Desktop\EAPPU\RESEARCH\SOURCES\\books\\test run 3.docx")
documentB_path = os.path.join(script_path, "C:\\Users\HP\Desktop\EAPPU\RESEARCH\SOURCES\\books\\test run 1.docx")

documentA = read_word(documentA_path)
documentB = read_word(documentB_path)

# Using sentence comparison
plagiarized_sentences = find_plagiarized_sentences(documentA, documentB)

plagiarism_percentage = len(plagiarized_sentences) / len(sent_tokenize(documentB)) * 100
print(f"Plagiarism Percentage: {plagiarism_percentage:.2f}%")
6
print("\nPlagiarized Sentences:")
for sentence in plagiarized_sentences:
    print(sentence)


