import os
import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import re


# Function to split large text into smaller chunks for embedding
def custom_text_splitter(text, chunk_size=1500, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


######################## Preprocessing PDF ########################################

# Function to format tables extracted from PDFs into readable text
def format_table_as_text(table):
    if not table or len(table) <  2:  # Ensure table has enough data
        return ""
    headers = table[0]  # Assume first row contains headers
    rows = table[1:]  # Remaining rows are data
    formatted_rows = [
        ", ".join(f"{headers[i]}: {row[i]}" for i in range(len(headers)) if i < len(row))  # Combine headers and rows
        for row in rows
    ]
    return " ".join(formatted_rows)  # Join all rows into a single string


# Function to process headings and bullets in text
def process_headings_and_bullets(text):
    processed_lines = []
    for line in text.split("\n"):  # Split text into lines
        line = line.strip()
        if re.match(r"^\s*[-•]\s+", line):  # Detect bullet points
            processed_lines.append(f"Bullet: {line}")
        elif line.endswith(":"):  # Detect headings
            processed_lines.append(f"Heading: {line}")
        else:  # Normal text
            processed_lines.append(line)
    return "\n".join(processed_lines)


# Function to normalize text by fixing spacing and removing unwanted whitespace
def normalize_text(text):
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"\(\s+", "(", text)  # Remove space after '('
    text = re.sub(r"\s+\)", ")", text)  # Remove space before ')'
    text = re.sub(r"\[\s+", "[", text)  # Remove space after '['
    text = re.sub(r"\s+\]", "]", text)  # Remove space before ']'
    return text.strip()


# Function to load and preprocess all PDFs in a folder
def load_documents_from_folder(folder_path):
    all_chunks = []  # Stores text chunks
    metadata = []  # Stores metadata for each chunk
    for filename in os.listdir(folder_path):  # Iterate through all files in the folder
        if filename.endswith(".pdf"):  # Process only PDF files
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            with pdfplumber.open(file_path) as pdf:  # Open PDF
                for page_number, page in enumerate(pdf.pages, start=1):  # Process each page
                    page_text = page.extract_text() or ""  # Extract text
                    page_text = normalize_text(process_headings_and_bullets(page_text))  # Preprocess text
                    tables = page.extract_tables()  # Extract tables if present
                    if tables:
                        for table in tables:
                            formatted_table = format_table_as_text(table)  # Convert table to text
                            if formatted_table:
                                page_text += f"\n{formatted_table}"  # Append table text
                    chunks = custom_text_splitter(page_text)  # Split text into chunks
                    all_chunks.extend(chunks)  # Add chunks to the list
                    metadata.extend([{"source": filename, "page": page_number}] * len(chunks))  # Add metadata
    return all_chunks, metadata


######################## Embedding Creation ########################################

# Function to create SentenceTransformer embeddings
def create_embeddings_for_documents(documents):
    print("Creating embeddings for document chunks...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a pre-trained embedding model
    embeddings = embedding_model.encode(documents, batch_size=32, show_progress_bar=True)  # Generate embeddings
    return np.array(embeddings), embedding_model


# Function to create TF-IDF embeddings
def create_tfidf_embeddings(documents):
    print("Creating TF-IDF embeddings for document chunks...")
    vectorizer = TfidfVectorizer(stop_words='english')  # Use English stop words
    tfidf_embeddings = vectorizer.fit_transform(documents)  # Fit and transform the text
    return tfidf_embeddings, vectorizer


######################## Store Embeddings ########################################

# Function to store embeddings and metadata
def store_embeddings(folder_path, documents, metadata, sentence_embeddings, tfidf_embeddings, vectorizer):
    os.makedirs(folder_path, exist_ok=True)  # Create directory if it doesn't exist

    # Store FAISS index for SentenceTransformer embeddings
    print("Storing SentenceTransformer FAISS index...")
    faiss_index = faiss.IndexFlatL2(sentence_embeddings.shape[1])  # Initialize FAISS index
    faiss_index.add(sentence_embeddings)  # Add embeddings to the index
    faiss.write_index(faiss_index, os.path.join(folder_path, "sentence_index.faiss"))  # Save FAISS index

    # Store TF-IDF embeddings
    print("Storing TF-IDF embeddings...")
    with open(os.path.join(folder_path, "tfidf_embeddings.pkl"), "wb") as f:
        pickle.dump((tfidf_embeddings, vectorizer), f)  # Save TF-IDF embeddings and vectorizer

    # Save metadata and documents
    print("Storing metadata and documents...")
    with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)  # Save metadata
    with open(os.path.join(folder_path, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)  # Save text chunks


######################## Main Execution ########################################

def main():
    folder_path = "Knowledge_base"  # Directory containing our knowledge base PDFs
    output_folder = "embedding_store"  # Directory to store embeddings
    print("Loading documents...")
    documents, metadata = load_documents_from_folder(folder_path)  # Load and preprocess documents
    print(f"Loaded {len(documents)} chunks.")  # Show number of chunks loaded

    # Generate embeddings
    sentence_embeddings, _ = create_embeddings_for_documents(documents)  # Create sentence embeddings
    tfidf_embeddings, vectorizer = create_tfidf_embeddings(documents)  # Create TF-IDF embeddings

    # Store embeddings
    store_embeddings(output_folder, documents, metadata, sentence_embeddings, tfidf_embeddings, vectorizer)
    print(f"Embeddings stored in {output_folder}.")  # Confirm embeddings are saved


if __name__ == "__main__":
    main()  
