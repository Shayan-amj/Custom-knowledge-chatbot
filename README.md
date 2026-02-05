# Custom-knowledge-chatbot


This project implements a **fully offline Retrieval-Augmented Generation (RAG) chatbot** that answers user questions based on a set of provided documents.

The system runs **entirely locally** using **open-source models and tools**.  
No paid APIs, no cloud-based LLMs, and no internet connection are required after setup.

---

## 📌 Project Overview

The chatbot follows a standard RAG pipeline:

1. Documents are loaded and preprocessed
2. Text is split into logical chunks
3. Embeddings are generated using an offline model
4. Embeddings are stored in a local vector database
5. User queries retrieve the most relevant chunks
6. A local LLM generates answers strictly from retrieved context

---

## 🏗️ System Architecture

The system is divided into **two main pipelines**:

---

### 1️⃣ Document Preprocessing & Indexing Pipeline

![Preprocessing Flowchart](flowcharts/preprocessing_flowchart.png)

**Steps:**
- Load PDF documents from the knowledge base
- Clean and normalize extracted text
- Detect headings, bullet points, and tables
- Split text into overlapping chunks
- Generate embeddings using SentenceTransformers
- Store embeddings locally using FAISS
- Save metadata for traceability (source & page number)

---

### 2️⃣ Query Handling & Answer Generation Pipeline

![Query Handling Flowchart](flowcharts/query_flowchart.png)

**Steps:**
- User enters a query via Streamlit UI
- Query is converted into an embedding
- Top relevant chunks are retrieved using:
  - FAISS (semantic similarity)
  - TF-IDF with cosine similarity (lexical match)
- Retrieved context is combined
- A local LLM generates the final answer using only retrieved data

---

## 🤖 Models & Tools Used

### 🔹 Embedding Model (Offline)
- **SentenceTransformers – all-MiniLM-L6-v2**
- Generates semantic embeddings locally

### 🔹 Vector Database
- **FAISS (local mode)**
- Used for fast similarity search
- TF-IDF used as a secondary retrieval method

### 🔹 Local LLM (Offline)
- **Ollama** as the local LLM runner
- Supported models:
  - Llama 3.0
  - Llama 3.1
  - Llama 3.2
  - Mistral
  - Qwen 2.5:7b

> The LLM generates responses strictly based on retrieved document context.

---

## 💬 Chat Interface

- **Streamlit-based web interface**
- Features:
  - Model selection
  - Interactive Q&A
  - Chat history
  - Fully offline execution

---

## 🛠️ Installation & Setup

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/offline-rag-chatbot.git
cd offline-rag-chatbot
```
---
### 2️⃣ Install Python Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Install & Run Ollama (Local LLM)

Download and install Ollama from:
https://ollama.com

Run a model locally:
```bash
ollama run llama3
```
### 4️⃣ Prepare Knowledge Base

Place PDF documents inside:
```
Knowledge_base/
```
5️⃣ Run Document Preprocessing
```
python Pre_Processing.py
```

This step:

1. Extracts and cleans document text

2. Creates embeddings

3. Stores everything locally in embedding_store/

### 6️⃣ Launch the Chatbot
```
streamlit run App.py
```
