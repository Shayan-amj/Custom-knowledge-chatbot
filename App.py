import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import base64
from PIL import Image
import os


# Function to set a custom background image for the Streamlit app
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .answer-section, .history-section {{
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            font-size: 20px;
            line-height: 1.6;
        }}
        .stSidebar {{
            background-color: rgba(0, 0, 0, 0.8);
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Function to load pre-saved embeddings, documents, and metadata from files
def load_embeddings(folder_path):
    sentence_index = faiss.read_index(f"{folder_path}/sentence_index.faiss")  # Load FAISS index
    with open(f"{folder_path}/tfidf_embeddings.pkl", "rb") as f:
        tfidf_embeddings, vectorizer = pickle.load(f)  # Load TF-IDF embeddings and vectorizer
    with open(f"{folder_path}/documents.pkl", "rb") as f:
        documents = pickle.load(f)  # Load document chunks
    with open(f"{folder_path}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)  # Load metadata for chunks
    return sentence_index, tfidf_embeddings, vectorizer, documents, metadata


# Function to search for relevant chunks using FAISS
def search_in_faiss(query, faiss_index, embedding_model, documents, metadata, k=3):
    query_embedding = embedding_model.encode([query]).reshape(1, -1)  # Encode the query
    distances, indices = faiss_index.search(query_embedding, k)  # Perform nearest-neighbor search
    results = [
        {
            "chunk": documents[idx],  # Retrieve the corresponding text chunk
            "metadata": metadata[idx],  # Retrieve metadata (source and page)
            "distance": distances[0][i],  # Record the distance score
        }
        for i, idx in enumerate(indices[0])
    ]
    return results


# Function to search for relevant chunks using TF-IDF
def search_in_tfidf(query, tfidf_embeddings, vectorizer, documents, metadata, k=3):
    query_vector = vectorizer.transform([query])  # Convert query to a TF-IDF vector
    similarities = cosine_similarity(query_vector, tfidf_embeddings).flatten()  # Compute cosine similarities
    top_indices = similarities.argsort()[-k:][::-1]  # Get top-k similar chunks
    return [
        {
            "chunk": documents[i],  # Retrieve the corresponding text chunk
            "metadata": metadata[i],  # Retrieve metadata (source and page)
            "similarity": similarities[i],  # Record the similarity score
        }
        for i in top_indices
    ]


# Function to generate an answer using the LLM
def generate_answer(context, question, model_name):
    template = """
    You are a helpful NUST student policy  assistant. You will get two separate chunks of data search both and find answer. 
    Use the provided context to answer the question below. Be precise, concise and to the point. Only answer what is asked no extra info.  
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)  # Create a prompt template
    model = OllamaLLM(model=model_name, base_url="http://localhost:11434")  # Load the LLM
    chain = prompt | model  # Combine the prompt and model into a chain
    result = chain.invoke({"context": context, "question": question})  # Generate an answer
    return result.strip()


def main():
    st.set_page_config(page_title="Custom Knowledge base Chatbot", layout="wide")  # Configure Streamlit app layout
    set_background("nust.jpg")  # Set background image

     # Load the NUST logo
    nust_logo = Image.open("National-University-of-Science-and-Technology-logo.png")

    # Encode the image in base64
    with open("nust_logo_temp.png", "wb") as image_file:  # Save temporarily
        nust_logo.save(image_file, format="PNG")
    with open("nust_logo_temp.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Clean up the temporary file (optional)
    os.remove("nust_logo_temp.png")

    # Create a container with flexbox layout and light background
    container = st.container()
    container.markdown(
        """
        <div style="display: flex; align-items: center;">
            <div style="background-color: white; padding: 10px; border-radius: 50%;">
                <img src="data:image/png;base64,{}" width="100">
            </div>
            <h1 style="margin-left: 10px;">NUST Student Policy Assistant</h1>
        </div>
        """.format(encoded_image),
        unsafe_allow_html=True
    )

    st.markdown(
    """
    <p style="font-size: 24px;">Welcome to the NUST chatbot powered by AI. Ask me anything related to NUST policies.</p>
    """,
    unsafe_allow_html=True
    )
    
    if "history" not in st.session_state:  # Initialize conversation history
        st.session_state.history = []

    # Sidebar for model selection and user query input
    model_name = st.sidebar.selectbox(
        "Select an LLM model:",
        [ "llama3.2", "qwen2.5:7b", "mistral", "llama3.1", "llama3"]  # Model options
    )
    user_input = st.sidebar.text_area("Enter your query:", placeholder="Type your question here...", key="input_query")

    # Check if the query is submitted
    if st.session_state.get("query_triggered", False):
        st.sidebar.session_state.query_triggered = False
        query_submitted = True
    else:
        query_submitted = st.sidebar.button("Submit")

    # Process user query
    if query_submitted and user_input.strip():
        folder_path = "embedding_store"  # Folder containing embeddings
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the embedding model
        sentence_index, tfidf_embeddings, vectorizer, documents, metadata = load_embeddings(folder_path)  # Load data

        with st.spinner("Searching for answers..."):  # Show a spinner while searching
            faiss_results = search_in_faiss(user_input, sentence_index, embedding_model, documents, metadata, k=3)  # FAISS
            tfidf_results = search_in_tfidf(user_input, tfidf_embeddings, vectorizer, documents, metadata, k=3)  # TF-IDF

            # Combine context from FAISS and TF-IDF results
            combined_context = "\n".join(
                [
                    f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}\n{result['chunk']}"
                    for result in faiss_results
                ]
            )
            combined_context += "\n" + "\n".join(
                [
                    f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}\n{result['chunk']}"
                    for result in tfidf_results
                ]
            )

            # Generate an answer using the LLM
            answer = generate_answer(combined_context, user_input, model_name)

        # Save the query and answer in history
        st.session_state.history.append({"question": user_input, "answer": answer})
        st.sidebar.success("Answer generated!")

    # Display the most recent answer
    st.subheader("Answer")
    if st.session_state.history:
        latest_entry = st.session_state.history[-1]
        with st.container():
            st.markdown(
                f'<div class="answer-section">'
                f'{latest_entry["answer"]}</div>',
                unsafe_allow_html=True,
            )

    # Display the history of questions and answers
    st.subheader("History")
    for i, entry in enumerate(reversed(st.session_state.history[:-1]), start=1):
        with st.container():
            st.markdown(
                f'<div class="history-section"><strong>Question:</strong> {entry["question"]}<br>'
                f'<strong>Answer:</strong> {entry["answer"]}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()  # Run the Streamlit app
