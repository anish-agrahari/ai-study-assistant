import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import tempfile

st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("AI Powered Study Assistant")
st.markdown("Upload your PDF and ask questions, generate summaries or quizzes.")

# Sidebar
st.sidebar.header("Settings")
model_size = st.sidebar.selectbox(
    "Select Model",
    ["flan-t5-small", "flan-t5-base"]
)

chunk_size = st.sidebar.slider("Chunk Size", 300, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 300, 100)

# Cache embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Cache model
@st.cache_resource
def load_generator(model_name):
    return pipeline(
        "text2text-generation",
        model=f"google/{model_name}",
        max_new_tokens=300
    )

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        texts = text_splitter.split_documents(documents)

        embeddings = load_embeddings()

        vectorstore = FAISS.from_documents(texts, embeddings)

    st.success("PDF Processed Successfully")

    generator = load_generator(model_size)

    st.divider()

    col1, col2 = st.columns(2)

    # Summarize
    with col1:
        if st.button("Summarize PDF"):
            with st.spinner("Generating summary..."):
                sample_text = "\n\n".join([doc.page_content for doc in texts[:10]])
                prompt = f"Summarize the following content in clear points:\n{sample_text}"
                response = generator(prompt)
                st.subheader("Summary")
                st.write(response[0]["generated_text"])

    # Quiz
    with col2:
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                sample_text = "\n\n".join([doc.page_content for doc in texts[:5]])
                prompt = f"Create 5 multiple choice questions from the following content:\n{sample_text}"
                response = generator(prompt)
                st.subheader("Quiz")
                st.write(response[0]["generated_text"])

    st.divider()
    st.subheader("Ask Question from PDF")

    query = st.text_input("Enter your question")

    if query:
        with st.spinner("Searching and generating answer..."):
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
Answer the question using only the context below.
If not found, say 'Not found in document'.

Context:
{context}

Question:
{query}
"""
            response = generator(prompt)

            st.subheader("Answer")
            st.write(response[0]["generated_text"])