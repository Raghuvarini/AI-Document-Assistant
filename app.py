
import streamlit as st
from sentence_transformers import SentenceTransformer

from document_processor import extract, chunk_pages
from hybrid_search import HybridIndex
from llm_model import load_groq_llm_model


# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.write("Upload your documents and ask questions using AI.")


# Load embedding model
@st.cache_resource
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "history" not in st.session_state:
    st.session_state.history = []


# Document upload
st.subheader("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)


# Process documents
if st.button("Process Documents"):

    if not uploaded_files:
        st.warning("Please upload at least one document.")

    else:
        try:
            all_chunks = []

            with st.spinner("Extracting and processing documents..."):

                for uploaded_file in uploaded_files:

                    data = uploaded_file.getvalue()

                    pages = extract(uploaded_file.name, data)

                    chunks = chunk_pages(
                        uploaded_file.name,
                        pages,
                        size=700,
                        overlap=120
                    )

                    all_chunks.extend(chunks)

                encoder = load_encoder()

                st.session_state.index = HybridIndex(
                    all_chunks,
                    encoder
                )

                st.session_state.llm = load_groq_llm_model()
                st.session_state.history = []

            st.success(
                f"Successfully processed {len(all_chunks)} chunks!"
            )

        except Exception as e:
            st.error(f"Processing error: {e}")


# Question answering
st.subheader("2. Ask Questions")

if st.session_state.index is None:

    st.info("Upload and process documents before asking questions.")

else:

    for message in st.session_state.history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about your documents...")

    if question:

        st.session_state.history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("Searching documents and generating answer..."):

                    results = st.session_state.index.search(
                        question,
                        top_k=5
                    )

                    if not results:
                        answer = (
                            "I could not find relevant information "
                            "in the uploaded documents."
                        )

                    else:
                        context = "\n\n".join(
                            f"Source: {item['chunk'].source}, "
                            f"Page: {item['chunk'].page}\n"
                            f"{item['chunk'].text}"
                            for item in results
                        )

                        prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using only the context below.

If the answer is not available in the context,
clearly say that the uploaded documents do not contain
enough information.

Context:
{context}

Question:
{question}

Answer:
"""

                        answer = st.session_state.llm.invoke(prompt)

                st.markdown(answer)

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer
                })

                if results:
                    with st.expander("View retrieved document sources"):
                        for item in results:
                            chunk = item["chunk"]

                            st.write(
                                f"**File:** {chunk.source} | "
                                f"**Page:** {chunk.page}"
                            )

                            st.write(chunk.text)

                            st.caption(
                                f"Cosine similarity: "
                                f"{item['cosine']:.3f}"
                            )

            except Exception as e:
                st.error(f"Question answering error: {e}")
