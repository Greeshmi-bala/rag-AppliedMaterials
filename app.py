import sys
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ------------------------------------------------------------
# Page configuration: keep this before other Streamlit commands
# ------------------------------------------------------------

st.set_page_config(
    page_title="Applied Materials AI Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Styling using CSS only (no embedded HTML elements)
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
        color: #17324f;
    }

    .block-container {
        max-width: 100%;
        padding: 1.5rem 3rem 3rem 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2eaf3;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(2rem, 3vw, 3rem);
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.6rem;
    }

    .hero-copy {
        color: #e6f3ff;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    .hero-label {
        color: #a9e0ff;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
    }

    .section-title {
        color: #0b315b;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.8rem;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2eaf3;
        border-radius: 14px;
        padding: 1rem;
    }

    [data-testid="stChatInput"] {
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2eaf3;
        border-radius: 12px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Load the existing RAG pipeline only when needed
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_rag():
    from rag_pipeline import AppliedMaterialsRAG

    return AppliedMaterialsRAG(top_k=3)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.title("🔬 Applied Materials")
    st.caption("AI Knowledge Assistant")
    st.divider()

    st.subheader("Suggested questions")

    example_questions = [
        "What does Applied Materials do?",
        "What is semiconductor manufacturing?",
        "What engineering capabilities does Applied Materials have in India?",
    ]

    for index, example in enumerate(example_questions):
        if st.button(
            example,
            key=f"example_question_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example
            st.rerun()

    st.divider()

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()
    st.subheader("System information")
    st.write("**Language model:** Gemini 2.5 Flash")
    st.write("**Vector database:** FAISS")
    st.write("**Embedding model:** all-MiniLM-L6-v2")

    st.caption(
        "Prototype: the current knowledge base contains manually "
        "prepared sample documents. Verify facts against official "
        "sources before relying on them."
    )


# ------------------------------------------------------------
# Main page header
# ------------------------------------------------------------

with st.container():
    st.markdown(
        """
        <div style="
            background: linear-gradient(125deg, #062d59, #075ca8, #1488c8);
            padding: 2.3rem;
            border-radius: 20px;
            margin-bottom: 1.2rem;
        ">
            <div class="hero-label">Intelligent knowledge discovery</div>
            <div class="hero-title">Applied Materials AI Assistant</div>
            <div class="hero-copy">
                Ask questions, explore relevant document passages, and
                review the sources used to generate each answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

feature1, feature2, feature3 = st.columns(3)

with feature1:
    with st.container(border=True):
        st.subheader("🔎 Semantic search")
        st.write("Find document passages related to your question.")

with feature2:
    with st.container(border=True):
        st.subheader("✨ AI-generated answers")
        st.write("Use Gemini to compose answers from retrieved context.")

with feature3:
    with st.container(border=True):
        st.subheader("📚 Source references")
        st.write("Inspect the retrieved passages and source links.")


with st.expander("About this prototype"):
    st.write(
        "This application uses a local FAISS index and the existing "
        "AppliedMaterialsRAG pipeline. The current dataset contains "
        "manually prepared sample documents that have not all been "
        "independently verified against official company sources."
    )


# ------------------------------------------------------------
# Initialize RAG and display any initialization error
# ------------------------------------------------------------

rag = None

try:
    with st.spinner("Preparing the AI assistant..."):
        rag = load_rag()
    st.success("AI assistant is ready.")
except Exception as error:
    st.error("The AI assistant could not be initialized.")
    with st.expander("Technical details", expanded=True):
        st.code(str(error))
    st.info(
        "Check the GOOGLE_API_KEY in your .env file and confirm that "
        "vectorstore/company_index.faiss and "
        "vectorstore/metadata.jsonl exist."
    )


# ------------------------------------------------------------
# Display chat history
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">💬 Ask your question</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("📚 View retrieved sources"):
                    for number, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**{number}. {source.get('title', 'Untitled')}**"
                        )

                        source_url = source.get("source_url", "")
                        if source_url:
                            st.markdown(f"[Open source]({source_url})")

                        score = source.get("similarity_score")
                        if score is not None:
                            st.caption(f"Similarity score: {score:.4f}")

                        st.write(source.get("text", ""))
                        st.divider()


# ------------------------------------------------------------
# Question input and response
# ------------------------------------------------------------

typed_question = st.chat_input(
    "Ask a question about the available Applied Materials documents..."
)

pending_question = st.session_state.pending_question
st.session_state.pending_question = None

question = typed_question or pending_question

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    sources = []

    with st.chat_message("assistant"):
        if rag is None:
            answer = (
                "The AI assistant is unavailable because initialization "
                "failed. Review the technical details shown above."
            )
            st.error(answer)
        else:
            try:
                with st.spinner("Searching documents and generating an answer..."):
                    result = rag.answer(question)

                answer = result.get("answer", "No answer was returned.")
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View retrieved sources"):
                        for number, source in enumerate(sources, start=1):
                            st.markdown(
                                f"**{number}. {source.get('title', 'Untitled')}**"
                            )

                            source_url = source.get("source_url", "")
                            if source_url:
                                st.markdown(f"[Open source]({source_url})")

                            score = source.get("similarity_score")
                            if score is not None:
                                st.caption(f"Similarity score: {score:.4f}")

                            st.write(source.get("text", ""))
                            st.divider()

            except Exception as error:
                answer = (
                    "I could not process that question. Check the "
                    "technical details below and try again."
                )
                st.error(answer)
                with st.expander("Technical details", expanded=True):
                    st.code(str(error))
                print(f"RAG request failed: {error}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()
st.caption(
    "Applied Materials AI Assistant · Streamlit · FAISS · "
    "Sentence Transformers · Gemini 2.5 Flash"
)
