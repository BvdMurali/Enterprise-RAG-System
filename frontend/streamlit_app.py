"""
Streamlit Web Application for Enterprise RAG System.
Provides a premium, responsive glassmorphism UI for document upload,
management, chat interaction, and source citation inspection.
"""

import sys
from pathlib import Path
import streamlit as st
import httpx

# Add project root to path to load settings and config
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from backend.config import get_settings

settings = get_settings()
# If backend_host is set to '0.0.0.0' (listen on all interfaces), connect via local loopback
backend_host = settings.backend_host
if backend_host == "0.0.0.0":
    backend_host = "127.0.0.1"
BACKEND_URL = f"http://{backend_host}:{settings.backend_port}"

# Configure page settings
st.set_page_config(
    page_title="Enterprise RAG Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic / Dark-mode UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Top title gradient */
    .hero-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: -0.05em;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 300;
    }
    
    /* Glassmorphism Card Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    /* Custom citation elements */
    .citation-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #C7D2FE;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        margin-right: 0.5rem;
    }
    .score-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #A7F3D0;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .source-block {
        border-left: 3px solid #6366F1;
        padding-left: 1rem;
        margin-top: 0.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 0 8px 8px 0;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Clean sidebar adjustments */
    .stSidebar {
        background-color: #0F172A !important;
    }
    
    /* Adjust buttons */
    .stButton>button {
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


# Helper functions to query the backend API
def check_backend_health() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{BACKEND_URL}/api/health")
            return r.status_code == 200
    except Exception:
        return False


def upload_document(file_bytes, filename: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        files = {"file": (filename, file_bytes, "application/pdf")}
        r = client.post(f"{BACKEND_URL}/api/upload", files=files)
        r.raise_for_status()
        return r.json()


def fetch_documents() -> list:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{BACKEND_URL}/api/documents")
            r.raise_for_status()
            return r.json()
    except Exception:
        return []


def delete_document(filename: str) -> dict:
    with httpx.Client(timeout=10.0) as client:
        r = client.delete(f"{BACKEND_URL}/api/documents/{filename}")
        r.raise_for_status()
        return r.json()


def ask_rag_pipeline(question: str, filter_doc: str = None) -> dict:
    with httpx.Client(timeout=60.0) as client:
        payload = {"question": question}
        if filter_doc:
            payload["filter_document"] = filter_doc
        r = client.post(f"{BACKEND_URL}/api/ask", json=payload)
        r.raise_for_status()
        return r.json()


# Initialize state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# HEADER
st.markdown('<div class="hero-title">Enterprise RAG Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Production-grade Retrieval-Augmented Generation grounded in your documents</div>', unsafe_allow_html=True)

# Connection Check
backend_online = check_backend_health()
if not backend_online:
    st.error(f"⚠️ Cannot connect to the backend server at {BACKEND_URL}. Please ensure the FastAPI backend is running.")
    st.stop()


# SIDEBAR: DOCUMENT HUB & CONTROLS
with st.sidebar:
    st.markdown("### 📥 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents to index", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            # We check if file has already been uploaded or processed to avoid repeating
            file_key = f"processed_{file.name}"
            if file_key not in st.session_state:
                with st.spinner(f"Indexing {file.name}..."):
                    try:
                        res = upload_document(file.getvalue(), file.name)
                        st.success(f"Indexed: {file.name} ({res['chunks_count']} chunks)")
                        st.session_state[file_key] = True
                    except Exception as e:
                        st.error(f"Failed to process {file.name}: {e}")
                        
    st.markdown("---")
    st.markdown("### 📚 Managed Documents")
    
    docs = fetch_documents()
    if not docs:
        st.info("No documents uploaded yet.")
    else:
        # Create a document dropdown filter for search
        doc_options = ["All Documents"] + [d["filename"] for d in docs]
        selected_filter = st.selectbox("Search Scope Filter", options=doc_options)
        
        # Display document list with delete buttons
        for d in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**📄 {d['filename']}**  \n*{d['chunk_count']} chunks*")
            with col2:
                # Unique key for delete buttons
                if st.button("🗑️", key=f"del_{d['filename']}"):
                    try:
                        delete_document(d["filename"])
                        st.success(f"Deleted: {d['filename']}")
                        # Remove processed flags
                        st.session_state.pop(f"processed_{d['filename']}", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown("### ⚡ System Status")
    st.markdown(f"**Embedding Model:** `{settings.embedding_model_name}`")
    st.markdown(f"**LLM Generator:** `{settings.llm_model_name}`")
    if settings.google_api_key == "your-gemini-api-key-here":
        st.markdown("**LLM API Key:** ❌ *Default placeholder*")
    else:
        st.markdown("**LLM API Key:** Configured")


# MAIN AREA: CHAT INTERFACE
st.markdown("### 💬 Chat Terminal")

# Display conversation logs
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If there are sources associated with the AI response, display them in an expander
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("🔍 View Retrieved Sources"):
                for src in message["sources"]:
                    score_pct = int(src['relevance_score'] * 100)
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding: 0.8rem; margin-bottom: 0.6rem;">
                            <span class="citation-badge">📄 {src['source']}</span>
                            <span class="citation-badge">Page {src['page']}</span>
                            <span class="score-badge">Match Score: {score_pct}%</span>
                            <div class="source-block">
                                <p style="font-size: 0.9rem; margin: 0; color: #E2E8F0; line-height: 1.4;">
                                    {src['content']}
                                </p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Accept user inputs
if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    # Add question to display
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving document chunks & generating answer..."):
            try:
                # Apply filter if not "All Documents"
                filter_doc_name = None
                if 'selected_filter' in locals() and selected_filter != "All Documents":
                    filter_doc_name = selected_filter
                
                result = ask_rag_pipeline(user_query, filter_doc_name)
                
                st.markdown(result["answer"])
                
                # Add response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", [])
                })
                
                # Render sources
                if result.get("sources"):
                    with st.expander("🔍 View Retrieved Sources"):
                        for src in result["sources"]:
                            score_pct = int(src['relevance_score'] * 100)
                            st.markdown(
                                f"""
                                <div class="glass-card" style="padding: 0.8rem; margin-bottom: 0.6rem;">
                                    <span class="citation-badge">📄 {src['source']}</span>
                                    <span class="citation-badge">Page {src['page']}</span>
                                    <span class="score-badge">Match Score: {score_pct}%</span>
                                    <div class="source-block">
                                        <p style="font-size: 0.9rem; margin: 0; color: #E2E8F0; line-height: 1.4;">
                                            {src['content']}
                                        </p>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
            except Exception as e:
                err_msg = f"An error occurred: {e}"
                st.error(err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
