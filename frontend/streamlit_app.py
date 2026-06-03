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
    
    /* Primary Button Gradient Overlay */
    button[kind="primary"] {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(129, 140, 248, 0.4) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(129, 140, 248, 0.6) !important;
    }

    /* Chat Message Glass Panels */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #C084FC !important;
        box-shadow: 0 8px 32px rgba(192, 132, 252, 0.2) !important;
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
    
    /* Sidebar section headers */
    .sidebar-section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.4rem;
    }
    
    /* Document card in sidebar */
    .doc-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    .doc-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(30, 41, 59, 0.55);
        transform: translateX(2px);
    }
    .doc-title {
        color: #F1F5F9;
        font-size: 0.82rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .doc-meta {
        color: #94A3B8;
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }
    
    /* System status card in sidebar */
    .status-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 0.9rem;
        margin-top: 0.5rem;
    }
    .status-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        padding-bottom: 0.35rem;
    }
    .status-item:last-child {
        margin-bottom: 0;
        border-bottom: none;
        padding-bottom: 0;
    }
    .status-label {
        color: #94A3B8;
    }
    .status-value {
        color: #E2E8F0;
        font-weight: 500;
    }
    .status-pill {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-pill-ok {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-pill-err {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    
    /* Adjust buttons */
    .stButton>button {
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    
    /* Primary button styling */
    button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar specific button styling */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
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


def fetch_conversations() -> dict:
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{BACKEND_URL}/api/conversations")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {}

def sync_conversations():
    try:
        # If we are in an archived conversation, ensure its dictionary entry is kept up to date
        if st.session_state.get("current_chat_title") != "Active Conversation":
            st.session_state.past_conversations[st.session_state.current_chat_title] = list(st.session_state.chat_history)
            
        with httpx.Client(timeout=3.0) as client:
            payload = {
                "current_chat_title": st.session_state.current_chat_title,
                "chat_history": st.session_state.chat_history,
                "past_conversations": st.session_state.past_conversations
            }
            client.post(f"{BACKEND_URL}/api/conversations", json=payload)
    except Exception:
        pass


# Initialize state
if "chat_history" not in st.session_state:
    state = fetch_conversations()
    st.session_state.chat_history = state.get("chat_history", [])
    st.session_state.past_conversations = state.get("past_conversations", {})
    st.session_state.current_chat_title = state.get("current_chat_title", "Active Conversation")


# HEADER
st.markdown('<div class="hero-title">Enterprise RAG Engine</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span style="color: #94A3B8; font-size: 1.1rem; font-weight: 300; margin-right: 12px;">Production-grade Retrieval-Augmented Generation</span>
        <span style="display: inline-block; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 20px; padding: 4px 14px; color: #C7D2FE; font-size: 0.9rem; font-weight: 500; transform: translateY(-2px);">
            💬 {st.session_state.current_chat_title}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Connection Check
backend_online = check_backend_health()
if not backend_online:
    st.error(f"⚠️ Cannot connect to the backend server at {BACKEND_URL}. Please ensure the FastAPI backend is running.")
    st.stop()


# SIDEBAR: DOCUMENT HUB & CONTROLS
with st.sidebar:
    st.markdown('<div class="sidebar-section-header">💬 Conversation Hub</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            if st.session_state.chat_history:
                if st.session_state.current_chat_title == "Active Conversation":
                    # Deduce conversation name based on first user query
                    first_query = next((msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"), "Conversation")
                    title = first_query[:22] + "..." if len(first_query) > 22 else first_query
                    # Ensure name uniqueness
                    base_title = title
                    counter = 1
                    while title in st.session_state.past_conversations:
                        title = f"{base_title} ({counter})"
                        counter += 1
                    st.session_state.past_conversations[title] = list(st.session_state.chat_history)
                else:
                    # Update the currently loaded past conversation before clearing
                    st.session_state.past_conversations[st.session_state.current_chat_title] = list(st.session_state.chat_history)
                
            st.session_state.chat_history = []
            st.session_state.current_chat_title = "Active Conversation"
            sync_conversations()
            st.rerun()
            
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_chat_title = "Active Conversation"
            sync_conversations()
            st.rerun()
            
    # Past conversations listing
    if st.session_state.past_conversations:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94A3B8; margin-top: 1.5rem; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;'>📂 Saved Conversations</div>", unsafe_allow_html=True)
        for key in list(st.session_state.past_conversations.keys()):
            col_t, col_d = st.columns([5, 1])
            with col_t:
                # Highlight if current loaded chat matches
                btn_label = f"💬 {key}"
                # Render button
                if st.button(btn_label, key=f"conv_{key}", use_container_width=True):
                    # Save current active conversation if it has history before loading
                    if st.session_state.chat_history:
                        if st.session_state.current_chat_title == "Active Conversation":
                            first_q = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), "Conversation")
                            new_t = first_q[:22] + "..." if len(first_q) > 22 else first_q
                            base_t = new_t
                            c = 1
                            while new_t in st.session_state.past_conversations:
                                new_t = f"{base_t} ({c})"
                                c += 1
                            st.session_state.past_conversations[new_t] = list(st.session_state.chat_history)
                        else:
                            st.session_state.past_conversations[st.session_state.current_chat_title] = list(st.session_state.chat_history)
                        
                    # Load selected conversation
                    st.session_state.chat_history = list(st.session_state.past_conversations[key])
                    st.session_state.current_chat_title = key
                    sync_conversations()
                    st.rerun()
            with col_d:
                if st.button("❌", key=f"del_conv_{key}", help=f"Delete '{key}'"):
                    st.session_state.past_conversations.pop(key)
                    if st.session_state.current_chat_title == key:
                        st.session_state.chat_history = []
                        st.session_state.current_chat_title = "Active Conversation"
                    sync_conversations()
                    st.rerun()

    st.markdown('<div class="sidebar-section-header">📥 Document Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF documents to index", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            file_key = f"processed_{file.name}"
            if file_key not in st.session_state:
                with st.spinner(f"Indexing {file.name}..."):
                    try:
                        res = upload_document(file.getvalue(), file.name)
                        st.success(f"Indexed: {file.name} ({res['chunks_count']} chunks)")
                        st.session_state[file_key] = True
                    except Exception as e:
                        st.error(f"Failed to process {file.name}: {e}")
                        
    st.markdown('<div class="sidebar-section-header">📚 Managed Documents</div>', unsafe_allow_html=True)
    
    docs = fetch_documents()
    if not docs:
        st.info("No documents uploaded yet.")
    else:
        # Create a document dropdown filter for search
        doc_options = ["All Documents"] + [d["filename"] for d in docs]
        selected_filter = st.selectbox("Search Scope Filter", options=doc_options)
        
        # Space layout
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Display document list with delete buttons
        for d in docs:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <div class="doc-title" title="{d['filename']}">📄 {d['filename']}</div>
                        <div class="doc-meta">{d['chunk_count']} indexed chunks</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                # Add a vertical spacer to center alignment
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{d['filename']}", help=f"Delete {d['filename']}"):
                    try:
                        delete_document(d["filename"])
                        st.success(f"Deleted: {d['filename']}")
                        st.session_state.pop(f"processed_{d['filename']}", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
    st.markdown('<div class="sidebar-section-header">⚡ System Status</div>', unsafe_allow_html=True)
    
    key_pill = '<span class="status-pill status-pill-ok">Connected</span>' if settings.google_api_key != "your-gemini-api-key-here" else '<span class="status-pill status-pill-err">Missing</span>'
    health_pill = '<span class="status-pill status-pill-ok">Online</span>' if backend_online else '<span class="status-pill status-pill-err">Offline</span>'
    
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-item">
                <span class="status-label">Embedding Model</span>
                <span class="status-value" style="font-family: monospace; font-size: 0.75rem;">{settings.embedding_model_name}</span>
            </div>
            <div class="status-item">
                <span class="status-label">LLM Generator</span>
                <span class="status-value" style="font-family: monospace; font-size: 0.75rem;">{settings.llm_model_name}</span>
            </div>
            <div class="status-item">
                <span class="status-label">LLM API Key</span>
                <span class="status-value">{key_pill}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Backend Status</span>
                <span class="status-value">{health_pill}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# MAIN AREA: CHAT INTERFACE
st.markdown("### 💬 Chat Terminal")

# Display conversation logs
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                if st.session_state.get(f"editing_{idx}", False):
                    new_query = st.text_input("Edit Query", value=message["content"], key=f"edit_input_{idx}", label_visibility="collapsed")
                    col_submit, col_cancel, _ = st.columns([1.5, 1.5, 7])
                    with col_submit:
                        if st.button("Submit", key=f"submit_{idx}", type="primary"):
                            st.session_state[f"editing_{idx}"] = False
                            st.session_state.chat_history = st.session_state.chat_history[:idx]
                            st.session_state.pending_query = new_query
                            st.rerun()
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_{idx}"):
                            st.session_state[f"editing_{idx}"] = False
                            st.rerun()
                else:
                    st.markdown(message["content"])
            with col2:
                if not st.session_state.get(f"editing_{idx}", False):
                    if st.button("✏️", key=f"edit_btn_{idx}", help="Edit this query"):
                        st.session_state[f"editing_{idx}"] = True
                        st.rerun()
        else:
            st.markdown(message["content"])
            
            # If there are sources associated with the AI response, display them in an expander
            if message.get("sources"):
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
user_input = st.chat_input("Ask a question about your uploaded documents...")
pending_query = st.session_state.pop("pending_query", None)

user_query = pending_query if pending_query else user_input

if user_query:
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
                sync_conversations()
                
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
                sync_conversations()
