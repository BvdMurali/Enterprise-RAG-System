"""
Streamlit Web Application for Enterprise RAG System.
Provides a premium, responsive glassmorphism UI for document upload,
management, chat interaction, and source citation inspection.
"""

import sys
import json
import re
import textwrap
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

# Intent → friendly label and icon mapping
INTENT_META = {
    "definition":  ("📘", "Definition"),
    "summarize":   ("📝", "Summary"),
    "compare":     ("🔄", "Comparison"),
    "list":        ("📋", "List"),
    "count":       ("🔢", "Count"),
    "extract":     ("📊", "Extraction"),
    "explain":     ("💡", "Explanation"),
    "summary":     ("📋", "Profile Summary"),
    "skills":      ("🛠️", "Skills Overview"),
    "projects":    ("🚀", "Projects & Experience"),
    "education":   ("🎓", "Education"),
    "contact":     ("📇", "Contact Details"),
    "comparison":  ("⚖️", "Comparison"),
    "qa":          ("💡", "Direct Answer"),
    "general":     ("💬", "General"),
}

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
    
    /* Intent badge */
    .intent-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(99, 102, 241, 0.12);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.6rem;
    }
    
    /* Confidence pill */
    .conf-high {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.73rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .conf-medium {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(245, 158, 11, 0.12);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.73rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .conf-low {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.73rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    /* Inline citation chips */
    .citation-chip {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        color: #7DD3FC;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 5px;
        padding: 1px 6px;
        font-size: 0.72rem;
        font-weight: 500;
        vertical-align: middle;
        margin: 0 2px;
        cursor: default;
    }

    /* Document category badge */
    .doc-type-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(56, 189, 248, 0.12);
        color: #7DD3FC;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.6rem;
        margin-left: 8px;
    }
    
    /* Answer summary container */
    .summary-container {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #818CF8;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        font-style: italic;
        font-size: 0.92rem;
        color: #CBD5E1;
        border-radius: 4px;
    }

    /* Follow-up suggestion chips */
    .followup-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.9rem;
    }
    .followup-label {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    /* Source card inside expander */
    .source-card {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #6366F1;
        border-radius: 0 10px 10px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }
    .source-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.4rem;
    }
    .source-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #C7D2FE;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .source-score-high {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #A7F3D0;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .source-score-med {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #FDE68A;
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .source-score-low {
        display: inline-block;
        background: rgba(239, 68, 68, 0.1);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .source-text {
        color: #CBD5E1;
        font-size: 0.85rem;
        line-height: 1.5;
        margin: 0;
        border-top: 1px solid rgba(255,255,255,0.04);
        padding-top: 0.4rem;
        margin-top: 0.4rem;
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


# ---------------------------------------------------------------------------
# Helper functions — backend API calls
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def get_friendly_header(sources: list, default_doc_type: str = "") -> str:
    """Generate a clean, professional, and friendly document title from source filenames."""
    if not sources:
        if default_doc_type and default_doc_type.lower() != "unknown":
            clean_type = default_doc_type.replace("_", " ").replace("-", " ").title()
            return f"📄 {clean_type} Assistant"
        return "💬 Intelligent Assistant"
    
    filenames = []
    for s in sources:
        src_name = s.get("source") or s.get("filename")
        if src_name:
            name = Path(src_name).name
            if name not in filenames:
                filenames.append(name)
                
    if not filenames:
        return "💬 Intelligent Assistant"
        
    clean_names = []
    for fname in filenames:
        name_without_ext = re.sub(r"\.pdf$", "", fname, flags=re.IGNORECASE)
        clean_name = name_without_ext.replace("_", " ").replace("-", " ")
        clean_name = " ".join(clean_name.split()).title()
        if len(clean_name) > 40:
            clean_name = clean_name[:37] + "..."
        clean_names.append(clean_name)
        
    if len(clean_names) <= 2:
        return ", ".join([f"📄 {n}" for n in clean_names])
    else:
        return f"📄 Multiple Documents ({len(clean_names)})"


def highlight_matched_text(content: str, matched_text: str) -> str:
    """Robustly highlight matched child text within parent content with case-insensitive support."""
    if not matched_text or not matched_text.strip():
        return content
        
    clean_matched = matched_text.strip()
    
    # Try exact substring match
    idx = content.lower().find(clean_matched.lower())
    if idx != -1:
        start = idx
        end = idx + len(clean_matched)
        return (
            content[:start] +
            '<mark style="background-color: rgba(245, 158, 11, 0.35); color: #ffffff; padding: 2px 4px; border-radius: 4px;">' +
            content[start:end] +
            '</mark>' +
            content[end:]
        )
        
    # Try flexible whitespace match
    try:
        words = [re.escape(w) for w in clean_matched.split()]
        if words:
            pattern = r"\s+".join(words)
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start, end = match.start(), match.end()
                return (
                    content[:start] +
                    '<mark style="background-color: rgba(245, 158, 11, 0.35); color: #ffffff; padding: 2px 4px; border-radius: 4px;">' +
                    content[start:end] +
                    '</mark>' +
                    content[end:]
                )
    except Exception:
        pass
        
    # Try partial phrase matches
    phrases = [p.strip() for p in re.split(r'[,;\.\!\?]', clean_matched) if len(p.strip()) > 15]
    for phrase in phrases:
        idx = content.lower().find(phrase.lower())
        if idx != -1:
            start = idx
            end = idx + len(phrase)
            content = (
                content[:start] +
                '<mark style="background-color: rgba(245, 158, 11, 0.35); color: #ffffff; padding: 2px 4px; border-radius: 4px;">' +
                content[start:end] +
                '</mark>' +
                content[end:]
            )
            
    return content


def _confidence_pill(confidence) -> str:
    """Return an HTML confidence pill for the given level or float score."""
    try:
        val = float(confidence)
        pct = int(val * 100)
        if val >= 0.85:
            return f'<span class="conf-high">✓ High ({pct}%)</span>'
        elif val >= 0.50:
            return f'<span class="conf-medium">◑ Medium ({pct}%)</span>'
        else:
            return f'<span class="conf-low">⚠ Low ({pct}%)</span>'
    except (ValueError, TypeError):
        conf_str = str(confidence)
        if conf_str == "High":
            return f'<span class="conf-high">✓ High</span>'
        elif conf_str == "Low":
            return f'<span class="conf-low">⚠ Low</span>'
        else:
            return f'<span class="conf-medium">◑ Medium</span>'


def _score_badge(score_pct: int) -> str:
    """Return a colour-coded score badge."""
    if score_pct >= 70:
        cls = "source-score-high"
    elif score_pct >= 40:
        cls = "source-score-med"
    else:
        cls = "source-score-low"
    return f'<span class="{cls}">▲ {score_pct}% match</span>'


def render_confidence_card(confidence: float, reasons: list, placeholder=None):
    """Render a premium card showing the calculated confidence score and backing reasons list."""
    try:
        val = float(confidence)
        pct = int(val * 100)
    except (ValueError, TypeError):
        val = 0.85
        pct = 85
        
    if val >= 0.85:
        level = "High"
        cls = "conf-high"
        color = "#34D399"
        icon = "✓"
    elif val >= 0.50:
        level = "Medium"
        cls = "conf-medium"
        color = "#FCD34D"
        icon = "◑"
    else:
        level = "Low"
        cls = "conf-low"
        color = "#F87171"
        icon = "⚠"
        
    reasons_html = ""
    if reasons:
        reasons_list_html = "".join([f'<li style="margin-bottom: 2px;">{r}</li>' for r in reasons])
        reasons_html = textwrap.dedent(f"""
            <div style="margin-top: 6px; font-size: 0.82rem; color: #94A3B8;">
                <ul style="margin: 0; padding-left: 16px; list-style-type: disc;">
                    {reasons_list_html}
                </ul>
            </div>
        """).strip()
        
    html = textwrap.dedent(f"""
        <div class="glass-card" style="padding: 12px 16px; margin-bottom: 15px; border-left: 4px solid {color}; background: rgba(30, 41, 59, 0.25); box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 600; font-size: 0.88rem; color: #E2E8F0; letter-spacing: 0.02em;">CONFIDENCE ASSESSMENT</span>
                <span class="{cls}" style="margin-left: 0px; font-size: 0.78rem;">{icon} {level} ({pct}%)</span>
            </div>
            {reasons_html}
        </div>
    """).strip()
    if placeholder:
        placeholder.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_key_insights(key_insights: list, placeholder=None):
    """Render a dedicated indigo callout container for summary key insights."""
    if not key_insights:
        return
    insights_list_html = "".join([f'<li style="margin-bottom: 4px;">{ins}</li>' for ins in key_insights])
    html = textwrap.dedent(f"""
        <div class="glass-card" style="background: rgba(99, 102, 241, 0.06); border-left: 4px solid #818CF8; border-radius: 12px; padding: 14px 18px; margin: 15px 0; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);">
            <div style="font-weight: 600; font-size: 0.92rem; color: #A5B4FC; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                <span>💡</span> KEY INSIGHTS
            </div>
            <ul style="margin: 0; padding-left: 18px; color: #E2E8F0; font-size: 0.88rem; line-height: 1.5; list-style-type: square;">
                {insights_list_html}
            </ul>
        </div>
    """).strip()
    if placeholder:
        placeholder.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_sources_expander(sources: list):
    """Render the source accordion with rich source cards highlighting matching text."""
    if not sources:
        return
    meaningful = [s for s in sources if s.get("relevance_score", 0) > 0.01]
    if not meaningful:
        return

    label = f"📚 {len(meaningful)} Source{'s' if len(meaningful) != 1 else ''} Used"
    with st.expander(label, expanded=False):
        for src in meaningful:
            score_pct = min(int(src.get("relevance_score", 0) * 100), 99)
            fname = Path(src.get("source", "Unknown")).name
            page = src.get("page", "?")
            content = src.get("content", "").strip()
            matched_text = src.get("matched_child_text", "").strip()

            highlighted_content = highlight_matched_text(content, matched_text)
            
            if matched_text and len(highlighted_content) > len(content):
                display_text = highlighted_content
            else:
                display_text = content[:350] + ("…" if len(content) > 350 else "")

            html = textwrap.dedent(f"""
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-badge">📄 {fname}</span>
                        <span class="source-badge">Page {page}</span>
                        {_score_badge(score_pct)}
                    </div>
                    <p class="source-text">{display_text}</p>
                </div>
            """).strip()
            st.markdown(html, unsafe_allow_html=True)


def render_followups(followups: list):
    """Render clickable follow-up suggestion chips using st.button."""
    if not followups:
        return
    st.markdown('<div class="followup-label">💡 Suggested follow-ups</div>', unsafe_allow_html=True)
    cols = st.columns(len(followups))
    for i, q in enumerate(followups):
        with cols[i]:
            if st.button(q, key=f"followup_{id(q)}_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()


def render_assistant_message(message: dict):
    """Render a fully structured assistant message from chat history."""
    sources = message.get("sources", [])
    doc_type = message.get("document_type", "")
    confidence = message.get("confidence", 0.85)
    reasons = message.get("confidence_reasons", [])
    key_insights = message.get("key_insights", [])
    followups = message.get("followups", [])
    summary = message.get("summary", "")
    chunks_retrieved = message.get("chunks_retrieved", len(sources))
    chunks_used = message.get("chunks_used", 0)

    # 1. Friendly document header
    friendly_title = get_friendly_header(sources, doc_type)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 0.5rem;">
            <span style="font-weight: 600; font-size: 1.1rem; color: #F1F5F9; display: flex; align-items: center; gap: 6px;">
                {friendly_title}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Confidence Card
    render_confidence_card(confidence, reasons)

    # 3. 1-Sentence Quick Summary Box
    if summary and summary.strip():
        st.markdown(
            f'<div class="summary-container"><strong>Summary:</strong> {summary.strip()}</div>',
            unsafe_allow_html=True
        )

    # 4. Main answer (supports markdown)
    st.markdown(message["content"])

    # 5. Key Insights callout container
    if key_insights:
        render_key_insights(key_insights)

    # 6. Retrieval Transparency line
    if chunks_retrieved > 0:
        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 15px; margin-bottom: 6px; display: flex; align-items: center; gap: 5px;">
                <span>🔍</span> <strong>Retrieval Transparency:</strong> Chunks Retrieved: {chunks_retrieved} | Chunks Used: {chunks_used}
            </div>
            """,
            unsafe_allow_html=True
        )

    # 7. Sources accordion
    render_sources_expander(sources)

    # 8. Follow-up chips
    render_followups(followups)



# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    state = fetch_conversations()
    st.session_state.chat_history = state.get("chat_history", [])
    st.session_state.past_conversations = state.get("past_conversations", {})
    st.session_state.current_chat_title = state.get("current_chat_title", "Active Conversation")


# ---------------------------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-section-header">💬 Conversation Hub</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            if st.session_state.chat_history:
                if st.session_state.current_chat_title == "Active Conversation":
                    first_query = next((msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"), "Conversation")
                    title = first_query[:22] + "..." if len(first_query) > 22 else first_query
                    base_title = title
                    counter = 1
                    while title in st.session_state.past_conversations:
                        title = f"{base_title} ({counter})"
                        counter += 1
                    st.session_state.past_conversations[title] = list(st.session_state.chat_history)
                else:
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
                if st.button(f"💬 {key}", key=f"conv_{key}", use_container_width=True):
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
        doc_options = ["All Documents"] + [d["filename"] for d in docs]
        selected_filter = st.selectbox("Search Scope Filter", options=doc_options)
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
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


# ---------------------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------------------

st.markdown("### 💬 Chat Terminal")

# Display conversation history
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
            # Rich assistant message rendering
            render_assistant_message(message)


# ---------------------------------------------------------------------------
# Handle new query
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask a question about your uploaded documents...")
pending_query = st.session_state.pop("pending_query", None)
user_query = pending_query if pending_query else user_input

if user_query:
    # Add question to display and history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        filter_doc_name = None
        if 'selected_filter' in locals() and selected_filter != "All Documents":
            filter_doc_name = selected_filter

        # Shared mutable state bag — avoids 'nonlocal' issues inside a nested generator
        _state = {
            "answer": "",
            "sources": [],
            "followups": [],
            "confidence": 0.85,
            "intent": "general",
            "summary": "",
            "document_type": "",
            "confidence_reasons": [],
            "key_insights": [],
            "chunks_retrieved": 0,
            "chunks_used": 0,
        }

        # Placeholders in the correct layout order:
        header_placeholder = st.empty()
        confidence_placeholder = st.empty()
        summary_placeholder = st.empty()
        response_placeholder = st.empty()
        insights_placeholder = st.empty()
        transparency_placeholder = st.empty()

        def stream_generator():
            """Collect SSE stream and yield text tokens; populate _state side-channel."""
            formatted_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.chat_history[:-1]
            ]
            payload = {
                "question": user_query,
                "chat_history": formatted_history,
            }
            if filter_doc_name:
                payload["filter_document"] = filter_doc_name

            try:
                with httpx.stream("POST", f"{BACKEND_URL}/api/ask/stream", json=payload, timeout=90.0) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = json.loads(line[6:])
                        if "keepalive" in data:
                            continue  # Silent keepalive ping — ignore and keep listening
                        elif "metadata" in data:
                            _state["intent"] = data["metadata"].get("intent", "general")
                        elif "sources" in data:
                            _state["sources"].extend(data["sources"])
                        elif "token" in data:
                            yield data["token"]
                        elif "parsed" in data:
                            _state["answer"] = data["parsed"].get("answer", "")
                            _state["confidence"] = data["parsed"].get("confidence", 0.85)
                            _state["followups"] = data["parsed"].get("followups", [])
                            _state["summary"] = data["parsed"].get("summary", "")
                            _state["document_type"] = data["parsed"].get("document_type", "Unknown")
                            _state["confidence_reasons"] = data["parsed"].get("confidence_reasons", [])
                            _state["key_insights"] = data["parsed"].get("key_insights", [])
                            _state["chunks_retrieved"] = data["parsed"].get("chunks_retrieved", 0)
                            _state["chunks_used"] = data["parsed"].get("chunks_used", 0)
                        elif "error" in data:
                            yield f"\n[Error: {data['error']}]"
            except Exception as e:
                yield f"\n[Connection Error: {str(e)}]"

        try:
            full_response = response_placeholder.write_stream(stream_generator())

            # --- Strip the structural tags from the displayed text ---
            clean_response = full_response
            answer_match = re.search(r"\[ANSWER\]\s*(.*?)(?=\n\[|\Z)", full_response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                clean_response = answer_match.group(1).strip()
            
            # If the backend returned a cleaned formatting with superscripts and citations, use it!
            if _state["answer"]:
                clean_response = _state["answer"]
            
            # Re-render response in its placeholder cleanly
            response_placeholder.markdown(clean_response)

            # --- Render Header ---
            friendly_title = get_friendly_header(_state["sources"], _state["document_type"])
            header_placeholder.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 0.5rem;">
                    <span style="font-weight: 600; font-size: 1.1rem; color: #F1F5F9; display: flex; align-items: center; gap: 6px;">
                        {friendly_title}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- Render Confidence Card ---
            render_confidence_card(_state["confidence"], _state["confidence_reasons"], placeholder=confidence_placeholder)

            # --- Render Summary ---
            summary = _state["summary"]
            if summary and summary.strip():
                summary_placeholder.markdown(
                    f'<div class="summary-container"><strong>Summary:</strong> {summary.strip()}</div>',
                    unsafe_allow_html=True
                )

            # --- Render Key Insights ---
            if _state["key_insights"]:
                render_key_insights(_state["key_insights"], placeholder=insights_placeholder)

            # --- Render Retrieval Transparency ---
            retrieved = _state["chunks_retrieved"] if _state["chunks_retrieved"] > 0 else len(_state["sources"])
            used = _state["chunks_used"]
            if retrieved > 0:
                transparency_placeholder.markdown(
                    f"""
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 15px; margin-bottom: 6px; display: flex; align-items: center; gap: 5px;">
                        <span>🔍</span> <strong>Retrieval Transparency:</strong> Chunks Retrieved: {retrieved} | Chunks Used: {used}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # --- Sources accordion ---
            render_sources_expander(_state["sources"])

            # --- Follow-up suggestions ---
            render_followups(_state["followups"])

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": clean_response,
                "sources": _state["sources"],
                "followups": _state["followups"],
                "confidence": _state["confidence"],
                "intent": _state["intent"],
                "summary": _state["summary"],
                "document_type": _state["document_type"],
                "confidence_reasons": _state["confidence_reasons"],
                "key_insights": _state["key_insights"],
                "chunks_retrieved": retrieved,
                "chunks_used": used,
            })
            sync_conversations()

        except Exception as e:
            err_msg = f"An error occurred: {e}"
            st.error(err_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
            sync_conversations()
