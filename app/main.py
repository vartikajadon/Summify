import os
import sys
from pathlib import Path

# Force disable mock mode for the live application
os.environ.pop("LUCIDOC_MOCK_LLM", None)

# Add project root directory to python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import streamlit as st
import yaml

from session.manager import create_session, delete_session, export_all, get_session_paths
from graph.build_graph import build_graph, LucidocState
from pipeline.structure import StructuredDocument
from pipeline.chat import answer_question
from pipeline.faithfulness import verify_and_refine_answer

# Load Configuration
def load_config() -> dict:
    config_path = project_root / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {"provider": "groq", "model": "gpt-oss-safeguard-20b"}

config = load_config()

# Page Setup & Modern Design Injection
st.set_page_config(
    page_title="Summify — AI Documentation Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Minimalist Stripe/Linear Light Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A !important;
    }
    
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Hide Standard Streamlit Chrome Header & Footer */
    header, [data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Content Padding & Container Constraints */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #0F172A;
    }

    .sidebar-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }

    .sidebar-card-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 8px;
    }

    .badge-pill {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 4px;
    }

    .status-active {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Floating Hero Card */
    .hero-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 8px;
    }

    .hero-title .gradient-text {
        background: linear-gradient(135deg, #2563EB 0%, #9333EA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #475569;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Minimalist Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.15s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background-color: #F1F5F9 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        background-color: #EFF6FF !important;
        border-bottom: 2px solid #2563EB !important;
    }

    .stTabs [aria-selected="true"] p, 
    .stTabs [aria-selected="true"] span {
        color: #2563EB !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Secondary Neutral Buttons */
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease !important;
    }

    .stButton>button:hover {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-color: #94A3B8 !important;
    }

    .stButton>button[type="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 2px 4px 0 rgba(37, 99, 235, 0.2) !important;
    }

    .stButton>button[type="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 4px 12px 0 rgba(37, 99, 235, 0.3) !important;
    }

    /* Dashed Border Callouts */
    .dashed-callout, [data-testid="stAlert"] {
        background-color: #F1F5F9 !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        color: #1E293B !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
        color: #1E293B !important;
    }

    /* Chat Message Cards */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03) !important;
    }

    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {
        color: #0F172A !important;
    }

    /* File Uploader Light Theme Styling */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzoneInstructions"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F8FAFC !important;
        border-color: #2563EB !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploaderDropzoneInstructions"] p,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: #334155 !important;
        font-weight: 500 !important;
    }

    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] button {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #E2E8F0 !important;
        border-color: #94A3B8 !important;
    }

    [data-testid="stFileUploaderFile"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }

    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] div,
    [data-testid="stFileUploaderFile"] small {
        color: #0F172A !important;
    }

    /* Sticky Bottom Container & Chat Input Light Theme Styling */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background-color: #F8FAFC !important;
        background: #F8FAFC !important;
        border-top: 1px solid #E2E8F0 !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="input"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    .stChatInput,
    .stChatInput > div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.04) !important;
        color: #0F172A !important;
    }

    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInputContainer"]:focus-within,
    [data-testid="stChatInput"] [data-baseweb="input"]:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInputTextArea"],
    .stChatInput textarea {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        font-size: 0.95rem !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder,
    .stChatInput textarea::placeholder {
        color: #94A3B8 !important;
    }

    [data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] button,
    .stChatInput button {
        color: #2563EB !important;
        background-color: #EFF6FF !important;
        border-radius: 8px !important;
        border: 1px solid #DBEAFE !important;
    }

    [data-testid="stChatInputSubmitButton"]:hover,
    [data-testid="stChatInput"] button:hover,
    .stChatInput button:hover {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }

    [data-testid="stChatInputSubmitButton"] svg,
    [data-testid="stChatInput"] button svg {
        fill: currentColor !important;
        color: currentColor !important;
    }

    /* General Form Controls & Headers */
    h1, h2, h3, h4, h5, h6, label, p, span {
        color: #0F172A !important;
    }
    
    input[type="text"], textarea, [data-baseweb="input"], [data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()

if "structured_doc" not in st.session_state:
    st.session_state.structured_doc = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

session_id = st.session_state.session_id
paths = get_session_paths(session_id)

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/2563eb/document.png", width=50)
    st.markdown('<div style="font-size: 1.3rem; font-weight: 800; color: #0F172A; margin-top: 4px; margin-bottom: 16px;">Summify Control</div>', unsafe_allow_html=True)
    
    provider_name = config.get("provider", "groq").upper()
    model_name = config.get("model", "openai/gpt-oss-20b")

    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Session Overview</div>
        <div style="font-size: 0.88rem; font-weight: 600; color: #0F172A; margin-bottom: 8px;">
            Session ID: <code style="background: #F1F5F9; color: #0F172A; padding: 2px 6px; border-radius: 4px;">{session_id[:8]}...</code>
        </div>
        <div style="margin-bottom: 8px;">
            <span class="status-active">● Active</span>
        </div>
        <div>
            <span class="badge-pill">Provider: {provider_name} ({model_name})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    st.subheader("Session Management")
    if st.button("🗑️ Delete My Session", use_container_width=True):
        st.session_state.confirm_delete = True

    if st.session_state.get("confirm_delete", False):
        st.warning("⚠️ Delete all uploaded files, generated docs, and vector indexes for this session?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, Delete", key="btn_confirm_delete"):
                delete_session(session_id)
                # Create fresh session
                st.session_state.session_id = create_session()
                st.session_state.structured_doc = None
                st.session_state.exports = None
                st.session_state.chat_messages = []
                st.session_state.confirm_delete = False
                st.success("Session deleted completely!")
                st.rerun()
        with col_no:
            if st.button("Cancel", key="btn_cancel_delete"):
                st.session_state.confirm_delete = False
                st.rerun()

# Main Application Body
st.markdown("""
<div class="hero-card">
    <div class="hero-title">Summify <span class="gradient-text">AI Documentation Engine</span></div>
    <div class="hero-subtitle">
        Transform messy, multi-file tech notes into beautifully organized, deduplicated documentation with trustworthy RAG Q&A.
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_upload, tab_doc, tab_chat = st.tabs([
    "📥 1. Document Ingestion",
    "📚 2. Structured Document & Exports",
    "💬 3. Trustworthy RAG Chat"
])

# Tab 1: File Upload & Processing
with tab_upload:
    st.subheader("Upload Source Documentation Files")
    st.markdown("Upload multiple `.md`, `.txt`, `.docx`, or text-extractable `.pdf` files.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["md", "txt", "docx", "pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected for processing.")
        
        if st.button("🚀 Generate Structured Document", type="primary"):
            # Save uploaded files into session uploads directory
            saved_paths = []
            for uf in uploaded_files:
                save_path = paths["uploads_dir"] / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_paths.append(str(save_path))

            # Run LangGraph Pipeline with visual progress stages
            progress_text = st.empty()
            progress_bar = st.progress(0)

            graph = build_graph()
            state_input: LucidocState = {
                "uploaded_files": saved_paths,
                "session_id": session_id,
                "chat_history": []
            }

            with st.spinner("Processing document pipeline..."):
                progress_text.markdown("🔄 **Stage 1/4:** Ingesting & Normalizing Text...")
                progress_bar.progress(25)

                result_state = graph.invoke(state_input)
                progress_bar.progress(100)
                progress_text.markdown("✅ **Pipeline Execution Complete!**")

            # Store generated document & pre-computed exports in session state
            raw_doc_dict = result_state.get("structured_document", {})
            st.session_state.structured_doc = raw_doc_dict
            st.session_state.exports = export_all(session_id, structured_doc=raw_doc_dict, data_root=None)
            
            # Surface ingestion errors if any
            raw_texts = result_state.get("raw_text", [])
            failed_files = [d for d in raw_texts if d.get("status") == "error"]
            if failed_files:
                for ff in failed_files:
                    st.warning(f"⚠️ Warning: Could not parse '{ff.get('source_filename')}': {ff.get('error_message')}")

            st.success("🎉 Structured documentation and vector index generated successfully!")

# Tab 2: Document View & Multi-format Exports
with tab_doc:
    doc_data = st.session_state.structured_doc
    if not doc_data or not doc_data.get("sections"):
        st.markdown("""
        <div class="dashed-callout">
            💡 <strong>No structured document generated yet.</strong> Please upload files in Tab 1 and click 'Generate'.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.subheader(doc_data.get("title", "Unified Documentation"))
        
        # Download Bar - Use cached exports to prevent re-generating PDF/DOCX on every UI interaction
        exports = st.session_state.get("exports")
        if not exports:
            exports = export_all(session_id, structured_doc=doc_data, data_root=None)
            st.session_state.exports = exports

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📝 Download Markdown (.md)",
                data=exports["markdown"],
                file_name="documentation.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📄 Download Word (.docx)",
                data=exports["docx"],
                file_name="documentation.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        with col3:
            st.download_button(
                "📕 Download PDF (.pdf)",
                data=exports["pdf"],
                file_name="documentation.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.divider()

        # Display Table of Contents
        st.markdown("### Table of Contents")
        for idx, item in enumerate(doc_data.get("toc", []), 1):
            st.markdown(f"**{idx}. {item['title']}** ({item['chunk_count']} sections)")

        st.divider()

        # Display Document Sections
        for sec in doc_data.get("sections", []):
            st.markdown(f"### {sec['display_title']}")
            synth = sec.get("synthesized_content")
            if synth and synth.strip():
                st.markdown(synth.strip())
            else:
                for chunk in sec.get("chunks", []):
                    heading = chunk.get("heading_path", "Root")
                    if heading and heading not in ["Root", "Table of Contents", "Contents", "Index", "TOC"]:
                        st.markdown(f"#### {heading}")
                    st.write(chunk.get("text", ""))

            sources = ", ".join(sec.get("source_files", []))
            st.caption(f"Source file(s): {sources}")
            st.markdown("---")

# Tab 3: Trustworthy RAG Chat
with tab_chat:
    st.subheader("Conversational RAG Q&A with Fact Verification")
    st.markdown("Ask questions grounded in your uploaded documents. Answers include explicit citations or refusal if facts are unsupported.")

    from pipeline.index import InMemoryVectorStore
    has_docs = bool(InMemoryVectorStore.get_collection(session_id))
    if not has_docs and not st.session_state.chat_messages:
        st.markdown("""
        <div class="dashed-callout">
            💡 <strong>Getting Started:</strong> No documents are indexed in this session yet. Upload documentation in Tab 1 to enable grounded RAG Q&A with source citations, or type a message below!
        </div>
        """, unsafe_allow_html=True)

    chat_container = st.container()

    # Render Chat History
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    with st.expander("📌 View Source Citations"):
                        for cit in msg["citations"]:
                            st.markdown(f"- **{cit['source_file']}** > `{cit['heading_path']}`")

    # Chat Input Box at Bottom
    if user_prompt := st.chat_input("Ask a question about your documents..."):
        # Append User Message
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

        history_input = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages[:-1]
        ]

        try:
            with st.spinner("Searching document index & auditing groundedness..."):
                draft = answer_question(session_id, user_prompt, chat_history=history_input)
                verified = verify_and_refine_answer(session_id, user_prompt, draft, chat_history=history_input)

                # Store in session state history
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": verified.answer_text,
                    "citations": verified.citations
                })
        except Exception as e:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"⚠️ Error processing query: {e}",
                "citations": []
            })

        st.rerun()
