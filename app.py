"""
Streamlit Biology Tutor Chatbot
Interactive interface for the RAG system
"""

import streamlit as st
from pathlib import Path
import time
from datetime import datetime

from rag_system import BiologyRAGSystem
from config import *


# ===== Page Configuration =====
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===== Custom CSS =====
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #2E7D32;
        padding: 1rem 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #558B2F;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Source cards */
    .source-card {
        background-color: #F1F8E9;
        border-left: 4px solid #2E7D32;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    /* Confidence badges */
    .confidence-high {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .confidence-medium {
        background-color: #FFC107;
        color: black;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .confidence-low {
        background-color: #FF5722;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ===== Initialize Session State =====
def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "system_loaded" not in st.session_state:
        st.session_state.system_loaded = False
    
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0


# ===== Load RAG System =====
@st.cache_resource
def load_rag_system(api_key):
    """Load RAG system (cached for performance)"""
    try:
        rag = BiologyRAGSystem(
            data_dir=str(DATA_DIR),
            api_key=api_key,
            chat_model=CHAT_MODEL
        )
        return rag, None
    except FileNotFoundError as e:
        return None, ERROR_DATA_NOT_FOUND
    except Exception as e:
        return None, f"Error loading system: {str(e)}"


# ===== UI Components =====
def render_header():
    """Render app header"""
    st.markdown(f'<div class="main-title">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with settings and info"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Retrieval settings
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=15,
            value=TOP_K_RETRIEVE,
            help="More sources = more comprehensive but slower"
        )
        
        show_sources = st.checkbox(
            "Show source references",
            value=SHOW_SOURCES_DEFAULT,
            help="Display textbook page references"
        )
        
        show_thinking = st.checkbox(
            "Show retrieval process",
            value=SHOW_THINKING_DEFAULT,
            help="Show what's happening behind the scenes"
        )
        
        st.divider()
        
        # Statistics
        st.header("📊 Statistics")
        st.metric("Questions Asked", st.session_state.request_count)
        st.metric("Chat History", len(st.session_state.messages) // 2)
        
        st.divider()
        
        # Info
        st.header("ℹ️ About")
        st.markdown("""
        **Biology Tutor** uses advanced RAG (Retrieval-Augmented Generation) to answer questions from your textbook.
        
        **Features:**
        - 🔍 Hybrid search (semantic + keyword)
        - 📚 Smart source citation
        - 🌐 Multi-language support
        - 💡 Educational explanations
        
        **Topics Covered:**
        - Bryophyta (মস উদ্ভিদ)
        - Pteridophyta (ফার্ন উদ্ভিদ)
        """)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        return top_k, show_sources, show_thinking


def render_source_card(source, index):
    """Render a source reference card"""
    confidence = source.get("score", 0)
    
    if confidence > 0.7:
        badge = "🟢 HIGH"
        badge_class = "confidence-high"
    elif confidence > 0.5:
        badge = "🟡 MEDIUM"
        badge_class = "confidence-medium"
    else:
        badge = "🔴 LOW"
        badge_class = "confidence-low"
    
    markers = []
    if source.get("has_list"):
        markers.append("📋 List")
    if source.get("has_heading"):
        markers.append("📌 Heading")
    
    marker_str = " | ".join(markers) if markers else ""
    
    # Use st.markdown without HTML encoding issues
    st.markdown(f"**Source {index + 1}** :{badge_class}[{badge}] | Page {source['page']} | Relevance: {confidence:.3f}")
    if marker_str:
        st.caption(marker_str)
    st.info(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])


def render_chat_message(role, content, sources=None):
    """Render a chat message"""
    with st.chat_message(role):
        st.markdown(content)
        
        # Show sources if available
        if sources and role == "assistant":
            with st.expander(f"📚 View {len(sources)} Textbook References"):
                for i, source in enumerate(sources):
                    render_source_card(source, i)


# ===== Main App =====
def main():
    """Main application logic"""
    init_session_state()
    
    # Render header
    render_header()
    
    # Get API key
    api_key = st.secrets.get("GEMINI_API_KEY", GEMINI_API_KEY)
    
    if not api_key:
        st.error(ERROR_NO_API_KEY)
        st.stop()
    
    # Check if data exists
    if not DATA_DIR.exists() or not (DATA_DIR / "chunks.json").exists():
        st.error(ERROR_DATA_NOT_FOUND)
        st.stop()
    
    # Load RAG system
    if not st.session_state.system_loaded:
        with st.spinner("🔄 Loading RAG system... (this may take a minute)"):
            rag_system, error = load_rag_system(api_key)
            
            if error:
                st.error(error)
                st.stop()
            
            st.session_state.rag_system = rag_system
            st.session_state.system_loaded = True
            st.success("✅ System ready!")
            time.sleep(1)
            st.rerun()
    
    # Render sidebar and get settings
    top_k, show_sources, show_thinking = render_sidebar()
    
    # Welcome message
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(WELCOME_MESSAGE)
    
    # Display ALL existing chat history
    for message in st.session_state.messages:
        render_chat_message(
            message["role"],
            message["content"],
            message.get("sources") if show_sources else None
        )
    
    # Chat input at the bottom (like ChatGPT)
    if prompt := st.chat_input("আপনার প্রশ্ন লিখুন... (Type your question...)"):
        # Check rate limit
        st.session_state.request_count += 1
        if st.session_state.request_count > MAX_REQUESTS_PER_USER:
            st.warning("⚠️ You've reached the maximum number of questions for this session. Please refresh the page.")
            st.stop()
        
        # Add user message to history FIRST
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response in background
        if show_thinking:
            with st.status("🔍 Thinking...", expanded=False) as status:
                st.write("🔄 Expanding query...")
                st.write("🔍 Searching textbook...")
                st.write("🧠 Generating answer...")
                
                result = st.session_state.rag_system.ask(
                    prompt,
                    top_k=top_k,
                    return_sources=True
                )
                
                status.update(label="✅ Complete!", state="complete")
        else:
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.rag_system.ask(
                    prompt,
                    top_k=top_k,
                    return_sources=True
                )
        
        # Extract answer and sources
        if isinstance(result, dict):
            answer = result["answer"]
            sources = result.get("sources", [])
        else:
            answer = result
            sources = []
        
        # Save assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources if show_sources else None
        })
        
        # Force clean rerun to display everything properly
        st.rerun()


if __name__ == "__main__":
    main()
