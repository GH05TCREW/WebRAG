"""UI components for WebRAG Streamlit interface"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import json

def render_sidebar_config():
    """Render configuration sidebar"""
    with st.sidebar:
        st.header("Configuration")
        
        # New Chat button right under Configuration header
        if st.button("New Chat +", type="secondary", use_container_width=True):
            st.session_state.messages = []
            from components.chat_engine import get_chat_engine
            chat_engine = get_chat_engine()
            chat_engine.clear_history()
            st.success("Chat cleared!")
            st.rerun()
        
        # API Key configuration
        st.subheader("OpenAI API")
        
        # Load saved API key
        from utils.config import config
        saved_api_key = config.get('openai_api_key', '') or st.session_state.get('openai_api_key', '')
        
        api_key = st.text_input(
            "API Key", 
            value=saved_api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key != saved_api_key:
            st.session_state.openai_api_key = api_key
            # Update environment
            import os
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Also save to config for persistence
            from utils.config import config
            config.set('openai_api_key', api_key)
            
            st.success("🔑 API Key updated!")
            st.rerun()
        
        # Model selection
        st.subheader("Model Settings")
        
        from utils.config import config
        
        embedding_models = [
            "text-embedding-3-large",
            "text-embedding-3-small", 
            "text-embedding-ada-002"
        ]
        
        chat_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        
        selected_embedding = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=embedding_models.index(config.get('embedding_model', 'text-embedding-3-large')) if config.get('embedding_model', 'text-embedding-3-large') in embedding_models else 0
        )
        
        selected_chat_model = st.selectbox(
            "Chat Model", 
            chat_models,
            index=chat_models.index(config.get('chat_model', 'gpt-4o-mini')) if config.get('chat_model', 'gpt-4o-mini') in chat_models else 0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.get('temperature', 0.7),
            step=0.1,
            help="Controls randomness in responses"
        )
        
        # Save settings
        if st.button("Save Settings"):
            from utils.config import config
            from components.vector_store import get_vector_store
            
            # Check if embedding model changed
            old_embedding_model = config.get('embedding_model', 'text-embedding-3-large')
            embedding_changed = old_embedding_model != selected_embedding
            
            config.update({
                'embedding_model': selected_embedding,
                'chat_model': selected_chat_model,
                'temperature': temperature
            })
            
            # If embedding model changed, reinitialize vector store
            if embedding_changed:
                vector_store = get_vector_store()
                # Note: This will reinitialize with new embedding model automatically
            
            st.success("Settings saved!")
            time.sleep(1)
            st.rerun()

def render_stats_section():
    """Render statistics section showing vector store stats"""
    # Vector store stats
    st.subheader("Vector Store")
    
    try:
        from components.vector_store import get_vector_store
        vector_store = get_vector_store()
        
        stats = vector_store.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        with col2:
            st.metric("Total Sources", stats['total_sources'])
        with col3:
            size_mb = stats['storage_size'] / (1024 * 1024)
            st.metric("Storage Size", f"{size_mb:.1f} MB")
    except Exception as e:
        st.error(f"Error loading vector store stats: {e}")

def render_url_input_section():
    """Render URL input section for content indexing"""
    st.subheader("Add Web Content")
    
    # URL input
    urls_text = st.text_area(
        "URLs to Index",
        height=150,
        placeholder="Enter URLs, one per line:\nhttps://example.com\nhttps://docs.example.com\n...",
        help="Enter one URL per line. The system will crawl and extract content from each URL."
    )
    
    # Settings
    col1, col2 = st.columns(2)
    
    with col1:
        max_pages = st.number_input(
            "Max Pages per Domain",
            min_value=1,
            max_value=100,
            value=5,
            help="Maximum number of pages to crawl per domain"
        )
    
    with col2:
        max_depth = st.number_input(
            "Crawl Depth",
            min_value=1,
            max_value=5,
            value=2,
            help="How deep to crawl (1 = only provided URLs, 2 = one level of internal links)"
        )
    
    # Process button and progress
    if st.button("🚀 Process URLs", type="primary", disabled=not urls_text.strip()):
        process_urls_with_progress(urls_text, max_pages, max_depth)

def process_urls_with_progress(urls_text: str, max_pages: int, max_depth: int):
    """Process URLs with progress tracking"""
    from utils.helpers import extract_urls_from_text
    from components.url_processor import url_processor
    from components.vector_store import get_vector_store
    from utils.config import config
    
    # Check API key first
    if not config.is_api_key_valid():
        st.error("❌ OpenAI API key is required for processing URLs. Please configure it in the sidebar.")
        return
    
    # Get fresh vector store instance
    vector_store = get_vector_store()
    
    # Extract URLs
    urls = extract_urls_from_text(urls_text)
    if not urls:
        st.error("No valid URLs found!")
        return
    
    # Show URLs to be processed
    st.info(f"Found {len(urls)} URL(s) to process:")
    for url in urls[:5]:  # Show first 5
        st.write(f"- {url}")
    if len(urls) > 5:
        st.write(f"... and {len(urls) - 5} more")
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Configure processor
    url_processor.max_pages_per_domain = max_pages
    url_processor.max_depth = max_depth
    url_processor.reset_tracking()
    
    def update_progress(message: str, percent: float):
        progress_bar.progress(min(percent, 100) / 100)
        status_text.text(f"{message} ({percent:.1f}%)")
        print(f"Progress: {message} ({percent:.1f}%)")  # Also log to console
    
    try:
        # Process URLs
        update_progress("Starting URL processing...", 0)
        results = url_processor.process_urls(urls, update_progress)
        
        if not results:
            st.error("❌ No content was extracted from the provided URLs. Check the console for detailed error messages.")
            return
        
        # Prepare documents for vector store
        update_progress("Adding to vector store...", 90)
        documents = []
        
        for url, content, title in results:
            documents.append({
                'url': url,
                'content': content,
                'title': title,
                'metadata': {
                    'indexed_at': datetime.now().isoformat(),
                    'source_type': 'web'
                }
            })
        
        # Add to vector store
        success = vector_store.add_documents(documents)
        
        if success:
            update_progress("Complete!", 100)
            st.success(f"✅ Successfully processed and indexed {len(results)} page(s)!")
            
            # Show detailed summary
            domains = {}
            total_content_length = 0
            
            for url, content, title in results:
                domain = url.split('/')[2]
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append({'title': title, 'length': len(content)})
                total_content_length += len(content)
            
            st.info(f"📊 **Summary:**\n- **{len(domains)} domain(s)** processed\n- **{total_content_length:,} characters** of content indexed")
            
            # Show per-domain breakdown
            for domain, pages in domains.items():
                with st.expander(f"🌐 {domain} ({len(pages)} page(s))"):
                    for page in pages:
                        st.write(f"- {page['title']} ({page['length']:,} chars)")
        else:
            update_progress("Failed to add to vector store", 100)
            st.error("❌ Failed to add documents to vector store. Please check your API key configuration.")
    
    except Exception as e:
        error_msg = f"❌ Error processing URLs: {str(e)}"
        st.error(error_msg)
        print(f"Detailed error: {e}")  # Log full error to console
        import traceback
        traceback.print_exc()
    
    finally:
        progress_bar.empty()
        status_text.empty()

def render_content_library():
    """Render content library showing indexed content"""
    st.subheader("Content Library")
    
    from components.vector_store import get_vector_store
    vector_store = get_vector_store()
    
    # Get indexed sources
    sources = vector_store.get_indexed_sources()
    
    if not sources:
        st.info("No content indexed yet. Add some URLs to get started!")
        return
    
    # Control buttons
    # Search/filter
    search_term = st.text_input("Search content", placeholder="Filter by title or domain...")
    
    # Reset confirmation if user interacts with other elements
    if search_term:
        pass  # Removed confirmation reset logic
    
    # Filter sources
    if search_term:
        sources = [s for s in sources if 
                  search_term.lower() in s['title'].lower() or 
                  search_term.lower() in s['domain'].lower()]
    
    # Show stats
    st.write(f"**Showing {len(sources)} of {len(vector_store.get_indexed_sources())} indexed sources**")
    
    # Clear all button positioned under the stats
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear All", type="secondary", help="Delete all indexed content"):
            if vector_store.delete_all_sources():
                st.rerun()
            else:
                st.error("Failed to clear content")
    
    # Display sources
    for i, source in enumerate(sources):
        with st.expander(f"{source['title']} - {source['domain']}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**URL:** {source['url']}")
                st.write(f"**Chunks:** {source['chunks']}")
                
            with col2:
                try:
                    indexed_date = datetime.fromisoformat(source['indexed_at'].replace('Z', '+00:00'))
                    st.write(f"**Indexed:** {indexed_date.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.write(f"**Indexed:** {source.get('indexed_at', 'Unknown')}")
                
            with col3:
                # Individual delete button
                delete_key = f"delete_{source['url']}_{i}"
                if st.button("Delete", key=delete_key, help=f"Delete content from {source['domain']}"):
                    with st.spinner("Deleting..."):
                        if vector_store.delete_source(source['url']):
                            st.rerun()
                        else:
                            st.error("Failed to delete content. Check the console for details.")
    
    # Removed confirmation state management

def render_chat_interface():
    """Render chat interface with ChatGPT-style layout"""
    st.subheader("Chat with Your Content")
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Check if we have indexed content
    from components.vector_store import get_vector_store
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    
    if stats['total_documents'] == 0:
        st.warning("No content indexed yet! Please add some URLs in the 'Add Content' tab first.")
        return
    
    # Chat controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"Ask questions about your {stats['total_sources']} indexed sources")
    
    # Note: New Chat button moved to sidebar
    
    # Display messages in normal order (oldest first, newest last like ChatGPT)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])
    
    # Chat input at bottom (ChatGPT style)
    if prompt := st.chat_input("Ask a question about your indexed content..."):
        # Add user message to end of messages (ChatGPT style)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with streaming
        from components.chat_engine import get_chat_engine
        chat_engine = get_chat_engine()
        
        # Show AI response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources = []
            
            try:
                for chunk in chat_engine.stream_response(prompt):
                    content = chunk.get('content', '')
                    sources = chunk.get('sources', [])
                    finished = chunk.get('finished', False)
                    
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                    
                    if finished:
                        break
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Show sources
                if sources:
                    render_sources(sources)
                
                # Add assistant response to messages
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
                    
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
    
    # Auto-scroll to bottom after messages are rendered
    st.markdown("""
    <script>
    setTimeout(function() {
        window.scrollTo(0, document.body.scrollHeight);
        
        // Reset chat input height after sending message
        const chatInput = document.querySelector('.stChatInput textarea[data-testid="stChatInputTextArea"]');
        if (chatInput && chatInput.value === '') {
            chatInput.style.height = 'auto';
            chatInput.style.height = '20px';
        }
    }, 100);
    
    // Add event listener to auto-resize textarea on input
    document.addEventListener('DOMContentLoaded', function() {
        const observer = new MutationObserver(function(mutations) {
            const chatInput = document.querySelector('.stChatInput textarea[data-testid="stChatInputTextArea"]');
            if (chatInput && !chatInput.hasAttribute('data-resizer-added')) {
                chatInput.setAttribute('data-resizer-added', 'true');
                
                function adjustHeight() {
                    if (chatInput.value === '') {
                        chatInput.style.height = '20px';
                    } else {
                        chatInput.style.height = 'auto';
                        chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
                    }
                }
                
                chatInput.addEventListener('input', adjustHeight);
                chatInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        setTimeout(adjustHeight, 50);
                    }
                });
                
                // Initial adjustment
                adjustHeight();
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
    </script>
    """, unsafe_allow_html=True)

def render_sources(sources: List[Dict[str, Any]]):
    """Render sources section"""
    if not sources:
        return
    
    with st.expander(f"Sources ({len(sources)} found)", expanded=False):
        for i, source in enumerate(sources):
            st.write(f"**{i+1}. [{source['title']}]({source['url']})**")
            st.write(f"Domain: {source['domain']}")
            st.write(f"Relevance: {source['relevance_score']:.3f}")
            
            if 'snippet' in source:
                st.write(f"Preview: {source['snippet']}")
            
            st.divider()

def render_domain_summary():
    """Render domain summary view"""
    from components.vector_store import get_vector_store
    vector_store = get_vector_store()
    
    domain_summary = vector_store.get_domain_summary()
    
    if not domain_summary:
        st.info("No indexed domains yet.")
        return
    
    # Create summary dataframe
    summary_data = []
    for domain, info in domain_summary.items():
        summary_data.append({
            'Domain': domain,
            'URLs': len(info['urls']),
            'Total Chunks': info['total_chunks'],
            'Last Indexed': info['last_indexed'][:19] if info['last_indexed'] else 'Unknown'
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True)
    
    # Detailed view for each domain
    for domain, info in domain_summary.items():
        with st.expander(f"{domain} ({len(info['urls'])} URLs)"):
            for url_info in info['urls']:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"[{url_info['title']}]({url_info['url']})")
                
                with col2:
                    st.write(f"Chunks: {url_info['chunks']}")
                
                with col3:
                    indexed_date = datetime.fromisoformat(url_info['indexed_at'])
                    st.write(f"{indexed_date.strftime('%m/%d %H:%M')}")

def check_api_key_status():
    """Check and display API key status"""
    from utils.config import config
    
    if not config.is_api_key_valid():
        st.error("⚠️ OpenAI API key not configured! Please add your API key in the sidebar.")
        return False
    
    return True

def render_welcome_section():
    """Render welcome section with quick start guide"""
    st.markdown("""
    # WebRAG
    
    ### Web Content RAG System
    
    Welcome to WebRAG! This application allows you to easily index web content and chat with an AI about it.
    
    ## Quick Start:
    
    1. **Configure API Key**
       - Add your OpenAI API key in the sidebar
       - Choose your preferred models
    
    2. **Add Content**
       - Go to the "Add Content" tab
       - Paste URLs you want to index
       - Click "Process URLs"
    
    3. **Start Chatting**
       - Switch to the "Chat" tab
       - Ask questions about your indexed content
       - Get AI responses with source citations
    
    ## Features:
    - Smart web content extraction
    - Automatic content chunking and indexing
    - AI-powered Q&A with source citations
    - Multiple embedding and chat models
    - Content management and search
    
    Get started by configuring your API key in the sidebar!
    """)

# Helper functions for session management
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.openai_api_key = ""

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    /* Off-white background, black text */
    .stApp {
        background-color: #f7f7f8;
        color: #000000;
    }
    
    /* Streamlit header - make it light gray */
    .stAppHeader.st-emotion-cache-gquqoo.e3g0k5y1,
    .stAppToolbar.st-emotion-cache-14vh5up.e3g0k5y2,
    header[data-testid="stHeader"] {
        background-color: #f0f0f0 !important;
        border-bottom: 1px solid #e5e5e5;
    }
    
    /* Sidebar styling with very light background */
    .stSidebar.st-emotion-cache-1lqf7hx.e1v5e29v0,
    .css-1d391kg,
    section[data-testid="stSidebar"] {
        background-color: #fafafa !important;
        border-right: 1px solid #e5e5e5;
    }
    
    /* Sidebar content background */
    div[data-testid="stSidebarContent"] {
        background-color: #fafafa !important;
        padding-top: 1rem !important;
    }
    
    /* Reduce sidebar content container padding */
    .css-1d391kg .block-container,
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Sidebar header spacing */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
/* Remove main container padding at top */
.main .block-container {
    padding-top: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
    padding-bottom: 100px; /* Add space for fixed chat input */
}    /* All text should be black */
    .stApp, .stApp * {
        color: #000000 !important;
    }
    
    /* Headers - ensure they are black */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Sidebar text should also be black */
    .css-1d391kg, .css-1d391kg *,
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
/* Chat container */
.stChatMessage {
    background-color: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #000000 !important;
}

/* Chat input centered at bottom */
.stChatInput {
    position: fixed !important;
    bottom: 24px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 600px !important;
    max-width: calc(100vw - 2rem) !important;
    background-color: #ffffff !important;
    padding: 0.75rem !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    z-index: 1000 !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
}

/* On mobile, slightly smaller and still centered */
@media (max-width: 768px) {
    .stChatInput {
        width: calc(100vw - 2rem) !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
    }
}

/* Clean up all chat input inner containers */
.stChatInput > div,
.stChatInput [data-baseweb="textarea"],
.stChatInput [data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Chat input textarea styling - override all conflicting styles */
.stChatInput textarea[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    color: #111827 !important;
    caret-color: #111827 !important;
    font-size: 16px !important;
    width: 100% !important;
    padding: 0.5rem 1rem !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    resize: none !important;
    height: auto !important;
    min-height: 20px !important;
    max-height: 200px !important;
    overflow-y: auto !important;
    transition: height 0.1s ease !important;
}

/* Placeholder text for chat input */
.stChatInput textarea[data-testid="stChatInputTextArea"]::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}

/* Focus state for textarea - keep it clean */
.stChatInput textarea[data-testid="stChatInputTextArea"]:focus {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* Chat input submit button - just icon, no background */
.stChatInput button[data-testid="stChatInputSubmitButton"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0.5rem !important;
    margin-left: 0.5rem !important;
    box-shadow: none !important;
    transition: color 0.2s !important;
    min-width: 40px !important;
    height: 40px !important;
    cursor: pointer !important;
}

.stChatInput button[data-testid="stChatInputSubmitButton"]:hover {
    background: #f3f4f6 !important;
    color: #374151 !important;
}

.stChatInput button[data-testid="stChatInputSubmitButton"]:disabled {
    background: transparent !important;
    color: #d1d5db !important;
    cursor: not-allowed !important;
}

.stChatInput button[data-testid="stChatInputSubmitButton"]:disabled:hover {
    background: transparent !important;
    color: #d1d5db !important;
}

    /* Button styling */
    .stButton > button {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
        color: #000000 !important;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #f0f0f0;
        border-color: #d0d7de;
        color: #000000 !important;
    }
    
    /* Secondary buttons - light gray background */
    button[data-testid="stBaseButton-secondary"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e5e5e5 !important;
        color: #000000 !important;
    }
    
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #e9ecef !important;
    }
    
    /* Button containers */
    .st-emotion-cache-1anq8dj {
        background-color: #f8f9fa !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px !important;
    }
    
    /* Input fields - ensure white background */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px;
        color: #000000 !important;
    }
    
    /* Placeholder text styling for textarea */
    .stTextArea > div > div > textarea::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
    }
    
    /* Placeholder text for input fields */
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
    }
    
    /* Sidebar input fields specifically */
    .css-1d391kg .stTextInput > div > div > input,
    .css-1d391kg .stTextArea > div > div > textarea,
    .css-1d391kg .stSelectbox > div > div > select,
    .css-1d391kg .stNumberInput > div > div > input,
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stTextArea > div > div > textarea,
    section[data-testid="stSidebar"] .stSelectbox > div > div > select,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        color: #000000 !important;
    }
    
    /* Selectbox dropdown styling */
    .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        color: #000000 !important;
    }
    
    /* Slider styling */
    .css-1d391kg .stSlider,
    section[data-testid="stSidebar"] .stSlider {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid #e5e5e5;
    }
    
    /* Tabs styling */
    .stTabs > div > div > div > button {
        color: #666666;
        font-weight: 500;
        background-color: transparent;
    }
    
    .stTabs > div > div > div > button:hover {
        color: #000000 !important;
    }
    
    .stTabs > div > div > div > button[aria-selected="true"] {
        color: #000000 !important;
    }
    
    /* Info boxes */
    .stInfo, .stWarning, .stError, .stSuccess {
        background-color: #ffffff;
        border-radius: 6px;
        border-left: 4px solid;
        padding: 1rem;
        color: #000000 !important;
    }
    
    .stInfo {
        border-left-color: #0969da;
    }
    
    .stSuccess {
        border-left-color: #1a7f37;
    }
    
    .stWarning {
        border-left-color: #bf8700;
    }
    
    .stError {
        border-left-color: #d1242f;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
        color: #000000 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-top: none;
        border-radius: 0 0 6px 6px;
        color: #000000 !important;
    }
    
    /* Modern expander styling - target emotion-based classes */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px !important;
    }
    
    /* Expander header - all states */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] .st-emotion-cache-c36nl0,
    .st-emotion-cache-c36nl0 {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 6px !important;
    }
    
    /* Expander header hover state */
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] .st-emotion-cache-c36nl0:hover,
    .st-emotion-cache-c36nl0:hover {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    /* Expander header focus/active states */
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active,
    [data-testid="stExpander"] .st-emotion-cache-c36nl0:focus,
    [data-testid="stExpander"] .st-emotion-cache-c36nl0:active,
    .st-emotion-cache-c36nl0:focus,
    .st-emotion-cache-c36nl0:active {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        outline: none !important;
    }
    
    /* Expander header when expanded */
    [data-testid="stExpander"][aria-expanded="true"] summary,
    [data-testid="stExpander"][aria-expanded="true"] .st-emotion-cache-c36nl0,
    .st-emotion-cache-c36nl0[aria-expanded="true"] {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        border-radius: 6px 6px 0 0 !important;
    }
    
    /* Ensure expander text and icons stay black */
    [data-testid="stExpander"] *,
    .st-emotion-cache-c36nl0 *,
    .st-emotion-cache-1tz5wcb *,
    .st-emotion-cache-zkd0x0 * {
        color: #000000 !important;
    }
    
    /* Ensure all markdown text is black */
    .stMarkdown, .stMarkdown * {
        color: #000000 !important;
    }
    
    /* Metrics text should be black */
    .metric-container {
        color: #000000 !important;
    }
    
    /* Progress bar text */
    .stProgress {
        color: #000000 !important;
    }
    
    /* Dataframe text */
    .stDataFrame {
        color: #000000 !important;
    }
    
    /* Sidebar labels and help text */
    .css-1d391kg label,
    section[data-testid="stSidebar"] label {
        color: #000000 !important;
    }
    
    .css-1d391kg .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }
    
    /* Remove Statistics emoji */
    section[data-testid="stSidebar"] h3 {
        color: #000000 !important;
    }
    
    /* Number input containers - white background */
    div[data-testid="stNumberInputContainer"],
    .st-emotion-cache-9gx57n {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px !important;
    }
    
    /* Number input field specifically */
    input[data-testid="stNumberInputField"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
    }
    
    /* Number input step buttons */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        color: #000000 !important;
    }
    
    button[data-testid="stNumberInputStepDown"]:hover,
    button[data-testid="stNumberInputStepUp"]:hover {
        background-color: #f0f0f0 !important;
    }
    
    /* Password input field - complete clean redesign */
    div[data-testid="stTextInputRootElement"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.2s ease !important;
        overflow: hidden !important;
    }
    
    /* Focus state */
    div[data-testid="stTextInputRootElement"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Base wrapper - clean it up */
    div[data-testid="stTextInputRootElement"] div[data-baseweb="base-input"] {
        background: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        display: flex !important;
        align-items: stretch !important;
    }
    
    /* Input field */
    div[data-testid="stTextInputRootElement"] input {
        background: none !important;
        border: none !important;
        outline: none !important;
        padding: 10px 12px !important;
        color: #111827 !important;
        font-size: 14px !important;
        flex: 1 !important;
        min-width: 0 !important;
    }
    
    /* Eye button - simple approach */
    div[data-testid="stTextInputRootElement"] button {
        background: none !important;
        border: none !important;
        padding: 10px 0 10px 12px !important;
        margin: 0 !important;
        cursor: pointer !important;
        border-left: 1px solid #e5e7eb !important;
    }
    
    /* Override specific Streamlit class padding */
    div[data-testid="stTextInputRootElement"] button.st-b9 {
        padding-right: 0 !important;
    }
    
    /* Eye icon - simple */
    div[data-testid="stTextInputRootElement"] button svg {
        width: 16px !important;
        height: 16px !important;
        color: #6b7280 !important;
        fill: currentColor !important;
        vertical-align: middle !important;
    }
    
    /* Remove all tooltips completely */
    div[data-testid="stTextInputRootElement"] *[title] {
        pointer-events: none !important;
    }
    
    div[data-testid="stTextInputRootElement"] button[title] {
        pointer-events: auto !important;
    }
    
    div[data-testid="stTextInputRootElement"] button[title]::before,
    div[data-testid="stTextInputRootElement"] button[title]::after {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
