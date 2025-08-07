"""
WebRAG: Web Content RAG System
Main Streamlit application entry point
"""
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="WebRAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components after setting up the page config
from components.ui_components import (
    initialize_session_state,
    apply_custom_css,
    render_sidebar_config,
    render_welcome_section,
    render_url_input_section,
    render_content_library,
    render_domain_summary,
    render_chat_interface,
    check_api_key_status
)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render sidebar configuration
    render_sidebar_config()
    
    # Create tabs without emojis
    tab1, tab2, tab3 = st.tabs(["Welcome", "Add Content", "Chat"])
    
    with tab1:
        # Welcome tab with quick start guide
        render_welcome_section()
        
        # Show system status
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Status")
            api_status = check_api_key_status()
            if api_status:
                st.success("API Key configured")
            else:
                st.error("API Key missing")
        
        with col2:
            st.subheader("Domain Overview")
            render_domain_summary()
    
    with tab2:
        # Content indexing tab
        if not check_api_key_status():
            st.warning("Please configure your OpenAI API key in the sidebar first.")
            return
        
        # URL input section
        render_url_input_section()
        
        st.markdown("---")
        
        # Content library
        render_content_library()
    
    with tab3:
        # Chat interface tab
        if not check_api_key_status():
            st.warning("Please configure your OpenAI API key in the sidebar first.")
            return
        
        render_chat_interface()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Try refreshing the page. If the error persists, check your configuration.")
