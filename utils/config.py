"""Configuration management for WebRAG 2.0"""
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration management"""
    
    def __init__(self):
        self.config_file = "data/config.json"
        self.default_config = {
            "embedding_model": "text-embedding-3-large",
            "chat_model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_pages_per_domain": 50,
            "max_crawl_depth": 2,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k_results": 5
        }
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                self.config = {**self.default_config, **config}
            else:
                self.config = self.default_config.copy()
                self.save_config()
            return self.config
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = self.default_config.copy()
            return self.config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value and save"""
        self.config[key] = value
        self.save_config()
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.config.update(updates)
        self.save_config()
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment, session state, or config file"""
        # First try environment variable
        env_key = os.getenv("OPENAI_API_KEY", "")
        if env_key:
            return env_key
        
        # Try session state if available
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and 'openai_api_key' in st.session_state:
                session_key = st.session_state.openai_api_key or ""
                if session_key:
                    return session_key
        except:
            pass
        
        # Fallback to saved config
        return self.config.get('openai_api_key', '')
    
    @property
    def openai_base_url(self) -> str:
        """Get OpenAI base URL from environment"""
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    def is_api_key_valid(self) -> bool:
        """Check if OpenAI API key is set"""
        return bool(self.openai_api_key and len(self.openai_api_key.strip()) > 0)

# Global config instance
config = Config()
