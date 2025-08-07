"""Vector store component for WebRAG 2.0"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from utils.config import config
from utils.helpers import generate_content_hash, get_domain
import pandas as pd

class VectorStore:
    """Manage vector storage and retrieval using ChromaDB"""
    
    def __init__(self):
        self.db_path = "data/vector_db"
        self.metadata_file = "data/content_metadata.json"
        self.client = None
        self.collection = None
        self.embeddings = None
        self.metadata = self._load_metadata()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="webrag_content",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embeddings
            if config.is_api_key_valid():
                self.embeddings = OpenAIEmbeddings(
                    model=config.get('embedding_model', 'text-embedding-3-large'),
                    openai_api_key=config.openai_api_key,
                    openai_api_base=config.openai_base_url
                )
            
            print(f"Vector store initialized with {self.collection.count()} documents")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.client = None
            self.collection = None
    
    def _ensure_embeddings_initialized(self):
        """Ensure embeddings are initialized, try to initialize if not"""
        if not self.embeddings and config.is_api_key_valid():
            try:
                embedding_model = config.get('embedding_model', 'text-embedding-3-large')
                api_key = config.openai_api_key
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                print(f"🔑 Vector Store using API key: {masked_key}")
                
                self.embeddings = OpenAIEmbeddings(
                    model=embedding_model,
                    openai_api_key=api_key,
                    openai_api_base=config.openai_base_url
                )
                print(f"✅ Embeddings initialized successfully with model: {embedding_model}")
                
                # Check if collection exists and if embedding dimensions match
                if self.collection and self.collection.count() > 0:
                    # Test embedding dimension
                    test_embedding = self.embeddings.embed_query("test")
                    expected_dim = len(test_embedding)
                    
                    # Get a sample from collection to check existing dimension
                    try:
                        sample = self.collection.peek(limit=1)
                        if sample['embeddings'] and len(sample['embeddings']) > 0:
                            existing_embedding = sample['embeddings'][0]
                            existing_dim = len(existing_embedding) if hasattr(existing_embedding, '__len__') else 0
                            if existing_dim != expected_dim:
                                print(f"⚠️ Embedding dimension mismatch! Existing: {existing_dim}, New: {expected_dim}")
                                print("🔄 Resetting collection to match new embedding model...")
                                self._reset_collection_for_new_model()
                    except Exception as e:
                        print(f"Could not check existing dimensions: {e}")
                        # If we can't check dimensions, assume they might be incompatible and reset
                        if "dimension" in str(e).lower() or "ambiguous" in str(e).lower():
                            print("🔄 Resetting collection due to dimension issues...")
                            self._reset_collection_for_new_model()
                        
            except Exception as e:
                print(f"❌ Error initializing embeddings: {e}")
    
    def _reset_collection_for_new_model(self):
        """Reset collection when embedding model changes"""
        try:
            # Delete existing collection
            self.client.delete_collection("webrag_content")
            # Create new collection
            self.collection = self.client.get_or_create_collection(
                name="webrag_content",
                metadata={"hnsw:space": "cosine"}
            )
            # Clear metadata
            self.metadata = {}
            self._save_metadata()
            print("✅ Collection reset for new embedding model")
        except Exception as e:
            print(f"❌ Error resetting collection: {e}")
    
    def reinitialize_embeddings(self):
        """Force reinitialize embeddings (useful when API key changes)"""
        self.embeddings = None
        self._ensure_embeddings_initialized()
        return self.embeddings is not None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        documents: List of dicts with keys: 'content', 'url', 'title', 'metadata'
        """
        # Try to initialize embeddings if not available
        self._ensure_embeddings_initialized()
        
        if not self.collection or not self.embeddings:
            if not config.is_api_key_valid():
                print("❌ Cannot add documents: OpenAI API key is not configured")
            else:
                print("❌ Vector store not properly initialized - collection or embeddings missing")
            return False
        
        if not documents:
            return True
        
        try:
            # Process documents
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                content = doc.get('content', '').strip()
                if not content or len(content) < 50:  # Skip very short content
                    continue
                
                url = doc.get('url', '')
                title = doc.get('title', '')
                doc_metadata = doc.get('metadata', {})
                
                # Split content into chunks
                from utils.text_processing import text_processor
                chunks = text_processor.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{generate_content_hash(url)}_{i}"
                    
                    # Skip if already exists
                    if self._document_exists(chunk_id):
                        continue
                    
                    texts.append(chunk)
                    
                    # Create metadata for chunk
                    chunk_metadata = {
                        'url': url,
                        'title': title,
                        'domain': get_domain(url),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'timestamp': datetime.now().isoformat(),
                        'content_hash': generate_content_hash(content),
                        **doc_metadata
                    }
                    
                    metadatas.append(chunk_metadata)
                    ids.append(chunk_id)
                    
                # Update content metadata
                self.metadata[url] = {
                    'title': title,
                    'domain': get_domain(url),
                    'chunks': len(chunks),
                    'indexed_at': datetime.now().isoformat(),
                    'content_hash': generate_content_hash(content),
                    **doc_metadata
                }
            
            if not texts:
                print("No new content to add")
                return True
            
            # Generate embeddings
            print(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            # Save metadata
            self._save_metadata()
            
            print(f"Added {len(texts)} chunks to vector store")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid_api_key" in error_msg:
                print(f"❌ Invalid API key error: {e}")
                print("💡 Please check your OpenAI API key in the sidebar settings")
            elif "dimension" in error_msg.lower():
                print(f"❌ Embedding dimension mismatch: {e}")
                print("🔄 Trying to reset collection for new embedding model...")
                try:
                    self._reset_collection_for_new_model()
                    print("✅ Collection reset. Please try again.")
                except:
                    print("❌ Failed to reset collection. You may need to manually delete the data/vector_db folder.")
            else:
                print(f"❌ Error adding documents to vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, filter_domain: str = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        Returns: List of dicts with 'content', 'metadata', 'score'
        """
        if not self.collection or not self.embeddings:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build where clause for filtering
            where_clause = None
            if filter_domain:
                where_clause = {"domain": filter_domain}
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'score': 1 - distance  # Convert distance to similarity score
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def delete_source(self, url: str) -> bool:
        """Delete all content from a specific URL"""
        if not self.collection:
            return False
        
        try:
            # Find all chunks for this URL using the correct API
            results = self.collection.get(
                where={"url": url}
            )
            
            if results and results.get('ids') and len(results['ids']) > 0:
                # Delete chunks
                self.collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} chunks for {url}")
            else:
                print(f"No chunks found for URL: {url}")
            
            # Remove from metadata
            if url in self.metadata:
                del self.metadata[url]
                self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error deleting source {url}: {e}")
            return False
    
    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """Get list of all indexed sources with metadata"""
        sources = []
        for url, metadata in self.metadata.items():
            sources.append({
                'url': url,
                'title': metadata.get('title', 'Untitled'),
                'domain': metadata.get('domain', ''),
                'chunks': metadata.get('chunks', 0),
                'indexed_at': metadata.get('indexed_at', ''),
                'content_hash': metadata.get('content_hash', '')
            })
        
        # Sort by indexed date (newest first)
        sources.sort(key=lambda x: x.get('indexed_at', ''), reverse=True)
        return sources
    
    def get_domain_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of indexed content by domain"""
        domain_summary = {}
        
        for url, metadata in self.metadata.items():
            domain = metadata.get('domain', 'unknown')
            
            if domain not in domain_summary:
                domain_summary[domain] = {
                    'urls': [],
                    'total_chunks': 0,
                    'last_indexed': ''
                }
            
            domain_summary[domain]['urls'].append({
                'url': url,
                'title': metadata.get('title', 'Untitled'),
                'chunks': metadata.get('chunks', 0),
                'indexed_at': metadata.get('indexed_at', '')
            })
            
            domain_summary[domain]['total_chunks'] += metadata.get('chunks', 0)
            
            # Update last indexed time
            indexed_at = metadata.get('indexed_at', '')
            if indexed_at > domain_summary[domain]['last_indexed']:
                domain_summary[domain]['last_indexed'] = indexed_at
        
        return domain_summary
    
    def update_embedding_model(self, model_name: str) -> bool:
        """Update the embedding model and re-initialize"""
        try:
            config.set('embedding_model', model_name)
            
            if config.is_api_key_valid():
                self.embeddings = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=config.openai_api_key,
                    openai_api_base=config.openai_base_url
                )
                return True
            return False
            
        except Exception as e:
            print(f"Error updating embedding model: {e}")
            return False
    
    def _document_exists(self, doc_id: str) -> bool:
        """Check if document already exists in collection"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load content metadata from file"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save content metadata to file"""
        try:
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            'total_documents': 0,
            'total_sources': len(self.metadata),
            'total_domains': len(set(meta.get('domain', '') for meta in self.metadata.values())),
            'storage_size': 0
        }
        
        if self.collection:
            stats['total_documents'] = self.collection.count()
        
        # Calculate storage size
        if os.path.exists(self.db_path):
            for root, dirs, files in os.walk(self.db_path):
                stats['storage_size'] += sum(os.path.getsize(os.path.join(root, name)) for name in files)
        
        return stats
    
    def delete_all_sources(self) -> bool:
        """Delete all content from vector store"""
        try:
            if self.collection:
                # Get all document IDs
                all_results = self.collection.get()
                
                if all_results and all_results.get('ids') and len(all_results['ids']) > 0:
                    # Delete all documents
                    self.collection.delete(ids=all_results['ids'])
                    print(f"Deleted all {len(all_results['ids'])} chunks from vector store")
                else:
                    print("No documents found to delete")
            
            # Clear metadata
            self.metadata = {}
            self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error deleting all sources: {e}")
            return False
    
    def reset_database(self) -> bool:
        """Reset/clear the entire vector database"""
        try:
            if self.client:
                self.client.reset()
            
            self.metadata = {}
            self._save_metadata()
            
            # Reinitialize
            self._initialize_db()
            
            return True
            
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False

# Remove global instance - create fresh instances when needed
def get_vector_store():
    """Get a fresh vector store instance with current config"""
    return VectorStore()
