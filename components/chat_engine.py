"""Chat engine component for WebRAG 2.0"""
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
import json
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from utils.config import config
from components.vector_store import get_vector_store

class ChatEngine:
    """Handle chat interactions with RAG-enhanced responses"""
    
    def __init__(self):
        self.chat_history_file = "data/chat_history.json"
        self.conversation_history = self._load_chat_history()
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if config.is_api_key_valid():
                api_key = config.openai_api_key
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                print(f"🔑 Chat Engine using API key: {masked_key}")
                
                self.llm = ChatOpenAI(
                    model=config.get('chat_model', 'gpt-4o-mini'),
                    temperature=config.get('temperature', 0.7),
                    openai_api_key=api_key,
                    openai_api_base=config.openai_base_url,
                    streaming=True
                )
                print(f"✅ Chat engine initialized with model: {config.get('chat_model', 'gpt-4o-mini')}")
            else:
                print("❌ Chat engine: API key not available")
                self.llm = None
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            self.llm = None
    
    def reinitialize_llm(self):
        """Force reinitialize the language model (useful when API key changes)"""
        self.llm = None
        self._initialize_llm()
        return self.llm is not None
    
    def generate_response(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Generate response using RAG
        Returns: {
            'response': str,
            'sources': List[Dict],
            'conversation_id': str,
            'timestamp': str,
            'model_used': str
        }
        """
        if not self.llm:
            return {
                'response': "Chat engine not properly initialized. Please check your OpenAI API key.",
                'sources': [],
                'conversation_id': None,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'none'
            }
        
        try:
            # Get fresh vector store instance
            vector_store = get_vector_store()
            
            # Retrieve relevant documents
            relevant_docs = vector_store.search(
                question, 
                top_k=config.get('top_k_results', 5)
            )
            
            # Build context from retrieved documents
            context = self._build_context(relevant_docs)
            
            # Create conversation messages
            messages = self._build_messages(question, context, session_id)
            
            # Generate response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Create conversation entry
            conversation_entry = {
                'id': self._generate_conversation_id(),
                'session_id': session_id,
                'question': question,
                'response': response_text,
                'sources': self._format_sources(relevant_docs),
                'timestamp': datetime.now().isoformat(),
                'model_used': config.get('chat_model', 'gpt-4o-mini'),
                'context_used': len(relevant_docs) > 0
            }
            
            # Add to history
            self._add_to_history(conversation_entry)
            
            return {
                'response': response_text,
                'sources': conversation_entry['sources'],
                'conversation_id': conversation_entry['id'],
                'timestamp': conversation_entry['timestamp'],
                'model_used': conversation_entry['model_used']
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return {
                'response': error_msg,
                'sources': [],
                'conversation_id': None,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'error'
            }
    
    def stream_response(self, question: str, session_id: str = "default") -> Iterator[Dict[str, Any]]:
        """
        Generate streaming response using RAG
        Yields: Dict with 'content', 'sources', 'finished' keys
        """
        if not self.llm:
            yield {
                'content': "Chat engine not properly initialized. Please check your OpenAI API key.",
                'sources': [],
                'finished': True
            }
            return
        
        try:
            # Get fresh vector store instance
            vector_store = get_vector_store()
            
            # Retrieve relevant documents
            relevant_docs = vector_store.search(
                question,
                top_k=config.get('top_k_results', 5)
            )
            
            # Build context from retrieved documents
            context = self._build_context(relevant_docs)
            
            # Create conversation messages
            messages = self._build_messages(question, context, session_id)
            
            # Stream response
            full_response = ""
            sources = self._format_sources(relevant_docs)
            
            for chunk in self.llm.stream(messages):
                content = chunk.content
                full_response += content
                
                yield {
                    'content': content,
                    'sources': sources,
                    'finished': False
                }
            
            # Final yield with complete response
            conversation_entry = {
                'id': self._generate_conversation_id(),
                'session_id': session_id,
                'question': question,
                'response': full_response,
                'sources': sources,
                'timestamp': datetime.now().isoformat(),
                'model_used': config.get('chat_model', 'gpt-4o-mini'),
                'context_used': len(relevant_docs) > 0
            }
            
            self._add_to_history(conversation_entry)
            
            yield {
                'content': '',
                'sources': sources,
                'finished': True,
                'conversation_id': conversation_entry['id']
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            yield {
                'content': error_msg,
                'sources': [],
                'finished': True
            }
    
    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Build context string from relevant documents"""
        if not relevant_docs:
            return "No relevant context found in the indexed content."
        
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            content = doc['content']
            metadata = doc['metadata']
            title = metadata.get('title', 'Unknown Title')
            url = metadata.get('url', 'Unknown URL')
            
            context_parts.append(f"Source {i+1} - {title}:\n{content}\nURL: {url}\n")
        
        return "\n---\n".join(context_parts)
    
    def _build_messages(self, question: str, context: str, session_id: str) -> List:
        """Build conversation messages for the LLM"""
        # System message with instructions
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context from web content. 

INSTRUCTIONS:
1. Use the provided context to answer the user's question accurately and comprehensively
2. If the context doesn't contain relevant information, say so clearly
3. Always cite your sources by mentioning the source title or URL when referencing information
4. Provide specific, detailed answers when possible
5. If asked about something not in the context, acknowledge the limitation
6. Use markdown formatting for better readability (headings, lists, code blocks, etc.)
7. Be conversational and helpful while maintaining accuracy

CONTEXT:
{context}

Remember to cite sources and use markdown formatting in your response."""

        messages = [
            SystemMessage(content=system_prompt.format(context=context))
        ]
        
        # Add recent conversation history for context
        recent_history = self._get_recent_history(session_id, limit=5)
        for entry in recent_history:
            messages.append(HumanMessage(content=entry['question']))
            messages.append(AIMessage(content=entry['response']))
        
        # Add current question
        messages.append(HumanMessage(content=question))
        
        return messages
    
    def _format_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for display"""
        sources = []
        seen_urls = set()
        
        for doc in relevant_docs:
            metadata = doc['metadata']
            url = metadata.get('url', '')
            
            if url and url not in seen_urls:
                sources.append({
                    'title': metadata.get('title', 'Unknown Title'),
                    'url': url,
                    'domain': metadata.get('domain', ''),
                    'relevance_score': round(doc.get('score', 0), 3),
                    'snippet': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                })
                seen_urls.add(url)
        
        return sources
    
    def _get_recent_history(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history for context"""
        session_history = [
            entry for entry in self.conversation_history 
            if entry.get('session_id') == session_id
        ]
        
        # Return most recent conversations
        return session_history[-limit:]
    
    def _add_to_history(self, conversation_entry: Dict[str, Any]):
        """Add conversation to history"""
        self.conversation_history.append(conversation_entry)
        
        # Keep only recent history (last 100 conversations)
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
        
        self._save_chat_history()
    
    def get_chat_history(self, session_id: str = None) -> List[Dict[str, Any]]:
        """Get chat history, optionally filtered by session"""
        if session_id:
            return [
                entry for entry in self.conversation_history 
                if entry.get('session_id') == session_id
            ]
        return self.conversation_history
    
    def clear_history(self, session_id: str = None):
        """Clear chat history"""
        if session_id:
            self.conversation_history = [
                entry for entry in self.conversation_history 
                if entry.get('session_id') != session_id
            ]
        else:
            self.conversation_history = []
        
        self._save_chat_history()
    
    def export_history(self, session_id: str = None, format: str = 'json') -> str:
        """Export chat history in specified format"""
        history = self.get_chat_history(session_id)
        
        if format == 'json':
            return json.dumps(history, indent=2, ensure_ascii=False)
        
        elif format == 'markdown':
            md_content = "# Chat History\n\n"
            for entry in history:
                timestamp = entry.get('timestamp', '')
                question = entry.get('question', '')
                response = entry.get('response', '')
                sources = entry.get('sources', [])
                
                md_content += f"## {timestamp}\n\n"
                md_content += f"**Question:** {question}\n\n"
                md_content += f"**Response:** {response}\n\n"
                
                if sources:
                    md_content += "**Sources:**\n"
                    for source in sources:
                        md_content += f"- [{source['title']}]({source['url']})\n"
                    md_content += "\n"
                
                md_content += "---\n\n"
            
            return md_content
        
        else:
            return json.dumps(history, indent=2, ensure_ascii=False)
    
    def update_model(self, model_name: str, temperature: float = None) -> bool:
        """Update chat model configuration"""
        try:
            config.set('chat_model', model_name)
            if temperature is not None:
                config.set('temperature', temperature)
            
            # Reinitialize LLM
            self._initialize_llm()
            return True
            
        except Exception as e:
            print(f"Error updating model: {e}")
            return False
    
    def suggest_followup_questions(self, last_response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Generate follow-up questions based on the response and sources"""
        # Simple rule-based follow-up suggestions
        suggestions = []
        
        # If sources are available, suggest diving deeper
        if sources:
            unique_domains = set(source['domain'] for source in sources)
            for domain in list(unique_domains)[:2]:
                suggestions.append(f"Tell me more about information from {domain}")
        
        # Generic follow-ups based on response content
        if "benefits" in last_response.lower():
            suggestions.append("What are the potential drawbacks or limitations?")
        
        if "how" in last_response.lower():
            suggestions.append("Can you provide a specific example?")
        
        if any(word in last_response.lower() for word in ["technology", "tool", "software"]):
            suggestions.append("What are the alternatives to this approach?")
        
        # Limit to 3 suggestions
        return suggestions[:3]
    
    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _load_chat_history(self) -> List[Dict[str, Any]]:
        """Load chat history from file"""
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading chat history: {e}")
        
        return []
    
    def _save_chat_history(self):
        """Save chat history to file"""
        try:
            os.makedirs(os.path.dirname(self.chat_history_file), exist_ok=True)
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {e}")

# Remove global instance - create fresh instances when needed
def get_chat_engine():
    """Get a fresh chat engine instance with current config"""
    return ChatEngine()
