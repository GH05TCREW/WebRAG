"""Utility helper functions for WebRAG 2.0"""
import re
import hashlib
from urllib.parse import urljoin, urlparse
from typing import List, Optional, Tuple
import requests
from datetime import datetime

def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate if a URL is properly formatted and accessible
    Returns: (is_valid, error_message)
    """
    if not url or not url.strip():
        return False, "URL is empty"
    
    url = url.strip()
    
    # Add http:// if no scheme is present
    if not re.match(r'^https?://', url):
        url = 'http://' + url
    
    # Basic URL format validation
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False, "Invalid URL format"
    except Exception:
        return False, "Invalid URL format"
    
    # Check if URL is accessible
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code == 405:  # Method not allowed, try GET
            response = requests.get(url, timeout=10, stream=True)
        
        if response.status_code >= 400:
            return False, f"URL not accessible (HTTP {response.status_code})"
        
        return True, url
    except requests.exceptions.RequestException as e:
        return False, f"Cannot access URL: {str(e)}"

def normalize_url(url: str) -> str:
    """Normalize URL for consistent storage"""
    parsed = urlparse(url)
    # Remove fragment and query parameters for normalization
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if normalized.endswith('/'):
        normalized = normalized[:-1]
    return normalized

def get_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""

def generate_content_hash(content: str) -> str:
    """Generate hash for content to detect duplicates"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from multiline text input"""
    lines = [line.strip() for line in text.split('\n')]
    urls = []
    
    for line in lines:
        if line and not line.startswith('#'):  # Skip comments
            # Handle multiple URLs per line (space or comma separated)
            potential_urls = re.split(r'[,\s]+', line)
            for url in potential_urls:
                url = url.strip()
                if url:
                    urls.append(url)
    
    return urls

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def safe_filename(filename: str) -> str:
    """Create safe filename from URL or title"""
    # Remove or replace invalid characters
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-')

def extract_internal_links(html_content: str, base_url: str) -> List[str]:
    """Extract internal links from HTML content"""
    from bs4 import BeautifulSoup
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        base_domain = get_domain(base_url)
        internal_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Check if it's an internal link
            if get_domain(absolute_url) == base_domain:
                # Skip fragments and certain file types
                parsed = urlparse(absolute_url)
                if not parsed.fragment and not parsed.path.lower().endswith(('.pdf', '.jpg', '.png', '.gif', '.css', '.js')):
                    normalized_url = normalize_url(absolute_url)
                    if normalized_url not in internal_links:
                        internal_links.append(normalized_url)
        
        return internal_links
    
    except Exception as e:
        print(f"Error extracting internal links: {e}")
        return []

def is_valid_content_type(content_type: str) -> bool:
    """Check if content type is suitable for processing"""
    if not content_type:
        return True  # Assume valid if not specified
    
    valid_types = [
        'text/html',
        'text/plain',
        'application/xhtml+xml'
    ]
    
    return any(ct in content_type.lower() for ct in valid_types)

def estimate_reading_time(text: str) -> int:
    """Estimate reading time in minutes (assuming 200 words per minute)"""
    word_count = len(text.split())
    return max(1, round(word_count / 200))

class ProgressTracker:
    """Track progress for long-running operations"""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.current_item = ""
        self.errors = []
    
    def update(self, item: str, increment: int = 1):
        """Update progress"""
        self.current_item = item
        self.completed_items += increment
    
    def add_error(self, error: str):
        """Add error to tracking"""
        self.errors.append(error)
    
    def get_progress(self) -> float:
        """Get progress as percentage"""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100
    
    def is_complete(self) -> bool:
        """Check if operation is complete"""
        return self.completed_items >= self.total_items
