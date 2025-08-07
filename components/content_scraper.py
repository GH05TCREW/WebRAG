"""Content scraping component for WebRAG 2.0"""
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List
import time
from datetime import datetime
import os
import json
from utils.helpers import get_domain, safe_filename, generate_content_hash
from utils.text_processing import text_processor

class ContentScraper:
    """Enhanced web content scraper with smart extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.cache_dir = "data/raw_content"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a single URL
        Returns: {
            'url': str,
            'title': str,
            'content': str,
            'metadata': dict,
            'timestamp': datetime,
            'hash': str
        }
        """
        try:
            # Check cache first
            cached_content = self._get_cached_content(url)
            if cached_content:
                return cached_content
            
            # Fetch fresh content
            response = self._fetch_with_retries(url)
            if not response:
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content
            title = self._extract_title(soup)
            content = self._extract_main_content(soup, url)
            metadata = self._extract_metadata(soup, response)
            
            # Create result
            result = {
                'url': url,
                'title': title,
                'content': content,
                'metadata': metadata,
                'timestamp': datetime.now(),
                'hash': generate_content_hash(content)
            }
            
            # Cache the result
            self._cache_content(url, result)
            
            return result
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def _fetch_with_retries(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Fetch URL with retries and error handling"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=15, allow_redirects=True)
                
                # Check if it's a valid HTML response
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                    print(f"Skipping non-HTML content: {content_type}")
                    return None
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    return None
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the best possible title from the page"""
        # Priority order for title extraction
        title_selectors = [
            'title',
            'h1',
            '[property="og:title"]',
            '[name="twitter:title"]',
            '.title',
            '.headline',
            'h2'
        ]
        
        for selector in title_selectors:
            elements = soup.select(selector)
            for element in elements:
                if selector.startswith('['):
                    title = element.get('content', '').strip()
                else:
                    title = element.get_text().strip()
                
                if title and len(title) > 3:
                    return title[:200]  # Limit title length
        
        return "Untitled"
    
    def _extract_main_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract main content using multiple strategies"""
        # Remove unwanted elements first
        self._remove_unwanted_elements(soup)
        
        # Strategy 1: Look for main content containers
        main_content = self._find_main_content_container(soup)
        
        if main_content:
            # Use text processor to clean and convert
            return text_processor.extract_main_content(str(main_content))
        
        # Strategy 2: Use the body as fallback
        body = soup.find('body')
        if body:
            return text_processor.extract_main_content(str(body))
        
        # Strategy 3: Use entire soup as last resort
        return text_processor.extract_main_content(str(soup))
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements"""
        # Elements to completely remove
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'iframe', 'object', 'embed', 'form', 'fieldset',
            'button', 'input', 'select', 'textarea'
        ]
        
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with unwanted classes/ids
        unwanted_patterns = [
            'advertisement', 'ad-', 'ads', 'banner', 'popup', 'modal',
            'cookie', 'gdpr', 'newsletter', 'subscription', 'social-share',
            'related-posts', 'comments', 'sidebar', 'widget', 'menu',
            'navigation', 'breadcrumb', 'tags', 'metadata'
        ]
        
        for pattern in unwanted_patterns:
            # Remove by class
            for element in soup.find_all(attrs={'class': lambda x: x and any(pattern in ' '.join(x).lower() for pattern in [pattern])}):
                element.decompose()
            
            # Remove by id
            for element in soup.find_all(attrs={'id': lambda x: x and pattern in x.lower()}):
                element.decompose()
    
    def _find_main_content_container(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content container using various strategies"""
        # Strategy 1: Look for semantic HTML5 elements
        semantic_selectors = ['main', 'article', '[role="main"]']
        
        for selector in semantic_selectors:
            element = soup.select_one(selector)
            if element and self._has_substantial_content(element):
                return element
        
        # Strategy 2: Look for common content class names
        content_selectors = [
            '.main-content', '.content', '.post-content', '.entry-content',
            '.article-content', '.page-content', '#main-content', '#content',
            '.container .content', '.wrapper .content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element and self._has_substantial_content(element):
                return element
        
        # Strategy 3: Find the element with the most text content
        potential_containers = soup.find_all(['div', 'section', 'article'])
        best_container = None
        max_text_length = 0
        
        for container in potential_containers:
            text_length = len(container.get_text().strip())
            if text_length > max_text_length and text_length > 500:  # Minimum content threshold
                max_text_length = text_length
                best_container = container
        
        return best_container
    
    def _has_substantial_content(self, element) -> bool:
        """Check if an element has substantial text content"""
        text = element.get_text().strip()
        return len(text) > 200  # Minimum text length
    
    def _extract_metadata(self, soup: BeautifulSoup, response: requests.Response) -> Dict[str, Any]:
        """Extract metadata from the page"""
        metadata = {
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(response.content),
            'status_code': response.status_code,
            'encoding': response.encoding,
            'description': '',
            'keywords': '',
            'author': '',
            'published_date': '',
            'language': ''
        }
        
        # Extract meta tags
        meta_mappings = {
            'description': ['name="description"', 'property="og:description"', 'name="twitter:description"'],
            'keywords': ['name="keywords"'],
            'author': ['name="author"', 'property="article:author"'],
            'published_date': ['name="date"', 'property="article:published_time"', 'name="pubdate"'],
            'language': ['http-equiv="content-language"', 'name="language"']
        }
        
        for key, selectors in meta_mappings.items():
            for selector in selectors:
                element = soup.select_one(f'meta[{selector}]')
                if element:
                    content = element.get('content', '').strip()
                    if content:
                        metadata[key] = content
                        break
        
        return metadata
    
    def _get_cached_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content if available and fresh"""
        cache_file = os.path.join(self.cache_dir, f"{safe_filename(get_domain(url))}_{generate_content_hash(url)}.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if cache is less than 24 hours old
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    if (datetime.now() - cached_time).total_seconds() < 24 * 3600:
                        data['timestamp'] = cached_time
                        return data
        except Exception as e:
            print(f"Error reading cache: {e}")
        
        return None
    
    def _cache_content(self, url: str, content: Dict[str, Any]):
        """Cache scraped content"""
        cache_file = os.path.join(self.cache_dir, f"{safe_filename(get_domain(url))}_{generate_content_hash(url)}.json")
        
        try:
            # Convert datetime to string for JSON serialization
            cache_data = content.copy()
            cache_data['timestamp'] = content['timestamp'].isoformat()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error caching content: {e}")

# Global content scraper instance
content_scraper = ContentScraper()
