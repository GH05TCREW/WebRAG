"""Text processing utilities for WebRAG 2.0"""
import re
import html2text
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """Handle text cleaning and chunking operations"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
    
    def html_to_text(self, html_content: str) -> str:
        """Convert HTML to clean text"""
        try:
            # Convert HTML to markdown-like text
            text = self.html_converter.handle(html_content)
            
            # Clean up the text
            text = self._clean_text(text)
            
            return text
        except Exception as e:
            print(f"Error converting HTML to text: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove common unwanted patterns
        patterns_to_remove = [
            r'Cookie[s]? (?:Policy|Notice|Settings)',
            r'Privacy Policy',
            r'Terms (?:of )?(?:Service|Use)',
            r'Accept (?:All )?Cookies',
            r'Manage Cookies',
            r'(?:Share|Follow) (?:on|us on) (?:Twitter|Facebook|LinkedIn|Instagram)',
            r'Subscribe to our newsletter',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up navigation and menu items
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely navigation
            if len(line) < 3:
                continue
            # Skip lines that are just symbols or numbers
            if re.match(r'^[\W\d]*$', line):
                continue
            # Skip common navigation patterns
            if re.match(r'^(Home|About|Contact|Menu|Search|Login|Sign up)$', line, re.IGNORECASE):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        try:
            if not text or len(text.strip()) == 0:
                return []
            
            chunks = self.text_splitter.split_text(text)
            
            # Filter out very short chunks
            filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
            
            return filtered_chunks
        except Exception as e:
            print(f"Error splitting text: {e}")
            return []
    
    def extract_main_content(self, html_content: str) -> str:
        """Extract main content from HTML, avoiding navigation and footer content"""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements first
            unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement', 'iframe', 'object', 'embed']
            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove elements with common unwanted classes/ids (more comprehensive)
            unwanted_patterns = [
                'nav', 'navigation', 'menu', 'sidebar', 'footer', 'header',
                'advertisement', 'ad-', 'ads', 'banner', 'cookie', 'popup', 'modal', 
                'share', 'social', 'comment', 'related', 'recommended', 'promo',
                'subscribe', 'newsletter', 'signup', 'login', 'search-box'
            ]
            
            for pattern in unwanted_patterns:
                # Remove by class (partial match)
                for element in soup.find_all(attrs={'class': re.compile(pattern, re.I)}):
                    element.decompose()
                # Remove by id (partial match)
                for element in soup.find_all(attrs={'id': re.compile(pattern, re.I)}):
                    element.decompose()
            
            # Strategy 1: Try to find main content containers (most specific first)
            content_selectors = [
                # Wikipedia specific
                '#mw-content-text',
                '.mw-parser-output',
                '#content',
                
                # GitHub specific
                '.repository-content',
                '.markdown-body',
                '.js-code-nav-container',
                
                # Generic semantic HTML5
                'main',
                'article',
                '[role="main"]',
                
                # Common content classes
                '.main-content',
                '.content',
                '.post-content',
                '.entry-content',
                '.article-content',
                '.page-content',
                '#main-content',
                '.container .content'
            ]
            
            main_content = None
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Take the first match that has substantial content
                    for element in elements:
                        text_length = len(element.get_text().strip())
                        if text_length > 500:  # Must have substantial content
                            main_content = element
                            break
                    if main_content:
                        break
            
            # Strategy 2: If no main content found, find the largest text container
            if not main_content:
                potential_containers = soup.find_all(['div', 'section', 'article', 'main'])
                best_container = None
                max_text_length = 0
                
                for container in potential_containers:
                    # Skip if it's likely navigation or sidebar
                    classes = container.get('class', [])
                    element_id = container.get('id', '')
                    
                    # Skip elements that are likely not main content
                    skip_patterns = ['nav', 'sidebar', 'menu', 'footer', 'header', 'ad']
                    if any(pattern in str(classes).lower() or pattern in element_id.lower() for pattern in skip_patterns):
                        continue
                    
                    text = container.get_text().strip()
                    text_length = len(text)
                    
                    if text_length > max_text_length and text_length > 200:
                        max_text_length = text_length
                        best_container = container
                
                main_content = best_container
            
            # Strategy 3: Fallback to body
            if not main_content:
                main_content = soup.find('body') or soup
            
            if main_content:
                # Convert to text using our HTML converter
                text = self.html_to_text(str(main_content))
                
                # Additional cleaning for extracted text
                if len(text.strip()) < 100:
                    # If we still don't have enough content, try a more aggressive approach
                    # Get all paragraph text
                    paragraphs = soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    all_text = []
                    for p in paragraphs:
                        p_text = p.get_text().strip()
                        if len(p_text) > 20:  # Only substantial paragraphs
                            all_text.append(p_text)
                    
                    text = '\n\n'.join(all_text)
                
                return text
            
            return ""
            
        except Exception as e:
            print(f"Error extracting main content: {e}")
            # Fallback: try simple text extraction
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text()
            except:
                return ""

# Global text processor instance
text_processor = TextProcessor()
