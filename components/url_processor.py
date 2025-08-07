"""URL processing component for WebRAG 2.0"""
import requests
from typing import List, Tuple, Optional, Callable
from datetime import datetime
import time
from urllib.parse import urljoin, urlparse
from utils.helpers import validate_url, normalize_url, get_domain, extract_internal_links, is_valid_content_type, ProgressTracker

class URLProcessor:
    """Process and validate URLs for content extraction"""
    
    def __init__(self, max_pages_per_domain: int = 50, max_depth: int = 2):
        self.max_pages_per_domain = max_pages_per_domain
        self.max_depth = max_depth
        self.session = requests.Session()
        
        # Better headers to avoid bot detection
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        self.processed_urls = set()
        self.domain_counts = {}
    
    def process_urls(self, urls: List[str], progress_callback: Optional[Callable] = None) -> List[Tuple[str, str, str]]:
        """
        Process a list of URLs and return content
        Returns: List of (url, content, title) tuples
        """
        if not urls:
            return []
        
        # Validate and normalize URLs
        valid_urls = []
        for url in urls:
            is_valid, result = validate_url(url)
            if is_valid:
                normalized_url = normalize_url(result)
                if normalized_url not in valid_urls:
                    valid_urls.append(normalized_url)
            else:
                print(f"Invalid URL skipped: {url} - {result}")
                if progress_callback:
                    progress_callback(f"Skipped invalid URL: {url}", 0)
        
        if not valid_urls:
            if progress_callback:
                progress_callback("No valid URLs to process", 100)
            return []
        
        # First, crawl to discover additional URLs if depth > 1
        all_urls_to_process = set(valid_urls)
        
        if self.max_depth > 1:
            for url in valid_urls:
                if progress_callback:
                    progress_callback(f"Discovering pages from {get_domain(url)}...", 0)
                
                crawled_urls = self._crawl_site(url, progress_callback)
                all_urls_to_process.update(crawled_urls)
        
        # Convert back to list for processing
        all_urls_to_process = list(all_urls_to_process)
        
        # Process all discovered URLs
        results = []
        total_urls = len(all_urls_to_process)
        
        if progress_callback:
            progress_callback(f"Processing {total_urls} URL(s)...", 0)
        
        for i, url in enumerate(all_urls_to_process):
            try:
                if progress_callback:
                    progress = (i / total_urls) * 90  # Reserve 10% for final steps
                    progress_callback(f"Fetching content from {get_domain(url)}...", progress)
                
                content, title = self._fetch_content(url)
                if content and len(content.strip()) > 200:  # Lowered threshold from 100 to 50
                    results.append((url, content, title))
                    print(f"✅ Successfully processed: {url} ({len(content)} chars)")
                else:
                    print(f"⚠️ Insufficient content found: {url} (only {len(content.strip()) if content else 0} chars)")
                    # Try alternative extraction method for difficult sites
                    if len(content.strip()) < 200:
                        print(f"🔄 Trying alternative extraction for: {url}")
                        alt_content, alt_title = self._fetch_content_alternative(url)
                        if alt_content and len(alt_content.strip()) > 200:
                            results.append((url, alt_content, alt_title or title))
                            print(f"✅ Alternative extraction succeeded: {url} ({len(alt_content)} chars)")
                        else:
                            print(f"❌ Alternative extraction also failed: {url}")
                
                # Small delay to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                error_msg = f"❌ Error processing {url}: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, (i / total_urls) * 90)
        
        # Final progress update
        if progress_callback:
            if results:
                progress_callback(f"Successfully processed {len(results)} page(s)", 95)
            else:
                progress_callback("No content could be extracted from any URL", 100)
        
        return results
    
    def _crawl_site(self, start_url: str, progress_callback: Optional[Callable] = None) -> List[str]:
        """Crawl a website to discover internal pages"""
        domain = get_domain(start_url)
        urls_to_crawl = [start_url]
        crawled_urls = set()
        discovered_urls = {start_url}
        depth = 0
        
        while urls_to_crawl and depth < self.max_depth:
            current_level_urls = urls_to_crawl.copy()
            urls_to_crawl.clear()
            
            for url in current_level_urls:
                # Check domain limits BEFORE processing
                if len(discovered_urls) >= self.max_pages_per_domain:
                    print(f"🛑 Reached max pages limit ({self.max_pages_per_domain}) for domain {domain}")
                    break
                
                if url in crawled_urls:
                    continue
                
                try:
                    if progress_callback:
                        progress_callback(f"Crawling {domain} (depth {depth + 1})", 0)
                    
                    # Fetch the page
                    response = self._fetch_page(url)
                    if not response:
                        continue
                    
                    crawled_urls.add(url)
                    self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
                    
                    # Extract internal links for next level
                    if depth < self.max_depth - 1 and len(discovered_urls) < self.max_pages_per_domain:
                        internal_links = extract_internal_links(response, url)
                        for link in internal_links:
                            if link not in discovered_urls and len(discovered_urls) < self.max_pages_per_domain:
                                discovered_urls.add(link)
                                urls_to_crawl.append(link)
                    
                    time.sleep(0.5)  # Be respectful
                    
                except Exception as e:
                    print(f"Error crawling {url}: {e}")
            
            # Break if we've reached the limit
            if len(discovered_urls) >= self.max_pages_per_domain:
                break
                
            depth += 1
        
        # Limit the final result to max_pages_per_domain
        final_urls = list(discovered_urls)[:self.max_pages_per_domain]
        if len(final_urls) < len(discovered_urls):
            print(f"📊 Limited to {len(final_urls)} pages (from {len(discovered_urls)} discovered) for domain {domain}")
        
        return final_urls
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not is_valid_content_type(content_type):
                return None
            
            return response.text
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def _fetch_content_alternative(self, url: str) -> Tuple[str, str]:
        """Alternative content extraction method for difficult sites"""
        try:
            html_content = self._fetch_page(url)
            if not html_content:
                return "", ""
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts and styles
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Get all text, then clean it
            raw_text = soup.get_text()
            
            # Clean up whitespace and empty lines
            lines = raw_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if len(line) > 10:  # Only keep substantial lines
                    cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)
            title = self._extract_title(html_content) or get_domain(url)
            
            return content, title
            
        except Exception as e:
            print(f"Error in alternative extraction for {url}: {e}")
            return "", ""
    
    def _fetch_content(self, url: str) -> Tuple[str, str]:
        """Fetch and extract content from URL"""
        try:
            html_content = self._fetch_page(url)
            if not html_content:
                return "", ""
            
            # Extract content using text processor
            from utils.text_processing import text_processor
            content = text_processor.extract_main_content(html_content)
            
            # Extract title
            title = self._extract_title(html_content) or get_domain(url)
            
            return content, title
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return "", ""
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content"""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try different title sources
            title_sources = [
                soup.find('title'),
                soup.find('h1'),
                soup.find('meta', property='og:title'),
                soup.find('meta', name='title')
            ]
            
            for source in title_sources:
                if source:
                    if source.name == 'meta':
                        title = source.get('content', '').strip()
                    else:
                        title = source.get_text().strip()
                    
                    if title:
                        return title
            
            return ""
            
        except Exception:
            return ""
    
    def reset_tracking(self):
        """Reset processing tracking for new session"""
        self.processed_urls.clear()
        self.domain_counts.clear()

# Global URL processor instance
url_processor = URLProcessor()
