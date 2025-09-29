"""
Enhanced web scraper with PDF support using MarkItDown and aiohttp client.
Supports HTML/PDF content extraction with rate limiting, retries, and proxy support.
"""

import re
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Union, Optional, Set, Tuple
from collections import Counter
import io

# PDF processing with markitdown (Microsoft library)
try:
    import markitdown
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# HTTP client with aiohttp
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout, ClientError
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# HTML parsing
try:
    from bs4 import BeautifulSoup, Tag
    BEAUTIFUL_SOUP_AVAILABLE = True
except ImportError:
    BEAUTIFUL_SOUP_AVAILABLE = False


@dataclass
class ScrapingConfig:
    """
    Configuration class for web scraping settings.
    
    @param timeout: Request timeout in seconds
    @param max_retries: Maximum number of retry attempts
    @param retry_delay: Base delay between retries in seconds
    @param rate_limit_delay: Minimum delay between requests in seconds
    @param follow_redirects: Whether to follow HTTP redirects
    @param max_redirects: Maximum number of redirects to follow
    @param user_agent: User-Agent string for requests
    @param headers: Additional HTTP headers
    @param respect_robots_txt: Whether to respect robots.txt (not implemented)
    @param max_content_length: Maximum content size in bytes (10MB default)
    @param enable_pdf_extraction: Whether to enable PDF processing
    @param pdf_max_pages: Maximum number of PDF pages to process
    """
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 1.0
    follow_redirects: bool = True
    max_redirects: int = 10
    user_agent: str = "Mozilla/5.0 (compatible; WebScraper/1.0)"
    headers: Dict[str, str] = field(default_factory=dict)
    respect_robots_txt: bool = False
    max_content_length: int = 10 * 1024 * 1024  # 10MB
    enable_pdf_extraction: bool = True
    pdf_max_pages: int = 50  # Limit PDF pages for performance


class WebScraperError(Exception):
    """Custom exception for web scraper errors."""
    pass


class RateLimiter:
    """
    Simple rate limiter to avoid overwhelming servers.
    
    @param delay: Minimum delay between requests in seconds
    """

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.last_request = 0.0

    async def wait(self):
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last_request = time.time()


class PDFProcessor:
    """Handle PDF content extraction using Microsoft's MarkItDown library."""

    @staticmethod
    def extract_text_markitdown(pdf_content: bytes, max_pages: int = 50) -> str:
        """
        Extract text from PDF using Microsoft's MarkItDown library.
        
        @param pdf_content: PDF file content as bytes
        @param max_pages: Maximum number of pages to process
        @return: Extracted text content
        @throws WebScraperError: If PDF processing fails
        """
        try:
            # Create MarkItDown processor instance
            processor = markitdown.MarkItDown()
            
            # Convert PDF bytes to file-like object
            pdf_file = io.BytesIO(pdf_content)
            
            # Process PDF with page limit
            result = processor.convert(
                pdf_file,
                max_pages=max_pages,
                output_format='text'
            )
            
            return result.text_content if hasattr(result, 'text_content') else str(result)
            
        except Exception as e:
            raise WebScraperError(f"MarkItDown PDF extraction failed: {e}")

    @classmethod
    def extract_text(cls, pdf_content: bytes, max_pages: int = 50) -> str:
        """
        Extract text from PDF using available PDF processor.
        
        @param pdf_content: PDF file content as bytes
        @param max_pages: Maximum number of pages to process
        @return: Extracted text content or error message
        """
        if not PDF_AVAILABLE:
            return "PDF text extraction failed - MarkItDown not available"
            
        try:
            return cls.extract_text_markitdown(pdf_content, max_pages)
        except WebScraperError as e:
            logging.error(f"PDF extraction error: {e}")
            return f"PDF text extraction failed: {str(e)}"


class EnhancedContentExtractor:
    """
    Enhanced content extractor for identifying main content areas in HTML.
    
    Uses semantic analysis and content scoring to identify primary content.
    """

    # Content indicators - tags that typically contain main content
    CONTENT_TAGS = ['article', 'main', 'section', 'div', 'p']

    # Semantic selectors for main content
    MAIN_CONTENT_SELECTORS = [
        'article', 'main', '[role="main"]', '.main', '#main',
        '.content', '#content', '.post', '.article', '.entry',
        '.post-content', '.article-content', '.entry-content',
        '.story', '.story-body', '.article-body', '.post-body'
    ]

    # Noise indicators - classes/ids that typically contain non-content
    NOISE_INDICATORS = [
        'nav', 'navigation', 'menu', 'sidebar', 'aside', 'footer',
        'header', 'ad', 'ads', 'advertisement', 'social', 'share',
        'comment', 'related', 'recommended', 'widget', 'popup',
        'modal', 'overlay', 'banner', 'promo', 'promotion'
    ]

    @staticmethod
    def _calculate_text_density(element: Tag) -> float:
        """
        Calculate text density (text length / HTML length ratio).
        
        @param element: BeautifulSoup element to analyze
        @return: Text density ratio (0.0-1.0)
        """
        if not element:
            return 0.0

        text_length = len(element.get_text(strip=True))
        html_length = len(str(element))

        if html_length == 0:
            return 0.0

        return text_length / html_length

    @staticmethod
    def _count_paragraphs(element: Tag) -> int:
        """
        Count paragraph tags within element.
        
        @param element: BeautifulSoup element to analyze
        @return: Number of paragraph tags
        """
        return len(element.find_all('p'))

    @staticmethod
    def _has_noise_indicators(element: Tag) -> bool:
        """
        Check if element has noise indicators in class or id.
        
        @param element: BeautifulSoup element to check
        @return: True if element contains noise indicators
        """
        classes = element.get('class', [])
        element_id = element.get('id', '')

        text_to_check = ' '.join(classes) + ' ' + element_id
        text_to_check = text_to_check.lower()

        return any(noise in text_to_check for noise in EnhancedContentExtractor.NOISE_INDICATORS)

    @staticmethod
    def _score_content_element(element: Tag) -> float:
        """
        Score an element based on content likelihood.
        
        @param element: BeautifulSoup element to score
        @return: Content likelihood score (higher = more likely to be main content)
        """
        score = 0.0

        # Base score for text density
        text_density = EnhancedContentExtractor._calculate_text_density(element)
        score += text_density * 10

        # Bonus for paragraph count
        para_count = EnhancedContentExtractor._count_paragraphs(element)
        score += min(para_count * 0.5, 5)  # Cap at 5 points

        # Bonus for semantic tags
        if element.name in ['article', 'main']:
            score += 15
        elif element.name == 'section':
            score += 10
        elif element.name == 'div':
            score += 2

        # Bonus for content-indicating classes/ids
        classes = element.get('class', [])
        element_id = element.get('id', '')
        text_to_check = ' '.join(classes) + ' ' + element_id
        text_to_check = text_to_check.lower()

        content_indicators = ['content', 'article', 'post', 'story', 'main', 'body', 'entry']
        for indicator in content_indicators:
            if indicator in text_to_check:
                score += 5
                break

        # Penalty for noise indicators
        if EnhancedContentExtractor._has_noise_indicators(element):
            score -= 10

        # Penalty for very short text
        text_length = len(element.get_text(strip=True))
        if text_length < 100:
            score -= 5

        return score


class WebScraper:
    """
    An enhanced tool for efficiently scraping static content from web pages and PDFs.
    
    Features: rate limiting, retries, configurable timeouts, better error handling,
    content filtering, PDF extraction, and structured data extraction with proxy support.
    
    @dependencies
    - aiohttp: Async HTTP client for web requests
    - markitdown: Microsoft library for PDF text extraction
    - beautifulsoup4: HTML parsing and content extraction
    - lxml: HTML parser backend for BeautifulSoup
    
    @notes
    - All HTTP requests are rate-limited and include retry logic
    - PDF processing requires markitdown library installation
    - Proxy support includes SOCKS and HTTP proxies via aiohttp
    - Content extraction includes semantic analysis for main content identification
    """

    def __init__(self, config: Optional[ScrapingConfig] = None):
        """
        Initialize the web scraper with configuration.
        
        @param config: Scraping configuration settings
        @throws WebScraperError: If required dependencies are missing
        """
        if not AIOHTTP_AVAILABLE:
            raise WebScraperError("aiohttp library is required but not available")
        if not BEAUTIFUL_SOUP_AVAILABLE:
            raise WebScraperError("beautifulsoup4 library is required but not available")
            
        self.config = config or ScrapingConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_delay)
        self.session_cache: Dict[str, aiohttp.ClientSession] = {}
        self.logger = logging.getLogger(__name__)

        # Build headers
        self.default_headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            **self.config.headers
        }

        # Check PDF capabilities
        if self.config.enable_pdf_extraction and not PDF_AVAILABLE:
            self.logger.warning(
                "PDF extraction requested but MarkItDown not available. "
                "Install markitdown for PDF support."
            )

    def _create_client_with_proxy(self, proxy_config: Optional[Dict[str, str]] = None) -> aiohttp.ClientSession:
        """
        Create aiohttp client session with proxy configuration.
        
        @param proxy_config: Proxy configuration dictionary
        @return: Configured aiohttp ClientSession
        @throws WebScraperError: If session creation fails
        """
        try:
            # Create timeout configuration
            timeout = ClientTimeout(total=self.config.timeout)
            
            # Build connector arguments
            connector_args = {}
            
            # Handle proxy configuration
            if proxy_config:
                # Extract proxy URL (supports http, https, socks)
                proxy_url = proxy_config.get('http') or proxy_config.get('https') or proxy_config.get('socks')
                if proxy_url:
                    connector_args['proxy'] = proxy_url
            
            # Create session with configuration
            session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.default_headers,
                **connector_args
            )
            
            return session
            
        except Exception as e:
            raise WebScraperError(f"Failed to create aiohttp client: {e}")

    @asynccontextmanager
    async def _get_client(self, proxy_config: Optional[Dict[str, str]] = None):
        """
        Context manager for HTTP client with proper cleanup and proxy support.
        
        @param proxy_config: Proxy configuration dictionary
        @yields: aiohttp ClientSession
        @throws WebScraperError: If client creation fails
        """
        client_key = str(proxy_config) if proxy_config else "default"
        
        if client_key not in self.session_cache:
            self.session_cache[client_key] = self._create_client_with_proxy(proxy_config)
            
        client = self.session_cache[client_key]
        
        try:
            yield client
        except Exception as e:
            self.logger.error(f"Error using client: {e}")
            raise
        # Note: We don't close the session here to allow reuse

    async def _close_sessions(self):
        """Close all cached client sessions."""
        for client in self.session_cache.values():
            await client.close()
        self.session_cache.clear()

    def _is_pdf_url(self, url: str) -> bool:
        """
        Check if URL likely points to a PDF.
        
        @param url: URL to check
        @return: True if URL appears to be a PDF
        """
        return url.lower().endswith('.pdf') or '.pdf' in url.lower()

    def _is_pdf_content(self, content_type: str, content: bytes) -> bool:
        """
        Check if content is PDF based on content-type or content signature.
        
        @param content_type: HTTP Content-Type header value
        @param content: Response content bytes
        @return: True if content appears to be PDF
        """
        if content_type and 'application/pdf' in content_type.lower():
            return True

        # Check PDF signature
        if content and content.startswith(b'%PDF-'):
            return True

        return False

    async def _fetch_content_with_retries(self, url: str, proxy_config: Optional[Dict[str, str]] = None) -> Optional[tuple]:
        """
        Fetch content with retry logic, rate limiting, and proxy support.
        
        @param url: URL to fetch
        @param proxy_config: Proxy configuration dictionary
        @return: Tuple of (content, content_type, is_pdf) or None if all retries fail
        @throws WebScraperError: If request fails after all retries
        """
        await self.rate_limiter.wait()

        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._get_client(proxy_config) as client:
                    async with client.get(url, allow_redirects=self.config.follow_redirects) as response:
                        # Check status code
                        if response.status >= 400:
                            raise WebScraperError(f"HTTP {response.status}: {response.reason}")

                        # Check content length
                        content_length = response.headers.get('content-length')
                        if content_length and int(content_length) > self.config.max_content_length:
                            raise WebScraperError(f"Content too large: {content_length} bytes")

                        # Get content type and content
                        content_type = response.headers.get('content-type', '').lower()
                        content = await response.read()

                        # Check if it's PDF
                        is_pdf = self._is_pdf_content(content_type, content)

                        if is_pdf:
                            if not self.config.enable_pdf_extraction:
                                raise WebScraperError("PDF content detected but PDF extraction is disabled")
                            if not PDF_AVAILABLE:
                                raise WebScraperError("PDF content detected but no PDF library is available")
                            return content, content_type, True

                        # Handle HTML/XML content
                        if any(ct in content_type for ct in ['text/html', 'application/xhtml', 'application/xml']):
                            text_content = await response.text()
                            return text_content, content_type, False

                        # Handle other text content
                        if content_type.startswith('text/'):
                            text_content = await response.text()
                            return text_content, content_type, False

                        self.logger.warning(f"Unexpected content type: {content_type}")
                        # Try to decode as text anyway
                        try:
                            text_content = content.decode('utf-8', errors='ignore')
                            return text_content, content_type, False
                        except:
                            return content, content_type, False

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                self.logger.warning(
                    f"Request error on attempt {attempt + 1}/{self.config.max_retries} for {url}: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except WebScraperError as e:
                last_exception = e
                if '429' in str(e) or '503' in str(e) or '504' in str(e):
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                        continue
                break
            except Exception as e:
                last_exception = e
                self.logger.error(f"Unexpected error on attempt {attempt + 1} for {url}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        if last_exception:
            raise WebScraperError(f"All retries failed for {url}: {last_exception}")
        return None

    def _clean_soup(self, soup: BeautifulSoup, remove_tags: Set[str] = None) -> BeautifulSoup:
        """
        Clean soup by removing unwanted elements.
        
        @param soup: BeautifulSoup object to clean
        @param remove_tags: Set of tag names to remove
        @return: Cleaned BeautifulSoup object
        """
        if remove_tags is None:
            remove_tags = {"script", "style", "meta", "link", "noscript", "iframe"}

        for tag in remove_tags:
            for element in soup.find_all(tag):
                element.decompose()

        return soup

    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """
        Extract and clean text content from soup.
        
        @param soup: BeautifulSoup object to extract text from
        @return: Cleaned text content
        """
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = []
        for line in text.splitlines():
            cleaned_line = ' '.join(line.split())
            if cleaned_line:
                lines.append(cleaned_line)

        return '\n'.join(lines)

    async def scrape_text(self, url: str, clean_text: bool = True, proxy_config: Optional[Dict[str, str]] = None) -> str:
        """
        Fetches a URL and extracts all human-readable text content from it (supports both HTML and PDF).
        
        @param url: The URL of the web page or PDF to scrape
        @param clean_text: Whether to clean and normalize the extracted text
        @param proxy_config: Optional proxy configuration dictionary
        @return: Extracted text content
        @throws WebScraperError: If scraping fails
        """
        try:
            result = await self._fetch_content_with_retries(url, proxy_config)
            if not result:
                return "Could not retrieve the webpage or document."

            content, content_type, is_pdf = result

            if is_pdf:
                # Handle PDF content
                self.logger.info(f"Processing PDF content from {url}")
                if isinstance(content, str):
                    content = content.encode('utf-8')

                text = PDFProcessor.extract_text(content, self.config.pdf_max_pages)

                if clean_text and text:
                    # Basic cleaning for PDF text
                    lines = []
                    for line in text.splitlines():
                        cleaned_line = ' '.join(line.split())
                        if cleaned_line:
                            lines.append(cleaned_line)
                    text = '\n'.join(lines)

                return text if text.strip() else "No text content found in the PDF."

            else:
                # Handle HTML/XML content
                if not isinstance(content, str):
                    content = content.decode('utf-8', errors='ignore')

                soup = BeautifulSoup(content, 'lxml')
                soup = self._clean_soup(soup)

                if clean_text:
                    text = self._extract_clean_text(soup)
                else:
                    text = soup.get_text()

                return text if text.strip() else "No text content found on the page."

        except Exception as e:
            self.logger.error(f"Error parsing content from {url}: {e}")
            return f"Error parsing document content: {str(e)}"

    async def scrape_links(self, url: str, filter_domains: Optional[Set[str]] = None,
                           include_internal_only: bool = False, proxy_config: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Fetches a URL and extracts all hyperlinks from it (HTML only).
        
        @param url: The URL of the web page to scrape
        @param filter_domains: Set of domains to include (if None, include all)
        @param include_internal_only: If True, only include links from the same domain
        @param proxy_config: Optional proxy configuration dictionary
        @return: List of dictionaries containing link information
        @throws WebScraperError: If scraping fails
        """
        try:
            result = await self._fetch_content_with_retries(url, proxy_config)
            if not result:
                return [{"error": "Could not retrieve the webpage."}]

            content, content_type, is_pdf = result

            if is_pdf:
                return [{"error": "Cannot extract links from PDF content. Links extraction only works with HTML pages."}]

            if not isinstance(content, str):
                content = content.decode('utf-8', errors='ignore')

            soup = BeautifulSoup(content, 'lxml')
            base_domain = urlparse(url).netloc
            links = []

            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                if not href or href.startswith('#'):
                    continue

                absolute_link = urljoin(url, href)
                link_domain = urlparse(absolute_link).netloc

                if include_internal_only and link_domain != base_domain:
                    continue
                if filter_domains and link_domain not in filter_domains:
                    continue

                link_info = {
                    'url': absolute_link,
                    'text': a_tag.get_text(strip=True),
                    'title': a_tag.get('title', ''),
                    'domain': link_domain,
                    'is_internal': link_domain == base_domain
                }

                link_info = {k: v for k, v in link_info.items() if v or k == 'is_internal'}
                links.append(link_info)

            return links if links else [{"message": "No links found on the page."}]

        except Exception as e:
            self.logger.error(f"Error parsing links from {url}: {e}")
            return [{"error": f"Error parsing webpage links: {str(e)}"}]

    async def scrape_with_selector(self, url: str, selector: str,
                                   extract_attributes: Optional[List[str]] = None,
                                   proxy_config: Optional[Dict[str, str]] = None) -> List[Dict[str, Union[str, Dict]]]:
        """
        Fetches a URL and extracts data using CSS selectors (HTML only).
        
        @param url: The URL of the web page to scrape
        @param selector: The CSS selector to find elements
        @param extract_attributes: List of attributes to extract from elements
        @param proxy_config: Optional proxy configuration dictionary
        @return: List of dictionaries containing extracted data
        @throws WebScraperError: If scraping fails
        """
        try:
            result = await self._fetch_content_with_retries(url, proxy_config)
            if not result:
                return [{"error": "Could not retrieve the webpage."}]

            content, content_type, is_pdf = result

            if is_pdf:
                return [{"error": "Cannot use CSS selectors on PDF content. Selector extraction only works with HTML pages."}]

            if not isinstance(content, str):
                content = content.decode('utf-8', errors='ignore')

            soup = BeautifulSoup(content, 'lxml')
            elements = soup.select(selector)
            results = []

            if extract_attributes is None:
                extract_attributes = ['href', 'src', 'alt', 'title', 'class', 'id']

            for i, el in enumerate(elements):
                result = {
                    'index': i,
                    'tag': el.name,
                    'text': el.get_text(strip=True),
                }

                attributes = {}
                for attr in extract_attributes:
                    if el.has_attr(attr):
                        attr_value = el.get(attr)
                        if attr in ['href', 'src'] and attr_value:
                            attr_value = urljoin(url, attr_value)
                        attributes[attr] = attr_value

                if attributes:
                    result['attributes'] = attributes

                result = {k: v for k, v in result.items() if v is not None and v != ''}
                results.append(result)

            return results if results else [{"message": f"No elements found with selector '{selector}'."}]

        except Exception as e:
            self.logger.error(f"Error parsing selector '{selector}' from {url}: {e}")
            return [{"error": f"Error parsing webpage with selector: {str(e)}"}]

    async def scrape_structured_data(self, url: str, proxy_config: Optional[Dict[str, str]] = None) -> Dict[str, Union[str, List, Dict]]:
        """
        Extract structured data including metadata, headings, images, main content, and articles.
        Enhanced to identify and extract main content areas and articles.
        
        @param url: The URL of the web page or PDF to scrape
        @param proxy_config: Optional proxy configuration dictionary
        @return: Dictionary containing structured page data with enhanced content extraction
        @throws WebScraperError: If scraping fails
        """
        try:
            result = await self._fetch_content_with_retries(url, proxy_config)
            if not result:
                return {"error": "Could not retrieve valid document content."}

            content, content_type, is_pdf = result

            if is_pdf:
                # Handle PDF structured data
                self.logger.info(f"Processing PDF structured data from {url}")
                if isinstance(content, str):
                    content = content.encode('utf-8')

                text = PDFProcessor.extract_text(content, self.config.pdf_max_pages)

                data = {
                    'url': url,
                    'content_type': 'application/pdf',
                    'title': url.split('/')[-1],
                    'is_pdf': True,
                    'text_length': len(text) if text else 0,
                    'word_count': len(text.split()) if text else 0,
                    'extraction_method': 'PDF processor',
                    'main_content': text[:5000] if text else '',
                }

                return data

            else:
                # Handle HTML structured data with enhanced content extraction
                if not isinstance(content, str):
                    content = content.decode('utf-8', errors='ignore')

                try:
                    soup = BeautifulSoup(content, 'lxml')
                except Exception:
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                    except Exception:
                        return {"error": "Failed to parse HTML with any available parser."}

                data = {
                    'url': url,
                    'content_type': content_type,
                    'title': '',
                    'meta': {},
                    'headings': {},
                    'images': [],
                    'links_count': 0,
                    'word_count': 0,
                    'is_pdf': False,
                    'articles': [],
                    'main_content': '',
                    'content_blocks': [],
                    'extracted_text': ''
                }

                # Extract title
                try:
                    title_tag = soup.find('title')
                    if title_tag:
                        data['title'] = title_tag.get_text(strip=True)
                except Exception as e:
                    self.logger.warning(f"Error extracting title from {url}: {e}")

                # Extract meta tags
                try:
                    for meta in soup.find_all('meta'):
                        try:
                            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
                            content_attr = meta.get('content')
                            if name and content_attr:
                                data['meta'][name] = content_attr
                        except Exception:
                            continue
                except Exception as e:
                    self.logger.warning(f"Error extracting meta tags from {url}: {e}")

                # Extract headings
                try:
                    for level in range(1, 7):
                        try:
                            headings = soup.find_all(f'h{level}')
                            if headings:
                                heading_texts = []
                                for h in headings:
                                    try:
                                        text = h.get_text(strip=True)
                                        if text:
                                            heading_texts.append(text)
                                    except Exception:
                                        continue
                                if heading_texts:
                                    data['headings'][f'h{level}'] = heading_texts
                        except Exception:
                            continue
                except Exception as e:
                    self.logger.warning(f"Error extracting headings from {url}: {e}")

                # Extract images
                try:
                    for img in soup.find_all('img'):
                        try:
                            img_data = {}
                            src = img.get('src')
                            if src:
                                img_data['src'] = urljoin(url, src)
                            alt = img.get('alt')
                            if alt:
                                img_data['alt'] = alt
                            title = img.get('title')
                            if title:
                                img_data['title'] = title
                            if img_data:
                                data['images'].append(img_data)
                        except Exception:
                            continue
                except Exception as e:
                    self.logger.warning(f"Error extracting images from {url}: {e}")

                # Count links
                try:
                    links = soup.find_all('a', href=True)
                    data['links_count'] = len(links) if links else 0
                except Exception as e:
                    self.logger.warning(f"Error counting links from {url}: {e}")
                    data['links_count'] = 0

                # Enhanced content extraction
                try:
                    # Extract articles using semantic HTML5 elements
                    articles = soup.find_all('article')
                    for i, article in enumerate(articles):
                        try:
                            article_data = {
                                'index': i,
                                'text': article.get_text(strip=True),
                                'word_count': len(article.get_text(strip=True).split()),
                                'html': str(article)[:1000]  # First 1000 chars of HTML
                            }

                            # Try to find article title
                            title_elem = article.find(['h1', 'h2', 'h3', '.title', '.headline'])
                            if title_elem:
                                article_data['title'] = title_elem.get_text(strip=True)

                            data['articles'].append(article_data)
                        except Exception as e:
                            self.logger.debug(f"Error processing article {i}: {e}")

                    # Find main content using multiple strategies
                    main_content_candidates = []

                    # Strategy 1: Look for semantic selectors
                    for selector in EnhancedContentExtractor.MAIN_CONTENT_SELECTORS:
                        try:
                            elements = soup.select(selector)
                            for elem in elements:
                                if isinstance(elem, Tag):
                                    score = EnhancedContentExtractor._score_content_element(elem)
                                    if score > 5:  # Minimum threshold
                                        main_content_candidates.append({
                                            'element': elem,
                                            'score': score,
                                            'method': f'selector: {selector}'
                                        })
                        except Exception:
                            continue

                    # Strategy 2: Score all div and section elements
                    for elem in soup.find_all(['div', 'section']):
                        try:
                            score = EnhancedContentExtractor._score_content_element(elem)
                            if score > 8:  # Higher threshold for generic elements
                                main_content_candidates.append({
                                    'element': elem,
                                    'score': score,
                                    'method': 'content_scoring'
                                })
                        except Exception:
                            continue

                    # Select best main content candidate
                    if main_content_candidates:
                        best_candidate = max(main_content_candidates, key=lambda x: x['score'])
                        main_elem = best_candidate['element']

                        data['main_content'] = main_elem.get_text(strip=True)
                        data['main_content_method'] = best_candidate['method']
                        data['main_content_score'] = best_candidate['score']

                        # Extract content blocks (paragraphs from main content)
                        paragraphs = main_elem.find_all('p')
                        for i, p in enumerate(paragraphs):
                            p_text = p.get_text(strip=True)
                            if len(p_text) > 50:  # Only substantial paragraphs
                                data['content_blocks'].append({
                                    'index': i,
                                    'text': p_text,
                                    'word_count': len(p_text.split())
                                })

                    # Fallback: Extract all substantial text blocks
                    if not data['main_content']:
                        self.logger.info("Using fallback text extraction method")
                        all_text_blocks = []

                        for elem in soup.find_all(['p', 'div']):
                            if not EnhancedContentExtractor._has_noise_indicators(elem):
                                text = elem.get_text(strip=True)
                                if len(text) > 100:  # Substantial text blocks
                                    all_text_blocks.append(text)

                        if all_text_blocks:
                            data['main_content'] = ' '.join(all_text_blocks[:5])  # Top 5 blocks
                            data['main_content_method'] = 'fallback_text_blocks'

                    # Extract all readable text for full text search
                    try:
                        cleaned_soup = self._clean_soup(soup)
                        if cleaned_soup:
                            full_text = self._extract_clean_text(cleaned_soup)
                            if full_text:
                                data['extracted_text'] = full_text
                                data['word_count'] = len(full_text.split())
                    except Exception as e:
                        self.logger.warning(f"Error extracting full text from {url}: {e}")
                        data['word_count'] = 0

                except Exception as e:
                    self.logger.error(f"Error in enhanced content extraction from {url}: {e}")

                return data

        except Exception as e:
            self.logger.error(f"Error extracting structured data from {url}: {e}")
            return {"error": f"Error parsing structured data: {str(e)}"}

    async def batch_scrape(self, urls: List[str], method: str = 'text',
                           max_concurrent: int = 5, proxy_config: Optional[Dict[str, str]] = None, **kwargs) -> Dict[str, Union[str, List, Dict]]:
        """
        Scrape multiple URLs concurrently (supports both HTML and PDF).
        
        @param urls: List of URLs to scrape
        @param method: Scraping method ('text', 'links', 'selector', 'structured')
        @param max_concurrent: Maximum number of concurrent requests
        @param proxy_config: Optional proxy configuration dictionary
        @param kwargs: Additional arguments for the scraping method
        @return: Dictionary mapping URLs to their scraped content
        @throws WebScraperError: If batch scraping fails
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_single(url: str):
            async with semaphore:
                try:
                    if method == 'text':
                        return await self.scrape_text(url, proxy_config=proxy_config, **kwargs)
                    elif method == 'links':
                        return await self.scrape_links(url, proxy_config=proxy_config, **kwargs)
                    elif method == 'selector':
                        return await self.scrape_with_selector(url, proxy_config=proxy_config, **kwargs)
                    elif method == 'structured':
                        return await self.scrape_structured_data(url, proxy_config=proxy_config)
                    else:
                        return {"error": f"Unknown scraping method: {method}"}
                except Exception as e:
                    return {"error": f"Error scraping {url}: {str(e)}"}

        tasks = [scrape_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return dict(zip(urls, results))

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self._close_sessions()


# Example usage and configuration
if __name__ == "__main__":
    async def main():
        # Custom configuration with PDF support
        config = ScrapingConfig(
            timeout=15.0,
            max_retries=2,
            rate_limit_delay=0.5,
            user_agent="MyBot/1.0",
            enable_pdf_extraction=True,
            pdf_max_pages=20
        )

        async with WebScraper(config) as scraper:
            # Test HTML scraping
            html_url = "https://www.iea.org/reports/world-energy-investment-2023/overview-and-key-findings"
            print("=== HTML Scraping ===")
            text = await scraper.scrape_text(html_url)
            print(f"HTML Text length: {len(text)}")
            # print(text[:500] + "..." if len(text) > 500 else text)
            print(text)

            # Test structured data with both types
            print("\n=== Structured Data ===")
            html_metadata = await scraper.scrape_structured_data(html_url)
            print(f"HTML metadata: {html_metadata}")

    # Run the example
    if PDF_AVAILABLE:
        print("MarkItDown PDF processing library available")
    else:
        print("MarkItDown not found. Install markitdown for PDF support.")
        
    if AIOHTTP_AVAILABLE:
        print("aiohttp library available")
    else:
        print("aiohttp not found. Install aiohttp for HTTP requests.")

    asyncio.run(main())