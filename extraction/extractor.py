"""
Content extraction from HTML documents.
"""

import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from bs4 import BeautifulSoup, NavigableString, Tag

from ..models import Document
from ..config import ExtractionConfig

logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    Extract clean content from HTML documents.
    Removes navigation, ads, scripts, and other noise.
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        
        # Tags to remove completely
        self.remove_tags = ['script', 'style', 'noscript', 'iframe', 'embed', 'object']
        
        if self.config.remove_navigation:
            self.remove_tags.extend(['nav', 'header'])
        
        if self.config.remove_ads:
            self.ad_classes = ['ad', 'ads', 'advertisement', 'banner', 'sponsor', 'promo']
        else:
            self.ad_classes = []
        
        if self.config.remove_footers:
            self.remove_tags.append('footer')
        
        # Common content containers
        self.content_tags = ['article', 'main', 'div[class*="content"]', 'div[class*="post"]',
                            'div[class*="article"]', 'div[class*="entry"]']
        
        # Code block patterns
        self.code_classes = ['code', 'highlight', 'syntax', 'pre', 'codehilite']
    
    def extract(self, document: Document) -> Document:
        """
        Extract content from a document's raw HTML.
        
        Args:
            document: Document with raw_html
            
        Returns:
            Document with extracted content and elements
        """
        if not document.raw_html:
            return document
        
        try:
            soup = BeautifulSoup(document.raw_html, 'html.parser')
            
            # Remove unwanted tags
            self._remove_unwanted_elements(soup)
            
            # Extract main content
            content, headings = self._extract_main_content(soup)
            
            # Extract tables if enabled
            tables = []
            if self.config.extract_tables:
                tables = self._extract_tables(soup)
            
            # Extract code blocks if enabled
            code_blocks = []
            if self.config.extract_code_blocks:
                code_blocks = self._extract_code_blocks(soup)
            
            # Extract links
            links = self._extract_links(soup)
            
            # Extract images
            images = self._extract_images(soup)
            
            # Update document
            document.content = content
            document.headings = headings
            document.tables = tables
            document.code_blocks = code_blocks
            document.links = links
            document.images = images
            
            logger.debug(f"Extracted content from {document.url}: {len(content)} chars")
            
        except Exception as e:
            logger.error(f"Error extracting content from {document.url}: {e}")
        
        return document
    
    def extract_batch(self, documents: List[Document]) -> List[Document]:
        """Extract content from multiple documents."""
        return [self.extract(doc) for doc in documents]
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements."""
        # Remove tags
        for tag_name in self.remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove elements with ad-related classes
        for ad_class in self.ad_classes:
            for element in soup.find_all(class_=re.compile(ad_class, re.I)):
                element.decompose()
        
        # Remove hidden elements
        for element in soup.find_all(style=re.compile(r'display:\s*none', re.I)):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, NavigableString) 
                                     and text.parent.name is None):
            comment.extract()
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Tuple[str, List[str]]:
        """Extract main content and headings."""
        # Try to find main content container
        content_element = None
        
        for selector in self.content_tags:
            if '[' in selector:
                # Handle attribute selectors
                tag, attr_pattern = selector.split('[')
                attr_name = attr_pattern.split('*=')[0]
                attr_value = attr_pattern.split('*="')[1].rstrip('"]')
                content_element = soup.find(tag, attrs={attr_name: re.compile(attr_value, re.I)})
            else:
                content_element = soup.find(selector)
            
            if content_element:
                break
        
        # Fall back to body
        if not content_element:
            content_element = soup.find('body') or soup
        
        # Extract headings
        headings = []
        for h_tag in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = h_tag.get_text(strip=True)
            if heading_text:
                headings.append(heading_text)
        
        # Get text content
        content = self._get_clean_text(content_element)
        
        return content, headings
    
    def _get_clean_text(self, element) -> str:
        """Get clean text from an element."""
        # Get text with proper spacing
        texts = []
        
        for string in element.stripped_strings:
            texts.append(string)
        
        # Join with proper spacing
        content = ' '.join(texts)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from HTML."""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = {
                'headers': [],
                'rows': []
            }
            
            # Extract headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    table_data['headers'] = [
                        th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])
                    ]
            
            # If no thead, try first row
            if not table_data['headers']:
                first_row = table.find('tr')
                if first_row and first_row.find('th'):
                    table_data['headers'] = [
                        th.get_text(strip=True) for th in first_row.find_all('th')
                    ]
            
            # Extract rows
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                # Skip header row
                if tr.find('th') and not tr.find('td'):
                    continue
                
                row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if row:
                    table_data['rows'].append(row)
            
            if table_data['rows']:
                tables.append(table_data)
        
        return tables
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract code blocks from HTML."""
        code_blocks = []
        
        # Find pre/code elements
        for pre in soup.find_all('pre'):
            code_element = pre.find('code')
            code_text = code_element.get_text() if code_element else pre.get_text()
            
            # Try to detect language
            language = None
            if code_element:
                classes = code_element.get('class', [])
                for cls in classes:
                    if cls.startswith('language-'):
                        language = cls.replace('language-', '')
                        break
                    elif cls.startswith('lang-'):
                        language = cls.replace('lang-', '')
                        break
            
            code_blocks.append({
                'code': code_text.strip(),
                'language': language
            })
        
        # Find standalone code elements with significant content
        for code in soup.find_all('code'):
            # Skip if already inside pre
            if code.find_parent('pre'):
                continue
            
            code_text = code.get_text().strip()
            if len(code_text) > 50:  # Only include substantial code blocks
                code_blocks.append({
                    'code': code_text,
                    'language': None
                })
        
        return code_blocks
    
    def _extract_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract links from HTML."""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href and not href.startswith('#') and not href.startswith('javascript:'):
                links.append(href)
        
        return list(set(links))
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract image URLs from HTML."""
        images = []
        
        for img in soup.find_all('img', src=True):
            src = img['src']
            if src:
                images.append(src)
        
        return list(set(images))
