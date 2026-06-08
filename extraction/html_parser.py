"""
Advanced HTML parsing utilities.
"""

import logging
import re
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class HTMLParser:
    """
    Advanced HTML parsing utilities.
    """
    
    @staticmethod
    def parse(html: str) -> BeautifulSoup:
        """Parse HTML string."""
        return BeautifulSoup(html, 'html.parser')
    
    @staticmethod
    def extract_metadata(html: str) -> Dict[str, Any]:
        """
        Extract metadata from HTML head.
        
        Args:
            html: HTML string
            
        Returns:
            Dictionary of metadata
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property', '')
            content = meta.get('content', '')
            
            if name and content:
                # Common meta tags
                if name.lower() in ['description', 'keywords', 'author', 'robots']:
                    metadata[name.lower()] = content
                
                # Open Graph
                if name.startswith('og:'):
                    og_key = name[3:]
                    metadata.setdefault('og', {})[og_key] = content
                
                # Twitter
                if name.startswith('twitter:'):
                    twitter_key = name[8:]
                    metadata.setdefault('twitter', {})[twitter_key] = content
        
        # Canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            metadata['canonical_url'] = canonical['href']
        
        # Language
        html_tag = soup.find('html')
        if html_tag:
            metadata['lang'] = html_tag.get('lang')
        
        return metadata
    
    @staticmethod
    def extract_structured_data(html: str) -> List[Dict[str, Any]]:
        """
        Extract JSON-LD structured data from HTML.
        
        Args:
            html: HTML string
            
        Returns:
            List of structured data objects
        """
        import json
        
        soup = BeautifulSoup(html, 'html.parser')
        structured_data = []
        
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    structured_data.extend(data)
                else:
                    structured_data.append(data)
            except (json.JSONDecodeError, TypeError):
                continue
        
        return structured_data
    
    @staticmethod
    def clean_html(html: str) -> str:
        """
        Clean HTML by removing scripts, styles, and comments.
        
        Args:
            html: HTML string
            
        Returns:
            Cleaned HTML string
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts and styles
        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.string)) 
                                     and str(text).strip().startswith('<!--')):
            comment.extract()
        
        return str(soup)
    
    @staticmethod
    def get_text_density(element) -> float:
        """
        Calculate text density of an HTML element.
        Higher density usually indicates content areas.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Text density score
        """
        text_length = len(element.get_text(strip=True))
        html_length = len(str(element))
        
        if html_length == 0:
            return 0
        
        return text_length / html_length
    
    @staticmethod
    def find_content_blocks(html: str, min_density: float = 0.3) -> List[str]:
        """
        Find content blocks based on text density.
        
        Args:
            html: HTML string
            min_density: Minimum text density threshold
            
        Returns:
            List of content block texts
        """
        soup = BeautifulSoup(html, 'html.parser')
        content_blocks = []
        
        for element in soup.find_all(['p', 'div', 'article', 'section']):
            density = HTMLParser.get_text_density(element)
            if density >= min_density:
                text = element.get_text(strip=True)
                if len(text) > 100:  # Minimum content length
                    content_blocks.append(text)
        
        return content_blocks
    
    @staticmethod
    def extract_article_content(html: str) -> Optional[str]:
        """
        Extract main article content using heuristics.
        
        Args:
            html: HTML string
            
        Returns:
            Article content or None
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try common article containers
        article_selectors = [
            ('article', {}),
            ('div', {'class': re.compile(r'article|content|post|entry', re.I)}),
            ('main', {}),
            ('div', {'id': re.compile(r'article|content|post|entry', re.I)}),
        ]
        
        best_content = None
        best_length = 0
        
        for tag, attrs in article_selectors:
            elements = soup.find_all(tag, attrs) if attrs else soup.find_all(tag)
            
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > best_length:
                    best_content = text
                    best_length = len(text)
        
        return best_content
