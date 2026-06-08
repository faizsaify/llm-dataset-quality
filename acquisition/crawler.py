"""
Web crawler implementation using Scrapy and Selenium.
"""

import logging
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime
from pathlib import Path
import json

try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    from scrapy.http import Response
    SCRAPY_AVAILABLE = True
except ImportError:
    SCRAPY_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

import requests
from bs4 import BeautifulSoup

from ..models import Document
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class WebCrawler:
    """
    Simple web crawler using requests and BeautifulSoup.
    Suitable for static websites.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.user_agent
        })
        self.visited_urls = set()
        self.documents: List[Document] = []
    
    def crawl(self, urls: List[str], max_pages: Optional[int] = None) -> List[Document]:
        """
        Crawl a list of URLs and return documents.
        
        Args:
            urls: List of URLs to crawl
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of Document objects
        """
        self.documents = []
        to_visit = list(urls)
        pages_crawled = 0
        
        while to_visit and (max_pages is None or pages_crawled < max_pages):
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            try:
                doc = self._fetch_url(url)
                if doc:
                    self.documents.append(doc)
                    self.visited_urls.add(url)
                    pages_crawled += 1
                    
                    # Extract links for further crawling
                    if self.config.max_depth > 0:
                        new_urls = self._extract_links(doc.raw_html, url)
                        to_visit.extend([u for u in new_urls if u not in self.visited_urls])
                    
                    logger.info(f"Crawled: {url} ({pages_crawled} pages)")
                    
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
        
        return self.documents
    
    def _fetch_url(self, url: str) -> Optional[Document]:
        """Fetch a single URL and return a Document."""
        try:
            response = self.session.get(
                url, 
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.string if soup.title else None
            
            return Document(
                content="",  # Will be extracted later
                url=url,
                title=title,
                raw_html=response.text,
                crawled_at=datetime.now()
            )
            
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML."""
        from urllib.parse import urljoin, urlparse
        
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            
            # Filter by allowed domains
            if self.config.allowed_domains:
                parsed = urlparse(full_url)
                if parsed.netloc not in self.config.allowed_domains:
                    continue
            
            links.append(full_url)
        
        return links


class SeleniumCrawler:
    """
    Web crawler using Selenium for JavaScript-rendered pages.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None, headless: bool = True):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is not installed. Install it with: pip install selenium")
        
        self.config = config or CrawlerConfig()
        self.headless = headless
        self.driver = None
        self.documents: List[Document] = []
    
    def _init_driver(self):
        """Initialize the Selenium WebDriver."""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-agent={self.config.user_agent}")
        
        self.driver = webdriver.Chrome(options=options)
    
    def crawl(self, urls: List[str], wait_time: int = 5) -> List[Document]:
        """
        Crawl URLs using Selenium.
        
        Args:
            urls: List of URLs to crawl
            wait_time: Time to wait for page to load
            
        Returns:
            List of Document objects
        """
        self._init_driver()
        self.documents = []
        
        try:
            for url in urls:
                try:
                    doc = self._fetch_url(url, wait_time)
                    if doc:
                        self.documents.append(doc)
                        logger.info(f"Crawled (Selenium): {url}")
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
        finally:
            self.close()
        
        return self.documents
    
    def _fetch_url(self, url: str, wait_time: int) -> Optional[Document]:
        """Fetch a single URL using Selenium."""
        self.driver.get(url)
        
        # Wait for page to load
        WebDriverWait(self.driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        html = self.driver.page_source
        title = self.driver.title
        
        return Document(
            content="",
            url=url,
            title=title,
            raw_html=html,
            crawled_at=datetime.now()
        )
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None


if SCRAPY_AVAILABLE:
    class DataFactorySpider(scrapy.Spider):
        """
        Scrapy spider for large-scale web crawling.
        """
        name = "data_factory_spider"
        
        def __init__(self, start_urls: List[str], allowed_domains: List[str] = None, 
                     output_file: str = "crawled_data.jsonl", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_urls = start_urls
            self.allowed_domains = allowed_domains or []
            self.output_file = output_file
        
        def parse(self, response: Response):
            """Parse a crawled page."""
            # Extract document data
            doc_data = {
                "url": response.url,
                "title": response.css("title::text").get(),
                "raw_html": response.text,
                "crawled_at": datetime.now().isoformat(),
            }
            
            # Save to file
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(doc_data, ensure_ascii=False) + '\n')
            
            yield doc_data
            
            # Follow links
            for href in response.css("a::attr(href)").getall():
                yield response.follow(href, callback=self.parse)


class ScrapySpider:
    """
    Wrapper for Scrapy spider to integrate with the pipeline.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        if not SCRAPY_AVAILABLE:
            raise ImportError("Scrapy is not installed. Install it with: pip install scrapy")
        
        self.config = config or CrawlerConfig()
    
    def crawl(self, start_urls: List[str], output_file: str = "crawled_data.jsonl") -> str:
        """
        Run the Scrapy spider.
        
        Args:
            start_urls: List of URLs to start crawling
            output_file: Output file path for crawled data
            
        Returns:
            Path to the output file
        """
        process = CrawlerProcess({
            'USER_AGENT': self.config.user_agent,
            'CONCURRENT_REQUESTS': self.config.concurrent_requests,
            'DOWNLOAD_DELAY': self.config.download_delay,
            'ROBOTSTXT_OBEY': self.config.respect_robots_txt,
            'DEPTH_LIMIT': self.config.max_depth,
            'LOG_LEVEL': 'INFO',
        })
        
        process.crawl(
            DataFactorySpider,
            start_urls=start_urls,
            allowed_domains=self.config.allowed_domains,
            output_file=output_file
        )
        process.start()
        
        return output_file
    
    def load_results(self, output_file: str) -> List[Document]:
        """Load crawled results as Document objects."""
        documents = []
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    doc = Document(
                        content="",
                        url=data.get("url"),
                        title=data.get("title"),
                        raw_html=data.get("raw_html"),
                        crawled_at=datetime.fromisoformat(data["crawled_at"]) if data.get("crawled_at") else None
                    )
                    documents.append(doc)
        
        return documents
