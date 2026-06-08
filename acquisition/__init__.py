"""
Data acquisition module for web crawling and data ingestion.
"""

from .crawler import WebCrawler, ScrapySpider
from .api_ingestion import APIIngester
from .batch_ingestion import BatchIngester

__all__ = ["WebCrawler", "ScrapySpider", "APIIngester", "BatchIngester"]
