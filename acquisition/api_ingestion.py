"""
API-based data ingestion for various sources.
"""

import logging
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime
import time

import requests

from ..models import Document
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class APIIngester:
    """
    Ingest data from various APIs.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.user_agent
        })
    
    def ingest_wikipedia(self, titles: List[str], language: str = "en") -> List[Document]:
        """
        Ingest articles from Wikipedia.
        
        Args:
            titles: List of article titles
            language: Wikipedia language code
            
        Returns:
            List of Document objects
        """
        documents = []
        base_url = f"https://{language}.wikipedia.org/w/api.php"
        
        for title in titles:
            try:
                params = {
                    "action": "query",
                    "titles": title,
                    "prop": "extracts|info",
                    "exintro": False,
                    "explaintext": True,
                    "inprop": "url",
                    "format": "json"
                }
                
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        continue
                    
                    doc = Document(
                        content=page_data.get("extract", ""),
                        url=page_data.get("fullurl"),
                        title=page_data.get("title"),
                        metadata={
                            "source": "wikipedia",
                            "language": language,
                            "page_id": page_id
                        },
                        crawled_at=datetime.now()
                    )
                    documents.append(doc)
                    logger.info(f"Ingested Wikipedia article: {title}")
                
                time.sleep(self.config.download_delay)
                
            except Exception as e:
                logger.error(f"Error ingesting Wikipedia article {title}: {e}")
        
        return documents
    
    def ingest_arxiv(self, query: str, max_results: int = 100) -> List[Document]:
        """
        Ingest papers from arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of Document objects
        """
        try:
            import arxiv
        except ImportError:
            raise ImportError("arxiv package not installed. Install with: pip install arxiv")
        
        documents = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            for result in search.results():
                doc = Document(
                    content=result.summary,
                    url=result.pdf_url,
                    title=result.title,
                    metadata={
                        "source": "arxiv",
                        "arxiv_id": result.entry_id,
                        "authors": [str(a) for a in result.authors],
                        "categories": result.categories,
                        "published": result.published.isoformat() if result.published else None,
                        "updated": result.updated.isoformat() if result.updated else None,
                    },
                    crawled_at=datetime.now()
                )
                documents.append(doc)
                logger.info(f"Ingested arXiv paper: {result.title[:50]}...")
                
        except Exception as e:
            logger.error(f"Error ingesting arXiv papers: {e}")
        
        return documents
    
    def ingest_stackoverflow(self, tags: List[str], max_questions: int = 100, 
                            min_score: int = 10) -> List[Document]:
        """
        Ingest questions and answers from Stack Overflow.
        
        Args:
            tags: List of tags to filter by
            max_questions: Maximum number of questions
            min_score: Minimum question score
            
        Returns:
            List of Document objects
        """
        documents = []
        base_url = "https://api.stackexchange.com/2.3/questions"
        
        try:
            params = {
                "order": "desc",
                "sort": "votes",
                "tagged": ";".join(tags),
                "site": "stackoverflow",
                "filter": "withbody",
                "pagesize": min(100, max_questions),
                "min": min_score
            }
            
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("items", [])[:max_questions]:
                # Combine question and accepted answer
                content = f"Question: {item.get('title', '')}\n\n{item.get('body', '')}"
                
                doc = Document(
                    content=content,
                    url=item.get("link"),
                    title=item.get("title"),
                    metadata={
                        "source": "stackoverflow",
                        "question_id": item.get("question_id"),
                        "tags": item.get("tags", []),
                        "score": item.get("score"),
                        "answer_count": item.get("answer_count"),
                        "is_answered": item.get("is_answered"),
                    },
                    crawled_at=datetime.now()
                )
                documents.append(doc)
                logger.info(f"Ingested SO question: {item.get('title', '')[:50]}...")
            
        except Exception as e:
            logger.error(f"Error ingesting Stack Overflow data: {e}")
        
        return documents
    
    def ingest_github_repos(self, query: str, max_repos: int = 50, 
                           token: Optional[str] = None) -> List[Document]:
        """
        Ingest README files from GitHub repositories.
        
        Args:
            query: Search query
            max_repos: Maximum number of repos
            token: GitHub API token (optional)
            
        Returns:
            List of Document objects
        """
        documents = []
        search_url = "https://api.github.com/search/repositories"
        
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
        
        try:
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(100, max_repos)
            }
            
            response = self.session.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            for repo in data.get("items", [])[:max_repos]:
                try:
                    # Get README
                    readme_url = f"https://api.github.com/repos/{repo['full_name']}/readme"
                    readme_response = self.session.get(readme_url, headers=headers)
                    
                    if readme_response.status_code == 200:
                        import base64
                        readme_data = readme_response.json()
                        content = base64.b64decode(readme_data.get("content", "")).decode("utf-8")
                        
                        doc = Document(
                            content=content,
                            url=repo.get("html_url"),
                            title=repo.get("full_name"),
                            metadata={
                                "source": "github",
                                "repo_name": repo.get("full_name"),
                                "description": repo.get("description"),
                                "stars": repo.get("stargazers_count"),
                                "forks": repo.get("forks_count"),
                                "language": repo.get("language"),
                                "topics": repo.get("topics", []),
                            },
                            crawled_at=datetime.now()
                        )
                        documents.append(doc)
                        logger.info(f"Ingested GitHub repo: {repo.get('full_name')}")
                    
                    time.sleep(self.config.download_delay)
                    
                except Exception as e:
                    logger.error(f"Error fetching README for {repo.get('full_name')}: {e}")
            
        except Exception as e:
            logger.error(f"Error searching GitHub repos: {e}")
        
        return documents
    
    def ingest_from_urls(self, urls: List[str]) -> List[Document]:
        """
        Ingest content from a list of URLs using requests.
        
        Args:
            urls: List of URLs to ingest
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for url in urls:
            try:
                response = self.session.get(url, timeout=self.config.request_timeout)
                response.raise_for_status()
                
                doc = Document(
                    content="",  # Will be extracted later
                    url=url,
                    raw_html=response.text,
                    crawled_at=datetime.now()
                )
                documents.append(doc)
                logger.info(f"Ingested URL: {url}")
                
                time.sleep(self.config.download_delay)
                
            except Exception as e:
                logger.error(f"Error ingesting URL {url}: {e}")
        
        return documents
