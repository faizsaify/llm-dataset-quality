"""
Batch ingestion for local files and existing datasets.
"""

import logging
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime
from pathlib import Path
import json

from ..models import Document, Batch
from ..utils import batch_iterator

logger = logging.getLogger(__name__)


class BatchIngester:
    """
    Ingest data from local files and existing datasets.
    """
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    def ingest_jsonl(self, filepath: Path, content_field: str = "text",
                     url_field: Optional[str] = "url",
                     title_field: Optional[str] = "title") -> Iterator[Batch]:
        """
        Ingest documents from a JSONL file.
        
        Args:
            filepath: Path to JSONL file
            content_field: Field name for content
            url_field: Field name for URL
            title_field: Field name for title
            
        Yields:
            Batches of documents
        """
        documents = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    doc = Document(
                        content=data.get(content_field, ""),
                        url=data.get(url_field) if url_field else None,
                        title=data.get(title_field) if title_field else None,
                        metadata={k: v for k, v in data.items() 
                                 if k not in [content_field, url_field, title_field]},
                        crawled_at=datetime.now()
                    )
                    documents.append(doc)
                    
                    if len(documents) >= self.batch_size:
                        yield Batch(documents=documents)
                        documents = []
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON line: {e}")
        
        if documents:
            yield Batch(documents=documents)
    
    def ingest_text_files(self, directory: Path, pattern: str = "*.txt") -> Iterator[Batch]:
        """
        Ingest documents from text files in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            
        Yields:
            Batches of documents
        """
        documents = []
        
        for filepath in Path(directory).glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    content=content,
                    title=filepath.stem,
                    metadata={
                        "source": "local_file",
                        "filepath": str(filepath)
                    },
                    crawled_at=datetime.now()
                )
                documents.append(doc)
                
                if len(documents) >= self.batch_size:
                    yield Batch(documents=documents)
                    documents = []
                    
            except Exception as e:
                logger.error(f"Error reading file {filepath}: {e}")
        
        if documents:
            yield Batch(documents=documents)
    
    def ingest_parquet(self, filepath: Path, content_column: str = "text") -> Iterator[Batch]:
        """
        Ingest documents from a Parquet file.
        
        Args:
            filepath: Path to Parquet file
            content_column: Column name for content
            
        Yields:
            Batches of documents
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("PyArrow not installed. Install with: pip install pyarrow")
        
        table = pq.read_table(filepath)
        df = table.to_pandas()
        
        documents = []
        for idx, row in df.iterrows():
            doc = Document(
                content=str(row.get(content_column, "")),
                metadata={k: v for k, v in row.items() if k != content_column},
                crawled_at=datetime.now()
            )
            documents.append(doc)
            
            if len(documents) >= self.batch_size:
                yield Batch(documents=documents)
                documents = []
        
        if documents:
            yield Batch(documents=documents)
    
    def ingest_huggingface_dataset(self, dataset_name: str, split: str = "train",
                                    text_column: str = "text",
                                    streaming: bool = True) -> Iterator[Batch]:
        """
        Ingest documents from a HuggingFace dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split
            text_column: Column name for text
            streaming: Whether to use streaming mode
            
        Yields:
            Batches of documents
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets not installed. Install with: pip install datasets")
        
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        
        documents = []
        for item in dataset:
            doc = Document(
                content=str(item.get(text_column, "")),
                metadata={
                    "source": f"huggingface:{dataset_name}",
                    "split": split,
                    **{k: v for k, v in item.items() if k != text_column}
                },
                crawled_at=datetime.now()
            )
            documents.append(doc)
            
            if len(documents) >= self.batch_size:
                yield Batch(documents=documents)
                documents = []
        
        if documents:
            yield Batch(documents=documents)
    
    def ingest_csv(self, filepath: Path, content_column: str = "text",
                  delimiter: str = ",") -> Iterator[Batch]:
        """
        Ingest documents from a CSV file.
        
        Args:
            filepath: Path to CSV file
            content_column: Column name for content
            delimiter: CSV delimiter
            
        Yields:
            Batches of documents
        """
        import csv
        
        documents = []
        
        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                doc = Document(
                    content=row.get(content_column, ""),
                    metadata={
                        "source": "csv",
                        "filepath": str(filepath),
                        **{k: v for k, v in row.items() if k != content_column}
                    },
                    crawled_at=datetime.now()
                )
                documents.append(doc)
                
                if len(documents) >= self.batch_size:
                    yield Batch(documents=documents)
                    documents = []
        
        if documents:
            yield Batch(documents=documents)
