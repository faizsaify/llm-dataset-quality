"""
Dataset export to various formats.
"""

import logging
import json
from typing import List, Optional, Dict, Any, Iterator
from pathlib import Path
from datetime import datetime

from ..models import Document, Batch
from ..config import ExportConfig
from ..utils import batch_iterator

logger = logging.getLogger(__name__)


class DatasetExporter:
    """
    Export datasets to various formats.
    
    Supported formats:
    - JSONL
    - Parquet
    - Arrow
    - HuggingFace Dataset
    - CSV
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(self, documents: List[Document], name: str = "dataset") -> Dict[str, Path]:
        """
        Export documents to configured formats.
        
        Args:
            documents: List of documents to export
            name: Dataset name
            
        Returns:
            Dictionary mapping format to output path
        """
        # Filter valid documents
        valid_docs = [doc for doc in documents if doc.content and not doc.is_duplicate]
        
        logger.info(f"Exporting {len(valid_docs)} documents...")
        
        output_paths = {}
        
        for fmt in self.config.formats:
            try:
                if fmt == "jsonl":
                    path = self.export_jsonl(valid_docs, name)
                elif fmt == "parquet":
                    path = self.export_parquet(valid_docs, name)
                elif fmt == "arrow":
                    path = self.export_arrow(valid_docs, name)
                elif fmt == "csv":
                    path = self.export_csv(valid_docs, name)
                elif fmt == "huggingface":
                    path = self.export_huggingface(valid_docs, name)
                else:
                    logger.warning(f"Unknown export format: {fmt}")
                    continue
                
                output_paths[fmt] = path
                logger.info(f"Exported to {fmt}: {path}")
                
            except Exception as e:
                logger.error(f"Error exporting to {fmt}: {e}")
        
        # Push to HuggingFace Hub if configured
        if self.config.push_to_hub and self.config.hub_repo_id:
            self._push_to_hub(valid_docs, name)
        
        return output_paths
    
    def export_jsonl(self, documents: List[Document], name: str) -> Path:
        """
        Export to JSONL format.
        
        Args:
            documents: Documents to export
            name: Dataset name
            
        Returns:
            Output path
        """
        filename = f"{name}.jsonl"
        if self.config.compression == "gzip":
            filename += ".gz"
        
        output_path = self.config.output_dir / filename
        
        if self.config.compression == "gzip":
            import gzip
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(self._doc_to_record(doc), ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(self._doc_to_record(doc), ensure_ascii=False) + '\n')
        
        return output_path
    
    def export_parquet(self, documents: List[Document], name: str) -> Path:
        """
        Export to Parquet format.
        
        Args:
            documents: Documents to export
            name: Dataset name
            
        Returns:
            Output path
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("PyArrow not installed. Install with: pip install pyarrow")
        
        output_path = self.config.output_dir / f"{name}.parquet"
        
        # Convert to records
        records = [self._doc_to_record(doc) for doc in documents]
        
        # Create table
        table = pa.Table.from_pylist(records)
        
        # Write with compression
        compression = self.config.compression or 'snappy'
        pq.write_table(table, output_path, compression=compression)
        
        return output_path
    
    def export_arrow(self, documents: List[Document], name: str) -> Path:
        """
        Export to Arrow format.
        
        Args:
            documents: Documents to export
            name: Dataset name
            
        Returns:
            Output path
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("PyArrow not installed. Install with: pip install pyarrow")
        
        output_path = self.config.output_dir / f"{name}.arrow"
        
        records = [self._doc_to_record(doc) for doc in documents]
        table = pa.Table.from_pylist(records)
        
        with pa.OSFile(str(output_path), 'wb') as f:
            writer = pa.ipc.RecordBatchFileWriter(f, table.schema)
            writer.write_table(table)
            writer.close()
        
        return output_path
    
    def export_csv(self, documents: List[Document], name: str) -> Path:
        """
        Export to CSV format.
        
        Args:
            documents: Documents to export
            name: Dataset name
            
        Returns:
            Output path
        """
        import csv
        
        output_path = self.config.output_dir / f"{name}.csv"
        
        if not documents:
            return output_path
        
        # Get all keys
        records = [self._doc_to_record(doc) for doc in documents]
        
        # Flatten metadata for CSV
        flat_records = []
        for record in records:
            flat = {k: v for k, v in record.items() if k != 'metadata'}
            if 'metadata' in record:
                for mk, mv in record['metadata'].items():
                    if isinstance(mv, (str, int, float, bool)):
                        flat[f"meta_{mk}"] = mv
            flat_records.append(flat)
        
        fieldnames = list(flat_records[0].keys()) if flat_records else []
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(flat_records)
        
        return output_path
    
    def export_huggingface(self, documents: List[Document], name: str) -> Path:
        """
        Export to HuggingFace Dataset format.
        
        Args:
            documents: Documents to export
            name: Dataset name
            
        Returns:
            Output path
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets not installed. Install with: pip install datasets")
        
        output_path = self.config.output_dir / name
        
        records = [self._doc_to_record(doc) for doc in documents]
        
        # Flatten for HuggingFace
        flat_records = []
        for record in records:
            flat = {k: v for k, v in record.items() if k != 'metadata'}
            flat_records.append(flat)
        
        dataset = Dataset.from_list(flat_records)
        dataset.save_to_disk(str(output_path))
        
        return output_path
    
    def _push_to_hub(self, documents: List[Document], name: str):
        """Push dataset to HuggingFace Hub."""
        try:
            from datasets import Dataset
            
            records = [self._doc_to_record(doc) for doc in documents]
            flat_records = [{k: v for k, v in r.items() if k != 'metadata'} for r in records]
            
            dataset = Dataset.from_list(flat_records)
            dataset.push_to_hub(self.config.hub_repo_id, private=True)
            
            logger.info(f"Pushed to HuggingFace Hub: {self.config.hub_repo_id}")
            
        except Exception as e:
            logger.error(f"Error pushing to Hub: {e}")
    
    def _doc_to_record(self, doc: Document) -> Dict[str, Any]:
        """Convert document to export record."""
        record = {
            "id": doc.doc_id,
            "text": doc.content,
            "url": doc.url,
            "title": doc.title,
            "language": doc.language,
            "quality_score": doc.quality_score,
            "readability_score": doc.readability_score,
        }
        
        # Add metadata as nested dict or flatten
        if doc.metadata:
            record["metadata"] = doc.metadata
        
        return record
    
    def export_streaming(self, documents: Iterator[Document], name: str, 
                        format: str = "jsonl") -> Path:
        """
        Export documents in streaming mode for large datasets.
        
        Args:
            documents: Iterator of documents
            name: Dataset name
            format: Output format
            
        Returns:
            Output path
        """
        if format == "jsonl":
            return self._stream_jsonl(documents, name)
        elif format == "parquet":
            return self._stream_parquet(documents, name)
        else:
            raise ValueError(f"Streaming not supported for format: {format}")
    
    def _stream_jsonl(self, documents: Iterator[Document], name: str) -> Path:
        """Stream export to JSONL."""
        output_path = self.config.output_dir / f"{name}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            count = 0
            for doc in documents:
                if doc.content and not doc.is_duplicate:
                    f.write(json.dumps(self._doc_to_record(doc), ensure_ascii=False) + '\n')
                    count += 1
                    
                    if count % 10000 == 0:
                        logger.info(f"Exported {count} documents...")
        
        logger.info(f"Streaming export complete: {count} documents")
        return output_path
    
    def _stream_parquet(self, documents: Iterator[Document], name: str) -> Path:
        """Stream export to Parquet with chunking."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("PyArrow not installed")
        
        output_path = self.config.output_dir / f"{name}.parquet"
        
        writer = None
        chunk_records = []
        total_count = 0
        
        for doc in documents:
            if doc.content and not doc.is_duplicate:
                chunk_records.append(self._doc_to_record(doc))
                
                if len(chunk_records) >= self.config.chunk_size:
                    table = pa.Table.from_pylist(chunk_records)
                    
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema)
                    
                    writer.write_table(table)
                    total_count += len(chunk_records)
                    chunk_records = []
                    logger.info(f"Exported {total_count} documents...")
        
        # Write remaining records
        if chunk_records:
            table = pa.Table.from_pylist(chunk_records)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_count += len(chunk_records)
        
        if writer:
            writer.close()
        
        logger.info(f"Streaming export complete: {total_count} documents")
        return output_path


class DatasetStats:
    """Generate statistics for exported datasets."""
    
    @staticmethod
    def compute(documents: List[Document]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not documents:
            return {"total_documents": 0}
        
        word_counts = [len(doc.content.split()) for doc in documents if doc.content]
        quality_scores = [doc.quality_score for doc in documents if doc.quality_score]
        
        stats = {
            "total_documents": len(documents),
            "total_words": sum(word_counts),
            "avg_words_per_doc": sum(word_counts) / len(word_counts) if word_counts else 0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "languages": list(set(doc.language for doc in documents if doc.language)),
            "duplicates_removed": sum(1 for doc in documents if doc.is_duplicate),
        }
        
        return stats
