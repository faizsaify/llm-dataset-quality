"""
Example usage of AI Training Data Factory.
"""

import sys
from pathlib import Path

# Add parent directory to path for running without installation
# examples/ is inside ai_training_data_factory/, so we need grandparent (parth/)
package_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(package_root))

from ai_training_data_factory import DataPipeline
from ai_training_data_factory.config import PipelineConfig
from ai_training_data_factory.models import Document
from pathlib import Path


def example_basic_usage():
    """Basic pipeline usage example."""
    print("=" * 50)
    print("Example 1: Basic Pipeline Usage")
    print("=" * 50)
    
    # Initialize with relaxed config for demo
    config = PipelineConfig()
    # Relax quality filters for short demo texts
    config.quality.min_word_count = 10
    config.quality.min_quality_score = 0.1
    config.quality.check_toxicity = False
    config.quality.check_pii = False
    
    pipeline = DataPipeline(config)
    
    # Create sample documents
    documents = [
        Document(
            content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It powers many applications from recommendation engines to self-driving cars.",
            url="https://example.com/ml-intro",
            title="Introduction to Machine Learning"
        ),
        Document(
            content="Deep learning uses neural networks with many layers to progressively extract higher-level features from raw input. It has revolutionized computer vision, speech recognition, and natural language processing.",
            url="https://example.com/deep-learning",
            title="Deep Learning Fundamentals"
        ),
        Document(
            content="Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language. Modern NLP uses transformers and large language models for tasks like translation and summarization.",
            url="https://example.com/nlp",
            title="NLP Overview"
        ),
    ]
    
    # Process documents
    processed = pipeline.process(documents)
    
    print(f"Processed {len(processed)} documents")
    for doc in processed:
        quality = doc.quality_score if doc.quality_score else 0
        readability = doc.readability_score if doc.readability_score else 0
        print(f"  - {doc.title}: quality={quality:.2f}, readability={readability:.1f}")
    
    return pipeline, processed


def example_web_crawling():
    """Web crawling example."""
    print("\n" + "=" * 50)
    print("Example 2: Web Crawling")
    print("=" * 50)
    
    config = PipelineConfig()
    config.crawler.max_depth = 2
    config.crawler.download_delay = 1.0
    
    pipeline = DataPipeline(config)
    
    # Note: This would actually crawl websites
    # documents = pipeline.crawl_and_process(
    #     urls=["https://en.wikipedia.org/wiki/Machine_learning"],
    #     max_pages=10
    # )
    
    print("Web crawling example (disabled for demo)")
    print("Usage: pipeline.crawl_and_process(urls=['...'], max_pages=10)")


def example_api_ingestion():
    """API ingestion example."""
    print("\n" + "=" * 50)
    print("Example 3: API Ingestion")
    print("=" * 50)
    
    config = PipelineConfig()
    pipeline = DataPipeline(config)
    
    # Note: This would actually call APIs
    # Wikipedia
    # documents = pipeline.ingest_and_process(
    #     source="wikipedia",
    #     titles=["Artificial intelligence", "Machine learning"]
    # )
    
    # arXiv
    # documents = pipeline.ingest_and_process(
    #     source="arxiv",
    #     query="transformer neural network",
    #     max_results=50
    # )
    
    print("API ingestion example (disabled for demo)")
    print("Usage: pipeline.ingest_and_process(source='wikipedia', titles=[...])")


def example_semantic_search():
    """Semantic search example."""
    print("\n" + "=" * 50)
    print("Example 4: Semantic Search")
    print("=" * 50)
    
    # Check if sentence-transformers is available
    try:
        from sentence_transformers import SentenceTransformer
        has_sentence_transformers = True
    except ImportError:
        has_sentence_transformers = False
        print("NOTE: sentence-transformers not installed. Semantic search demo skipped.")
        print("Install with: pip install sentence-transformers")
        return
    
    config = PipelineConfig()
    config.quality.min_word_count = 10
    config.quality.min_quality_score = 0.1
    config.quality.check_toxicity = False
    config.quality.check_pii = False
    
    pipeline = DataPipeline(config)
    
    # Create sample documents
    documents = [
        Document(
            content="Python is a high-level programming language known for its simple syntax and readability. It's widely used in data science and machine learning applications.",
            title="Python Programming"
        ),
        Document(
            content="JavaScript is a scripting language primarily used for web development. It runs in browsers and can also be used server-side with Node.js framework.",
            title="JavaScript Overview"
        ),
        Document(
            content="TensorFlow is an open-source machine learning framework developed by Google. It's used for building and training neural networks efficiently.",
            title="TensorFlow Guide"
        ),
        Document(
            content="PyTorch is a deep learning framework developed by Facebook's AI Research lab. It's known for its dynamic computational graphs approach.",
            title="PyTorch Tutorial"
        ),
    ]
    
    # Process documents
    processed = pipeline.process(documents)
    
    if not processed:
        print("No documents passed quality filter. Skipping search demo.")
        return
    
    # Build search index
    print("Building search index...")
    pipeline.build_search_index(processed)
    
    # Search
    query = "machine learning frameworks"
    print(f"\nSearching for: '{query}'")
    
    results = pipeline.search(query, top_k=3)
    
    print("\nResults:")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc.title}")
        print(f"           {doc.content[:80]}...")


def example_quality_filtering():
    """Quality filtering example."""
    print("\n" + "=" * 50)
    print("Example 5: Quality Filtering")
    print("=" * 50)
    
    config = PipelineConfig()
    config.quality.min_quality_score = 0.4
    config.quality.min_word_count = 15
    config.quality.check_toxicity = False
    config.quality.check_pii = False
    
    pipeline = DataPipeline(config)
    
    documents = [
        Document(
            content="This is a high-quality article about machine learning. It contains detailed information about various algorithms and their applications in real-world scenarios and production systems.",
            title="High Quality Article"
        ),
        Document(
            content="bad txt lol",  # Low quality
            title="Low Quality"
        ),
        Document(
            content="The quick brown fox jumps over the lazy dog. " * 10,  # Repetitive
            title="Repetitive Content"
        ),
    ]
    
    processed = pipeline.process(documents)
    
    print(f"Input: {len(documents)} documents")
    print(f"Output: {len(processed)} documents (after quality filtering)")
    
    for doc in processed:
        quality = doc.quality_score if doc.quality_score else 0
        print(f"  - {doc.title}: quality={quality:.2f}")


def example_export():
    """Export example."""
    print("\n" + "=" * 50)
    print("Example 6: Dataset Export")
    print("=" * 50)
    
    config = PipelineConfig()
    config.export.output_dir = Path("./output")
    config.export.formats = ["jsonl", "parquet"]
    
    pipeline = DataPipeline(config)
    
    documents = [
        Document(
            content="Sample document for export demonstration.",
            title="Export Demo",
            url="https://example.com/demo"
        ),
    ]
    
    processed = pipeline.process(documents)
    
    print("Export formats available: JSONL, Parquet, Arrow, CSV, HuggingFace")
    print("Usage: pipeline.export('dataset_name')")
    
    # Uncomment to actually export
    # output_paths = pipeline.export("demo_dataset")
    # for fmt, path in output_paths.items():
    #     print(f"  {fmt}: {path}")


def main():
    """Run all examples."""
    print("AI Training Data Factory - Usage Examples")
    print("=" * 50)
    
    # Run examples
    pipeline, processed = example_basic_usage()
    example_web_crawling()
    example_api_ingestion()
    example_semantic_search()
    example_quality_filtering()
    example_export()
    
    # Print final stats
    print("\n" + "=" * 50)
    print("Pipeline Statistics")
    print("=" * 50)
    stats = pipeline.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
