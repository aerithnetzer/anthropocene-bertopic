import torch
import time
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import cuml
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP


def load_custom_dataset(
    corpus_path="~/big_corpus.txt",
    categories_path="~/big-categories.txt",
    dates_path="~/big_dates.txt",
    n_samples=None,
):
    """
    Load a custom dataset from text files.

    Args:
        corpus_path: Path to the corpus text file
        categories_path: Path to the categories text file
        dates_path: Path to the dates text file
        n_samples: Number of samples to load (None for all)

    Returns:
        documents: List of text documents
        categories: List of categories for each document
        timestamps: List of timestamps for each document
    """
    print("Loading custom dataset...")

    # Load corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    # Load categories
    with open(categories_path, "r", encoding="utf-8") as f:
        categories = [line.strip() for line in f if line.strip()]

    # Load dates
    with open(dates_path, "r", encoding="utf-8") as f:
        timestamps = [line.strip() for line in f if line.strip()]

    # Ensure all lists have the same length
    min_length = min(len(documents), len(categories), len(timestamps))
    documents = documents[:min_length]
    categories = categories[:min_length]
    timestamps = timestamps[:min_length]

    # Sample if needed
    if n_samples and n_samples < len(documents):
        documents = documents[:n_samples]
        categories = categories[:n_samples]
        timestamps = timestamps[:n_samples]

    print(f"Dataset loaded: {len(documents)} documents")
    return documents, categories, timestamps


def test_bertopic_performance(
    documents,
    categories=None,
    timestamps=None,
    use_cuda=True,
    embedding_model="all-MiniLM-L6-v2",
):
    """
    Test BERTopic performance with or without CUDA, using categories for supervised learning
    and timestamps for time-based analysis.

    Args:
        documents: List of documents to process
        categories: List of categories for supervised learning
        timestamps: List of timestamps for time-based analysis
        use_cuda: Whether to use CUDA for embeddings
        embedding_model: Name of the SentenceTransformer model to use
    """
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if use_cuda and not cuda_available:
        print("CUDA requested but not available. Falling back to CPU.")
        use_cuda = False

    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")

    if use_cuda:
        gpu_info = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_info}")

    # Initialize embedding model with specified device
    print(f"Initializing embedding model: {embedding_model}")
    embedding_model = SentenceTransformer(embedding_model, device=device)

    # Create dimensionality reduction model
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    # Create clustering model
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Create BERTopic model with supervised learning if categories are provided
    if categories is not None:
        print("Using supervised learning with categories")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=True,
        )
    else:
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=True,
        )

    # Measure performance
    start_time = time.time()

    # Fit the model
    if categories is not None:
        # Use semi-supervised learning with categories
        topics, probs = topic_model.fit_transform(documents, y=categories)
    else:
        topics, probs = topic_model.fit_transform(documents)

    end_time = time.time()
    processing_time = end_time - start_time
    topic_model.save("bertopic_model")
    # Get results
    performance_metrics = {
        "device": device,
        "num_documents": len(documents),
        "total_time_seconds": processing_time,
        "time_per_document_ms": (processing_time / len(documents)) * 1000,
        "num_topics": len(set(topics)) - 1,  # -1 to exclude the -1 noise cluster
    }

    # Print results
    print("\nPerformance Metrics:")
    print(f"Device: {performance_metrics['device']}")
    print(f"Number of documents: {performance_metrics['num_documents']}")
    print(
        f"Total processing time: {performance_metrics['total_time_seconds']:.2f} seconds"
    )
    print(f"Time per document: {performance_metrics['time_per_document_ms']:.2f} ms")
    print(f"Number of topics found: {performance_metrics['num_topics']}")

    # Display most frequent topics
    print("\nTop 10 Topics:")
    for _, topic_info in topic_model.get_topic_info().iloc[:10].iterrows():
        print(
            f"Topic {topic_info['Topic']}: {topic_info['Count']} documents - {topic_info['Name']}"
        )

    # Time-based topic analysis if timestamps are provided
    if timestamps is not None:
        print("\nPerforming time-based topic analysis...")
        try:
            # Convert timestamps to datetime format
            timestamps_pd = pd.to_datetime(timestamps)

            # Generate topics over time visualization data
            topics_over_time = topic_model.topics_over_time(
                documents, topics, timestamps_pd
            )
            print(
                f"Topics over time analysis complete. Found {len(topics_over_time)} data points."
            )
        except Exception as e:
            print(f"Error in time-based analysis: {e}")

    return topic_model, performance_metrics


def compare_cpu_vs_gpu(
    documents, categories=None, timestamps=None, sample_sizes=[1000, 5000, 10000]
):
    """Compare performance between CPU and GPU for different sample sizes"""
    results = []

    for size in sample_sizes:
        if size > len(documents):
            size = len(documents)

        sample_docs = documents[:size]
        sample_cats = categories[:size] if categories is not None else None
        sample_times = timestamps[:size] if timestamps is not None else None

        print(f"\n{'=' * 50}")
        print(f"Testing with {size} documents on CPU")
        print(f"{'=' * 50}")
        _, cpu_metrics = test_bertopic_performance(
            sample_docs, sample_cats, sample_times, use_cuda=False
        )

        print(f"\n{'=' * 50}")
        print(f"Testing with {size} documents on GPU")
        print(f"{'=' * 50}")
        _, gpu_metrics = test_bertopic_performance(
            sample_docs, sample_cats, sample_times, use_cuda=True
        )

        speedup = cpu_metrics["total_time_seconds"] / gpu_metrics["total_time_seconds"]

        results.append(
            {
                "sample_size": size,
                "cpu_time": cpu_metrics["total_time_seconds"],
                "gpu_time": gpu_metrics["total_time_seconds"],
                "speedup": speedup,
            }
        )

    # Print comparison results
    print("\nPerformance Comparison Summary:")
    print(
        f"{'Sample Size':<15} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<10}"
    )
    print("-" * 55)

    for result in results:
        print(
            f"{result['sample_size']:<15} {result['cpu_time']:.2f}s{' ':<9} {result['gpu_time']:.2f}s{' ':<9} {result['speedup']:.2f}x"
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Check for CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))

    # Load custom dataset
    documents, categories, timestamps = load_custom_dataset(
        corpus_path="./big_corpus.txt",
        categories_path="./big-categories.txt",
        dates_path="./big_dates.txt",
        n_samples=20000,
    )

    # Option 1: Run a single performance test with GPU
    topic_model, metrics = test_bertopic_performance(
        documents, categories, timestamps, use_cuda=True
    )

    # Option 2: Compare CPU vs GPU performance
    comparison_df = compare_cpu_vs_gpu(
        documents, categories, timestamps, sample_sizes=[1000, 5000, 10000]
    )

    # Save results
    comparison_df.to_csv("bertopic_performance_comparison.csv", index=False)
    print("\nResults saved to bertopic_performance_comparison.csv")
