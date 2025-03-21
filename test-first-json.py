import torch
import time
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import cuml
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import json
import boto3
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from datetime import datetime


def download_nltk_resources():
    """Download necessary NLTK resources if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("punkt")
        nltk.download("stopwords")


def clean_text(text: str) -> str:
    """
    Clean text by removing punctuation, numbers, and stopwords.

    Args:
        text: The text to clean

    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    # Rejoin tokens
    return " ".join(tokens)


def load_first_jsonl_from_s3(
    bucket_name: str, prefix: str = "constellate/"
) -> List[Dict[Any, Any]]:
    """
    Loads the first JSONL file from an S3 bucket with the given prefix.

    Args:
        bucket_name: The name of the S3 bucket
        prefix: The prefix to filter objects by (default: "constellate/")

    Returns:
        A list of dictionaries containing the data from the first JSONL file
    """
    s3_client = boto3.client("s3")
    documents = []

    # List all objects with the given prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    first_jsonl_key = None
    for page in pages:
        if "Contents" not in page:
            continue

        # Find the first JSONL file
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".jsonl"):
                first_jsonl_key = key
                break

        if first_jsonl_key:
            break

    if not first_jsonl_key:
        print("No JSONL files found in the specified bucket and prefix")
        return []

    # Process the first JSONL file
    print(f"Processing: {first_jsonl_key}")
    response = s3_client.get_object(Bucket=bucket_name, Key=first_jsonl_key)
    content = response["Body"].read().decode("utf-8")

    # Process each line in the JSONL file
    for line in content.strip().split("\n"):
        if line:
            try:
                doc = json.loads(line)
                if all(k in doc for k in ["datePublished", "TDMCategory", "fullText"]):
                    documents.append(doc)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON line in {first_jsonl_key}")

    return documents


def test_bertopic_performance(
    texts, categories, dates, use_cuda=True, embedding_model="all-MiniLM-L6-v2"
):
    """
    Test BERTopic performance with RAPIDS GPU implementations.

    Args:
        texts: List of text documents
        categories: List of categories for supervised learning
        dates: List of dates for temporal analysis
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

    # Create dimensionality reduction model using RAPIDS cuML UMAP
    print("Initializing RAPIDS cuML UMAP")
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    # Create clustering model using RAPIDS cuML HDBSCAN
    print("Initializing RAPIDS cuML HDBSCAN")
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Create BERTopic model with RAPIDS components
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True,
    )

    # Measure performance
    start_time = time.time()

    # Fit the model with category supervision
    topics, probs = topic_model.fit_transform(texts, y=categories)

    end_time = time.time()
    processing_time = end_time - start_time

    # Get results
    performance_metrics = {
        "device": device,
        "num_documents": len(texts),
        "total_time_seconds": processing_time,
        "time_per_document_ms": (processing_time / len(texts)) * 1000,
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
    for topic_id, topic_info in topic_model.get_topic_info().iloc[:10].iterrows():
        print(
            f"Topic {topic_info['Topic']}: {topic_info['Count']} documents - {topic_info['Name']}"
        )

    # Perform time-based analysis
    try:
        timestamps_pd = pd.to_datetime(dates)
        # Generate topics over time visualization data
        topics_over_time = topic_model.topics_over_time(texts, topics, timestamps_pd)
        print(
            f"\nTemporal analysis complete. Found {len(topics_over_time)} data points."
        )

        # Save temporal analysis to CSV
        topics_over_time.to_csv("topics_over_time.csv", index=False)
        print("Temporal analysis saved to topics_over_time.csv")
    except Exception as e:
        print(f"Error in time-based analysis: {e}")

    return topic_model, performance_metrics


def main():
    # Download NLTK resources
    download_nltk_resources()

    # Check for CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("RAPIDS cuML version:", cuml.__version__)

    # Configure S3 bucket information
    bucket_name = "anthropocene-data"  # Replace with your actual bucket name
    prefix = "constellate/"

    # Load the first JSONL file from S3
    documents = load_first_jsonl_from_s3(bucket_name, prefix)
    print(f"Loaded {len(documents)} documents from the first JSONL file")

    if not documents:
        print("No documents found. Exiting.")
        return

    # Extract text, categories, and dates
    texts = [clean_text(doc["fullText"]) for doc in documents]
    categories = [
        doc["TDMCategory"][1] if len(doc["TDMCategory"]) > 1 else "Unknown"
        for doc in documents
    ]
    dates = [doc["datePublished"] for doc in documents]

    print(f"Processing {len(texts)} documents")
    print(f"Unique categories: {len(set(categories))}")

    # Run BERTopic performance test
    topic_model, metrics = test_bertopic_performance(
        texts, categories, dates, use_cuda=True
    )

    # Save the model and results
    topic_model.save("bertopic_model")
    print("Model saved to bertopic_model")

    # Save document topics
    result_df = pd.DataFrame(
        {
            "text": texts[
                :100
            ],  # Just save first 100 texts to keep file size reasonable
            "topic": topic_model.topics_[:100],
            "category": categories[:100],
            "date": dates[:100],
        }
    )
    result_df.to_csv("topic_results.csv", index=False)
    print("Results saved to topic_results.csv")


if __name__ == "__main__":
    main()
