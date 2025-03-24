import torch
from tqdm import tqdm
import time
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import boto3
import pandas as pd
from typing import List, Dict, Any
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

print("Modules loaded")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


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


def clean_texts_parallel(texts: List[str], max_workers: int = 4) -> List[str]:
    """
    Clean a list of texts in parallel.

    Args:
        texts: List of texts to clean
        max_workers: Maximum number of workers for parallelization (default: CPU count)

    Returns:
        List of cleaned texts
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        cleaned_texts = list(executor.map(clean_text, texts))

    return cleaned_texts


def load_jsonl_from_s3(bucket_name: str, prefix: str = "constellate/batch-1"):
    """Recursively loads all JSONL files from an S3 bucket with the given prefix and returns a DataFrame.

    Args:
        bucket_name: The name of the S3 bucket
        prefix: The prefix to filter objects by (default: "constellate/")

    Returns:
        A DataFrame containing the data from the first JSONL file with columns: fullText, TDMCategory, and datePublished
    """
    s3_client = boto3.client("s3")
    documents = []

    # List all objects with the given prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    documents = []
    dates = []
    categories = []
    for page in pages:
        if "Contents" not in page:
            continue

        for obj in [page["Contents"][0]]:
            key = obj["Key"]
            if key.endswith(".jsonl"):
                print(f"Processing: {key}")
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = response["Body"].read().decode("utf-8")
                this_df = pd.read_json(content, lines=True)
                print(this_df["fullText"].head())
                for index, row in this_df.iterrows():
                    for part in this_df["fullText"]:
                        if len(part) > 0:
                            documents.append(part)
                            categories.append(this_df["tdmCategory"][index])
                            dates.append(this_df["datePublished"][index])
    return documents, dates, categories


def main():
    # Configure S3 bucket information
    bucket_name = "anthropocene-data"  # Replace with your bucket name
    prefix = "constellate/batch-1"

    # Load documents from S3
    documents, dates, categories = load_jsonl_from_s3(bucket_name, prefix)

    # Clean the data
    print("Cleaning data...")
    documents = clean_texts_parallel(documents)
    # Configure UMAP for dimensionality reduction
    # umap_model = UMAP(
    #     n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42
    # )
    #
    # # Configure HDBSCAN for clustering
    # hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True)

    # Configure CountVectorizer for topic representation

    # Initialize representation model for better topic representations

    # Initialize BERTopic with components and class labels
    # topic_model = BERTopic(
    #     embedding_model=embedding_model,
    #     umap_model=umap_model,
    #     hdbscan_model=hdbscan_model,
    #     verbose=True,
    # )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        verbose=True,
        calculate_probabilities=True,
        language="english",
    )

    topic_model = topic_model.fit_transform(documents)
    # Get the classes from the DataFrame
    print("\nTopic model info:")

    topic_model.save("topic_model")


if __name__ == "__main__":
    main()
