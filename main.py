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
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
import concurrent.futures

# Precompile regex patterns
PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
NUMBERS_PATTERN = re.compile(r"\d+")

# Load stopwords and English words as frozensets for faster lookup
STOP_WORDS = frozenset(stopwords.words("english"))
ENGLISH_WORDS = frozenset(words.words())
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
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = NUMBERS_PATTERN.sub(" ", text)

    # Tokenize and filter
    tokens = word_tokenize(text)
    clean_tokens = [
        token for token in tokens if token in ENGLISH_WORDS and token not in STOP_WORDS
    ]

    return " ".join(clean_tokens)


def clean_texts_parallel(texts: List[str], max_workers=None) -> List[str]:
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


def fetch_jsonl(s3_client, bucket_name, key):
    """Fetches a JSONL file from S3 and extracts relevant fields."""
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    content = response["Body"].read().decode("utf-8")
    df = pd.read_json(content, lines=True)

    # Extract relevant columns and preprocess
    df["fullText"] = (
        df["fullText"]
        .astype(str)
        .str.split()
        .apply(lambda x: " ".join(x[:100]).replace("\n", " "))
    )

    return df[["fullText", "tdmCategory", "datePublished"]]


def load_jsonl_from_s3(bucket_name: str, prefix: str = "constellate/"):
    """
    Recursively loads all JSONL files from an S3 bucket with the given prefix and returns lists.

    Args:
        bucket_name: The name of the S3 bucket
        prefix: The prefix to filter objects by (default: "constellate/batch-3")

    Returns:
        Lists containing fullText, TDMCategory, and datePublished values
    """
    s3_client = boto3.client("s3")

    # List all JSONL files with the given prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    jsonl_keys = [
        obj["Key"]
        for page in pages
        if "Contents" in page
        for obj in page["Contents"]
        if obj["Key"].endswith(".jsonl")
    ]

    documents, categories, dates = [], [], []

    # Load files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(fetch_jsonl, s3_client, bucket_name, key): key
            for key in jsonl_keys
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing files",
        ):
            try:
                df = future.result()
                documents.extend(df["fullText"].tolist())
                categories.extend(df["tdmCategory"].tolist())
                dates.extend(df["datePublished"].tolist())
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    return documents, categories, dates


def main(batch_number: int = 0):
    # Configure S3 bucket information
    bucket_name = "anthropocene-data"  # Replace with your bucket name
    prefix = f"constellate/batch-{batch_number}/"

    # Load documents from S3
    documents, dates, categories = load_jsonl_from_s3(bucket_name, prefix)

    # Clean the data
    print("Cleaning data...")
    documents = clean_texts_parallel(documents, max_workers=None)

    umap_model = UMAP(
        n_components=2, n_neighbors=15, min_dist=0.0, random_state=42, verbose=True
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=15, min_samples=1, prediction_data=True, verbose=True
    )

    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    embeddings = normalize(embeddings)

    reduced_embeddings = umap_model.fit_transform(embeddings)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )

    topic_model = topic_model.fit(documents, embeddings)

    topic_model.save(f"topic_model_batch_{batch_number}")
    with open(f"corpus-{batch_number}.txt", "w") as f:
        for document in tqdm(documents, desc="Writing corpora"):
            f.write(str(document) + "\n")

    with open(f"categories-{batch_number}.txt", "w") as f:
        for category in tqdm(categories, desc="Writing categories"):
            f.write(str(category) + "\n")

    with open(f"dates-{batch_number}.txt", "w") as f:
        for date in tqdm(dates, desc="Writing dates"):
            f.write(str(date) + "\n")

    # topic_model.visualize_documents(
    #     docs=documents,
    #     sample=0.05,
    #     embeddings=embeddings,
    #     reduced_embeddings=reduced_embeddings,
    # ).write_html(f"documents_batch-{batch_number}.html")
    #
    # topic_model.visualize_topics().write_html(f"topics-{batch_number}.html")
    # topic_model.visualize_hierarchy().write_html(f"hierarchy-{batch_number}.html")
    # topic_model.visualize_heatmap().write_html(f"heatmap-{batch_number}.html")
    # topic_model.visualize_barchart().write_html(f"barchart-{batch_number}.html")
    #
    # topics_over_time = topic_model.topics_over_time(documents, dates, nr_bins=100)
    # topics_over_time.visualize_topics_over_time().write_html(
    #     f"topics_over_time-batch{batch_number}.html"
    # )
    # topics_per_category = topic_model.topics_per_class(documents, categories)
    # topics_per_category.visualize_topics_per_class().write_html(
    #     f"topics_per_category{batch_number}.html"
    # )


if __name__ == "__main__":
    main(batch_number=2)
