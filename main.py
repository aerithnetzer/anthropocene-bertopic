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
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize

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


def load_jsonl_from_s3(bucket_name: str, prefix: str = "constellate/"):
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

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".jsonl"):
                print(f"Processing: {key}")
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = response["Body"].read().decode("utf-8")
                this_df = pd.read_json(content, lines=True)
                print(this_df["fullText"].head())
                for index, row in this_df.iterrows():
                    text = str(this_df["fullText"][index])
                    # Split text into chunks of 100 words
                    print(type(text))
                    text = text.split()
                    text = " ".join(text[:100])
                    text = text.replace("\n", " ")
                    documents.append(text)
                    categories.append(this_df["tdmCategory"][index])
                    dates.append(this_df["datePublished"][index])
    return documents, dates, categories


def main():
    # Configure S3 bucket information
    bucket_name = "anthropocene-data"  # Replace with your bucket name
    prefix = "constellate/"

    # Load documents from S3
    documents, dates, categories = load_jsonl_from_s3(bucket_name, prefix)

    # Clean the data
    print("Cleaning data...")
    documents = clean_texts_parallel(documents)
    # Configure UMAP for dimensionality reduction

    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.0, random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=1, prediction_data=True)
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    embeddings = normalize(embeddings)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )

    topic_model = topic_model.fit(
        documents, embeddings, reduced_emeddings=reduced_embeddings
    )

    topic_model.save("topic_model-batch-1")
    with open("corpus.txt", "w") as f:
        for document in documents:
            f.write(f"{document}\n")

    with open("dates.txt", "w") as f:
        for date in dates:
            f.write(f"{date}\n")

    with open("categories.txt", "w") as f:
        for category in categories:
            f.write(f"{category}\n")

    topic_model.visualize_documents(
        docs=documents,
        sample=0.05,
        embeddings=embeddings,
        reduced_embeddings=reduced_embeddings,
    ).write_html("documents_whole_batch.html")
    topic_model.visualize_topics().write_html("topics.html")
    topic_model.visualize_hierarchy().write_html("hierarchy.html")
    topic_model.visualize_heatmap().write_html("heatmap.html")
    topic_model.visualize_barchart().write_html("barchart.html")

    topics_over_time = topic_model.topics_over_time(documents, dates, nr_bins=100)
    topics_over_time.visualize_topics_over_time().write_html("topics_over_time.html")
    topics_per_category = topic_model.topics_per_class(documents, categories)
    topics_per_category.visualize_topics_per_class().write_html(
        "topics_per_category.html"
    )


if __name__ == "__main__":
    main()
