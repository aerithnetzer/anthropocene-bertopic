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


def load_jsonl_from_s3(bucket_name: str, prefix: str = "constellate/") -> pd.DataFrame:
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

    df = pd.DataFrame(columns=["fullText", "TDMCategory", "datePublished"])
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
                this_df = this_df[["fullText", "tdmCategory", "datePublished"]]
                pd.concat([df, this_df], ignore_index=True)
                break

    df.to_csv("constellate.csv", index=False)
    return df


def main():
    # Configure S3 bucket information
    bucket_name = "anthropocene-data"  # Replace with your bucket name
    prefix = "constellate/"

    # Load documents from S3
    df = load_jsonl_from_s3(bucket_name, prefix)

    # Clean the data
    print("Cleaning data...")
    df["fullText"] = df["fullText"].astype(str)
    df["fullText"] = df["fullText"].apply(clean_text)
    df["fullText"] = df["fullText"].astype(str)
    docs = df["fullText"].tolist()
    print(docs)
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

    # Fit the model with class information
    docs = df["fullText"].tolist()

    topic_model = topic_model.fit(docs)
    # Get the classes from the DataFrame
    categories = df["TDMCategory"].tolist()
    # Get the dates from the DataFrame
    dates = df["datePublished"].tolist()
    # Get topic information
    print("\nTopic model info:")

    # Get class-topic mappings
    class_topic_mapping = topic_model.topics_per_class(docs, categories)
    class_date_mapping = topic_model.topics_per_class(docs, dates)

    print("\nClass-topic mapping:")
    for class_name, topics_data in class_topic_mapping.items():
        print(f"\nClass: {class_name}")
        for topic_data in topics_data[:3]:  # Show top 3 topics per class
            print(f"Topic {topic_data[0]}: {topic_data[1]}")
    topic_model.visualize_documents(docs).write_html("visualize_documents.html")
    topics_per_class = topic_model.topics_per_class(docs, categories)
    meow = topic_model.topics_over_time(docs, dates)

    #     # Save the visualization
    topics_over_time.visualize_to.write_html("topics_over_time.html")
    # Optional: Save results
    topic_model.save("topic_model")


if __name__ == "__main__":
    main()
