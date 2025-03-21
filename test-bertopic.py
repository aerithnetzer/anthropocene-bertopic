import boto3
from bertopic import BERTopic
from tqdm import tqdm
import json
import pandas as pd
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def load_first_jsonl_from_s3(
    bucket_name: str = "anthropocene-data", prefix: str = "constellate/"
):
    """
    Recursively loads all JSONL files from an S3 bucket with the given prefix.
    Args:
        bucket_name: The name of the S3 bucket
        prefix: The prefix to filter objects by (default: "constellate/")
    Returns:
        A dataframe containing the data from the JSONL files
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    # Create empty DataFrame - use pandas instead of cuDF for initial loading
    anthropocene_dataframe = pd.DataFrame()

    for response in tqdm(response_iterator, desc="Loading JSONL files"):
        if "Contents" in response:
            for obj in response["Contents"]:
                if obj["Key"].endswith(".jsonl"):
                    # Load the JSONL file
                    obj_response = s3_client.get_object(
                        Bucket=bucket_name, Key=obj["Key"]
                    )
                    jsonl_content = obj_response["Body"].read().decode("utf-8")

                    # Use pandas to read the JSONL
                    df = pd.read_json(jsonl_content, lines=True)
                    print(df.columns)

                    # Check for column names with case-insensitive matching
                    date_col = next(
                        (col for col in df.columns if col.lower() == "datepublished"),
                        None,
                    )
                    category_col = next(
                        (col for col in df.columns if col.lower() == "tdmcategory"),
                        None,
                    )
                    text_col = next(
                        (col for col in df.columns if col.lower() == "fulltext"), None
                    )

                    # Only proceed if all required columns exist
                    if date_col and category_col and text_col:
                        # Create a new DataFrame with just the columns we need
                        subset_df = pd.DataFrame(
                            {
                                "datePublished": df[date_col],
                                "tdmCategory": df[category_col],
                                "fullText": df[text_col],
                            }
                        )

                        # Concatenate to the main DataFrame
                        if anthropocene_dataframe.empty:
                            anthropocene_dataframe = subset_df
                        else:
                            anthropocene_dataframe = pd.concat(
                                [anthropocene_dataframe, subset_df], ignore_index=True
                            )
                    else:
                        print(f"Warning: Required columns not found in {obj['Key']}")

                    # Only process the first JSONL file for now
                    break

    return anthropocene_dataframe


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
    tokens = [
        token
        for token in tqdm(tokens, desc="cleaning")
        if token not in stop_words and len(token) > 2
    ]

    # Rejoin tokens
    return " ".join(tokens)


def clean_texts_parallel(
    texts: List[str], max_workers: int = multiprocessing.cpu_count()
) -> List[str]:
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


def train_bertopic(texts, dates, categories):
    """
    Train a BERTopic model on the given texts, dates, and categories.
    Args:
        texts: List of text documents
        dates: List of dates for each document
        categories: List of categories for each document
    Returns:
        Trained BERTopic model
    """
    topic_model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5),
        hdbscan_model=HDBSCAN(min_cluster_size=15),
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(texts, dates, categories)
    return topic_model


def main():
    anthropocene_data = load_first_jsonl_from_s3()
    texts = anthropocene_data["fullText"].astype(str).tolist()
    cleaned_texts = clean_texts_parallel(texts)
    print("Cleaning complete.")
    print("Starting BERTopic...")
    dates = anthropocene_data["datePublished"].astype(str).tolist()
    categories = anthropocene_data["tdmCategory"].astype(str).tolist()
    topic_model = train_bertopic(cleaned_texts, dates, categories)
    topic_model.save("my_model")
    print("BERTopic training complete.")
    print("visualizing...")
    with open("texts.txt", "w") as f:
        for text in cleaned_texts:
            f.write(text + "\n")
    with open("dates.txt", "w") as f:
        for date in dates:
            f.write(date + "\n")
    with open("categories.txt", "w") as f:
        for category in categories:
            f.write(category + "\n")

    topic_model.visualize_topics().write_html("./topics.html")
    topic_model.visualize_documents(texts, sample=0.05).write_html("./documents.html")
    topic_model.visualize_hierarchy().write_html("./hierarchy.html")
    topic_model.visualize_heatmap().write_html("./heatmap.html")
    topics_over_time = topic_model.topics_over_time(docs=texts, timestamps=dates)
    topics_over_time.visualize().write_html("./topics_over_time.html")
    topics_cats = topic_model.topics_per_class(docs=texts, classes=categories)
    topics_cats.visualize().write_html("./topics_per_class.html")


main()
