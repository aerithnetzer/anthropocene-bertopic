"""
Large-Scale BERTopic Analysis Pipeline for 4 Million Documents

Features:
- Batch processing for scalability
- Embedding generation with SentenceTransformer
- BERTopic modeling with UMAP and HDBSCAN
- Topics over time analysis
- Topics per class analysis
- Incremental topic updates

Requirements:
- bertopic
- sentence-transformers
- umap-learn
- hdbscan
- pandas
- numpy
"""

import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from typing import List, Tuple

# ---------------------------
# Configuration Parameters
# ---------------------------

BATCH_SIZE = 100_000  # Number of documents per batch
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
UMAP_PARAMS = {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
}
HDBSCAN_PARAMS = {
    "min_cluster_size": 50,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}
DATA_PATH = "path_to_your_data"  # Folder or file path for documents
OUTPUT_PATH = "output"  # Folder to save models and results

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------------------
# Helper Functions
# ---------------------------


def load_documents_batch(batch_index: int, batch_size: int) -> pd.DataFrame:
    """
    Load a batch of documents with metadata.
    Expected columns: ['doc_id', 'text', 'timestamp', 'class_label']

    Replace this function with your actual data loading logic.
    """
    # Example: load from CSV in chunks
    # Adjust this to your data source (database, parquet, etc.)
    csv_file = os.path.join(DATA_PATH, "documents.csv")
    skiprows = batch_index * batch_size + 1  # +1 to skip header
    try:
        batch_df = pd.read_csv(
            csv_file,
            skiprows=range(1, skiprows),
            nrows=batch_size,
            usecols=["doc_id", "text", "timestamp", "class_label"],
            parse_dates=["timestamp"],
        )
        if batch_df.empty:
            return None
        return batch_df
    except FileNotFoundError:
        print(f"Data file not found: {csv_file}")
        return None


def generate_embeddings(
    docs: List[str], model: SentenceTransformer
) -> np.ndarray:
    """Generate embeddings for a list of documents."""
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=512)
    return embeddings


def initialize_bertopic() -> BERTopic:
    """Initialize BERTopic with custom UMAP and HDBSCAN."""
    umap_model = UMAP(**UMAP_PARAMS)
    hdbscan_model = HDBSCAN(**HDBSCAN_PARAMS)
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    return topic_model


def save_model(topic_model: BERTopic, batch_index: int):
    """Save BERTopic model to disk."""
    model_path = os.path.join(OUTPUT_PATH, f"bertopic_model_batch_{batch_index}")
    topic_model.save(model_path)
    print(f"Saved BERTopic model to {model_path}")


def load_model(batch_index: int) -> BERTopic:
    """Load BERTopic model from disk."""
    model_path = os.path.join(OUTPUT_PATH, f"bertopic_model_batch_{batch_index}")
    if os.path.exists(model_path):
        topic_model = BERTopic.load(model_path)
        print(f"Loaded BERTopic model from {model_path}")
        return topic_model
    else:
        print(f"Model path does not exist: {model_path}")
        return None


def analyze_topics_over_time(
    topic_model: BERTopic, docs: List[str], timestamps: List[pd.Timestamp]
):
    """Compute and visualize topics over time."""
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.show()
    return topics_over_time


def analyze_topics_per_class(
    docs: List[str], topics: List[int], classes: List[str]
) -> pd.DataFrame:
    """Calculate topic distribution per class."""
    df = pd.DataFrame({"doc": docs, "topic": topics, "class": classes})
    topic_dist_per_class = df.groupby(["class", "topic"]).size().unstack(fill_value=0)
    print("Topic distribution per class:")
    print(topic_dist_per_class)
    return topic_dist_per_class


# ---------------------------
# Main Pipeline
# ---------------------------


def main():
    # Initialize embedding model once
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize empty topic model (will be updated incrementally)
    topic_model = initialize_bertopic()

    batch_index = 0
    while True:
        print(f"\nLoading batch {batch_index}...")
        batch_df = load_documents_batch(batch_index, BATCH_SIZE)
        if batch_df is None or batch_df.empty:
            print("No more data to process. Exiting.")
            break

        docs = batch_df["text"].tolist()
        timestamps = batch_df["timestamp"].tolist()
        classes = batch_df["class_label"].tolist()

        print(f"Generating embeddings for batch {batch_index}...")
        embeddings = generate_embeddings(docs, embedding_model)

        if batch_index == 0:
            # Fit model on first batch
            print("Fitting BERTopic model on first batch...")
            topics, probs = topic_model.fit_transform(docs, embeddings)
        else:
            # Incrementally update model with new batch
            print("Updating BERTopic model with new batch...")
            topics, probs = topic_model.update_topics(docs, embeddings)

        # Save model after each batch
        save_model(topic_model, batch_index)

        # Analyze topics over time for this batch
        print("Analyzing topics over time for this batch...")
        analyze_topics_over_time(topic_model, docs, timestamps)

        # Analyze topics per class for this batch
        print("Analyzing topics per class for this batch...")
        analyze_topics_per_class(docs, topics, classes)

        batch_index += 1


if __name__ == "__main__":
    main()
