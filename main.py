from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import numpy as np


def main():
    # Create sample documents
    docs = [
        "document about sports",
        "document about politics",
        "document about science",
        "document about technology",
    ]

    # Configure RAPIDS-based UMAP for GPU acceleration
    umap_model = UMAP(
        n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42
    )

    # Configure RAPIDS-based HDBSCAN for GPU acceleration
    hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True)

    # Initialize BERTopic with CUDA-accelerated components
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)

    # Fit the model on documents
    topics, probs = topic_model.fit_transform(docs)

    print("Topics:", topics)
    print("Topic model info:")
    print(topic_model.get_topic_info())


if __name__ == "__main__":
    main()
