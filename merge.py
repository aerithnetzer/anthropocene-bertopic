from bertopic import BERTopic
import glob
from sentence_transformers import SentenceTransformer
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
# Identify models to merge

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
bertopic_models = [
    "topic-model-1",
    "topic-model-2",
    "topic-model-3",
    "topic-model-4",
    "topic-model-5",
    "topic-model-6",
]

merged_model = BERTopic.merge_models(bertopic_models)

topic_model = merged_model

batch_number = "999"

print("visualizing topics")
topic_model.visualize_topics().write_html(f"topics-{batch_number}.html")

print("visualizing Hierarchy")
topic_model.visualize_hierarchy(top_n_topics=50).write_html(
    f"hierarchy-{batch_number}.html"
)

print("Visualizing heatmap")
topic_model.visualize_heatmap(top_n_topics=50).write_html(
    f"heatmap-{batch_number}.html"
)

print("Visualizing barchart")
topic_model.visualize_barchart(top_n_topics=50).write_html(
    f"barchart-{batch_number}.html"
)
