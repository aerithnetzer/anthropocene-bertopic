from bertopic import BERTopic
import glob
from sentence_transformers import SentenceTransformer
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
# Identify models to merge

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
bertopic_models = [
    "topic_model_batch_1",
    "topic_model_batch_2",
    "topic_model_batch_3",
    "topic_model_batch_4",
    "topic_model_batch_5",
    "topic_model_batch_6",
]

merged_model = BERTopic().merge_models(bertopic_models)

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
