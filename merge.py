from PIL.Image import merge
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

merged_model.save("./merged1-6model")

doc_files = glob.glob("../document*.txt")

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
