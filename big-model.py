from bertopic import BERTopic
import glob
from sentence_transformers import SentenceTransformer
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

corpus = []
timestamps = []
categories = []

corpus_files = glob.glob("corpus-*.txt")

for corpus_file in corpus_files:
    with open(corpus_file, "r") as f:
        corpus.append(f.readlines())

dates_files = glob.glob("dates-*.txt")

for dates_file in dates_files:
    with open(dates_file, "r") as f:
        timestamps.append(f.readlines())

categories_files = glob.glob("categories-*.txt")

for categories_file in categories_files:
    with open(categories_file, "r") as f:
        categories.append(f.readlines())

umap_model = UMAP(
    n_components=2, n_neighbors=15, min_dist=0.0, random_state=42, verbose=True
)

hdbscan_model = HDBSCAN(
    min_cluster_size=15, min_samples=1, prediction_data=True, verbose=True
)

embeddings = embedding_model.encode(corpus, show_progress_bar=True)

embeddings = normalize(embeddings)

reduced_embeddings = umap_model.fit_transform(embeddings)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
)

topic_model = topic_model.fit(corpus, embeddings)

print("visualizing documents")
topic_model.visualize_documents(
    docs=corpus,
    sample=0.05,
    embeddings=embeddings,
    reduced_embeddings=reduced_embeddings,
    topics=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
).write_html("documents_large.html")

print("visualizing topics")
topic_model.visualize_topics().write_html("topics.html")

print("visualizing Hierarchy")
topic_model.visualize_hierarchy(top_n_topics=50).write_html("hierarchy.html")

print("Visualizing heatmap")
topic_model.visualize_heatmap(top_n_topics=50).write_html("heatmap.html")
print("Visualizing barchart")
topic_model.visualize_barchart(top_n_topics=50).write_html("barchart.html")

print("Calculating topics over time")
topics_over_time = topic_model.topics_over_time(corpus, timestamps, nr_bins=50)

print("Visualizing topics over time")
topics_over_time.visualize_topics_over_time(top_n_topics=20).write_html(
    "topics_over_time.html"
)

print("Calculating Topics per category")

topics_per_category = topic_model.topics_per_class(corpus, categories)

print("Visualizing Topics per category")
topics_per_category.visualize_topics_per_class().write_html("topics_per_category.html")
