from bertopic import BERTopic
import glob
from pandas.core.dtypes.common import classes
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

loaded_models = []

for model in bertopic_models:
    model = BERTopic(verbose=True).load(model)
    loaded_models.append(model)


corpora_docs = [
    "corpus-1.txt",
    "corpus-2.txt",
    "corpus-3.txt",
    "corpus-4.txt",
    "corpus-5.txt",
    "corpus-6.txt",
]

category_docs = [
    "dates-1.txt",
    "dates-2.txt",
    "dates-3.txt",
    "dates-4.txt",
    "dates-5.txt",
    "dates-6.txt",
]

date_docs = [
    "categories-1.txt",
    "categories-2.txt",
    "categories-3.txt",
    "categories-4.txt",
    "categories-5.txt",
    "categories-6.txt",
]

documents = []
for corpus_doc in corpora_docs:
    with open(corpus_doc, "r") as f:
        for line in f.readlines():
            documents.append(line)
categories = []
for category_doc in category_docs:
    with open(category_doc, "r") as f:
        for line in f.readlines():
            categories.append(
                line.strip().replace("[", "").replace("]", "").split(",")[0].strip("' ")
            )

dates = []
for date_doc in date_docs:
    with open(date_doc, "r") as f:
        for line in f.readlines():
            dates.append(line)

print(len(dates))
print(len(documents))
print(len(categories))

print("Date: ", dates[0])
print("Doc: ", documents[0])
print("Category: ", categories[0])

model = BERTopic().merge_models(loaded_models)

model.update_topics(documents)

topic_model = model

batch_number = "999"

print("visualizing topics")
topic_model.visualize_topics(top_n_topics=20).write_html(f"topics-{batch_number}.html")

topics = list(range(1, 21))
print("Visualizing topics over time")

print("Calculating topics over time")
topics_over_time = topic_model.topics_over_time(documents, dates, nr_bins=50)

print("Visualizing topics over time")
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=50).write_html(
    f"topics_over_time-batch{batch_number}.html"
)

print("Calculating topics per category")
topics_per_class = topic_model.topics_per_class(docs=documents, classes=categories)
topic_model.visualize_topics_per_class(
    topics_per_class, top_n_topics=50, normalize_frequency=True
)

print("visualizing Hierarchy")
topic_model.visualize_hierarchy(top_n_topics=20).write_html(
    f"hierarchy-{batch_number}.html"
)

print("Visualizing heatmap")
topic_model.visualize_heatmap(top_n_topics=20).write_html(
    f"heatmap-{batch_number}.html"
)

print("Visualizing barchart")
topic_model.visualize_barchart(top_n_topics=20).write_html(
    f"barchart-{batch_number}.html"
)
