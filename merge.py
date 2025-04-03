import cudf
import glob
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load pre-trained SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load BERTopic models
bertopic_models = [
    "v2_viz/large-bertopic-test-1",
    "v2_viz/large-bertopic-test-2",
    "v2_viz/large-bertopic-test-3",
    "v2_viz/large-bertopic-test-4",
    "v2_viz/large-bertopic-test-5",
    "v2_viz/large-bertopic-test-6",
    "v2_viz/large-bertopic-test-7",
]

# Get all .h5 files
h5_files = [
    "v2_viz/cleaned_text1.h5",
    "v2_viz/cleaned_text2.h5",
    "v2_viz/cleaned_text3.h5",
    "v2_viz/cleaned_text4.h5",
    "v2_viz/cleaned_text5.h5",
    "v2_viz/cleaned_text6.h5",
    "v2_viz/cleaned_text7.h5",
]  # Adjust path if needed

# Initialize empty lists for documents, categories, and dates
documents = []
categories = []
dates = []
titles = []
# Read each HDF5 file
for h5_file in h5_files:
    df = cudf.read_hdf(h5_file, key="df")  # Assuming key="df" in each file

    # Append data to lists
    documents.extend(df["cleaned_text"].to_arrow().to_pylist())
    categories.extend(df["tdmCategory"].to_arrow().to_pylist())
    dates.extend(df["datePublished"].to_arrow().to_pylist())
    titles.extend(df["title"].to_arrow().to_pylist())

# Check dataset sizes
print(len(dates), len(documents), len(categories))

# Merge BERTopic models
model = BERTopic(verbose=True).load(
    "v2_viz/big-merged-model"
)  # model = BERTopic().merge_models(loaded_models)
# model.update_topics(documents)
# representative_docs = model.get_representative_docs()
# print(representative_docs)
#
# for doc in representative_docs:
#     if representative_docs in documents:
#         index = documents.index(doc)
#         print(titles[index])
# Read the corpus file and find line numbers of representative documents
# Define batch number
batch_number = "999"

topics = model.get_topics()
print("Topics:", len(topics))

# Generate visualizations
print("Generating visualizations...")
model.visualize_topics(top_n_topics=20).write_html(f"v2_viz/topics-{batch_number}.html")

print("Calculating topics over time")
topics_over_time = model.topics_over_time(documents, dates, nr_bins=50)
model.visualize_topics_over_time(
    topics_over_time, top_n_topics=50, normalize_frequency=True
).write_html(f"v2_viz/topics_over_time-batch{batch_number}.html")

print("Calculating topics per category")
topics_per_class = model.topics_per_class(docs=documents, classes=categories)
model.visualize_topics_per_class(
    topics_per_class, top_n_topics=50, normalize_frequency=True
).write_html(f"v2_viz/categories-{batch_number}.html")

model.visualize_hierarchy(top_n_topics=20).write_html(
    f"v2_viz/hierarchy-{batch_number}.html"
)
model.visualize_heatmap(top_n_topics=20).write_html(
    f"v2_viz/heatmap-{batch_number}.html"
)
model.visualize_barchart(top_n_topics=20).write_html(
    f"v2_viz/barchart-{batch_number}.html"
)

print("Processing complete.")
