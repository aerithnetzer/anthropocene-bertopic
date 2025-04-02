import cudf
import glob
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load pre-trained SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load BERTopic models
bertopic_models = [
    "topic_model_batch_1",
    "topic_model_batch_2",
    "topic_model_batch_3",
    "topic_model_batch_4",
    "topic_model_batch_5",
    "topic_model_batch_6",
]

loaded_models = [BERTopic(verbose=True).load(model) for model in bertopic_models]

# Get all .h5 files
h5_files = glob.glob("*.h5")  # Adjust path if needed

# Initialize empty lists for documents, categories, and dates
documents = []
categories = []
dates = []

# Read each HDF5 file
for h5_file in h5_files:
    df = cudf.read_hdf(h5_file, key="df")  # Assuming key="df" in each file

    # Append data to lists
    documents.extend(df["cleaned_txt"].to_arrow().to_pylist())
    categories.extend(df["tdmCategory"].to_arrow().to_pylist())
    dates.extend(df["datePublished"].to_arrow().to_pylist())

# Check dataset sizes
print(len(dates), len(documents), len(categories))
print("Sample Data:")
print("Date:", dates[0])
print("Doc:", documents[0])
print("Category:", categories[0])

# Merge BERTopic models
model = BERTopic().merge_models(loaded_models)
model.update_topics(documents)

# Define batch number
batch_number = "999"

# Generate visualizations
print("Generating visualizations...")
model.visualize_topics(top_n_topics=20).write_html(f"topics-{batch_number}.html")

print("Calculating topics over time")
topics_over_time = model.topics_over_time(documents, dates, nr_bins=50)
model.visualize_topics_over_time(topics_over_time, top_n_topics=50).write_html(
    f"topics_over_time-batch{batch_number}.html"
)

print("Calculating topics per category")
topics_per_class = model.topics_per_class(docs=documents, classes=categories)
model.visualize_topics_per_class(
    topics_per_class, top_n_topics=50, normalize_frequency=True
)

model.visualize_hierarchy(top_n_topics=20).write_html(f"hierarchy-{batch_number}.html")
model.visualize_heatmap(top_n_topics=20).write_html(f"heatmap-{batch_number}.html")
model.visualize_barchart(top_n_topics=20).write_html(f"barchart-{batch_number}.html")

print("Processing complete.")
