import cudf
import glob
import nltk
import re
from bertopic import BERTopic
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords, words

# Download necessary NLTK datasets
nltk.download("stopwords")
nltk.download("words")
batch_number = 1


# Get English stopwords and valid English words
stop_words = set(stopwords.words("english"))
valid_words = set(words.words())

print("Reading json files")

# Get all JSONL files in the current directory
jsonl_files = glob.glob("part-*.jsonl")


# Function to clean text
def clean_text(text):
    if isinstance(text, list):  # If it's a list, join it into a single string
        text = " ".join(text)

    if not isinstance(text, str):  # Ensure it's a string
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only English letters and spaces
    words_list = text.split()  # Tokenize
    words_list = [
        word for word in words_list if word in valid_words and word not in stop_words
    ]  # Filter
    return " ".join(words_list)


print("Loading and cleaning data")
# Load and clean data
df_list = []
for file in jsonl_files:
    df = cudf.read_json(file, lines=True)

    # Convert to pandas first, apply the function, then convert back to cuDF
    if "fullText" in df.columns:
        df["cleaned_text"] = df["fullText"].to_pandas().astype(str).apply(clean_text)
        print(df["fullText"].head())
        print(df["cleaned_text"].head())

        # Convert back to cuDF for further processing
        df["cleaned_text"] = cudf.Series(df["cleaned_text"])

        df_list.append(df)

# Combine all files into one DataFrame
df = cudf.concat(df_list, ignore_index=True) if df_list else cudf.DataFrame()
print(df["fullText"].head())
# Convert cuDF column to a pandas list for BERTopic
documents = df["cleaned_text"].to_pandas().dropna().tolist()

print(df["cleaned_text"].head())
# Ensure only relevant columns are kept
columns_to_save = ["title", "tdmCategory", "datePublished", "cleaned_text"]
dates = df["datePublished"]
categories = []
categories = [
    str(category).strip().replace("[", "").replace("]", "").strip("' ")
    for category in df["tdmCategory"].to_pandas()
]

print(categories[0])
print(categories[1])
print(categories[2])
# Check if all columns exist in df (to avoid errors if a column is missing)
existing_columns = [col for col in columns_to_save if col in df.columns]

print("saving columns")
# Save selected columns to HDF5
df[existing_columns].to_pandas().to_hdf(
    f"cleaned_text{batch_number}.h5", key="df", mode="w"
)

print("Saved cleaned DataFrame to cleaned_text.h5 with columns:", existing_columns)
# Load a SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text into embeddings
embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# GPU-accelerated UMAP for dimensionality reduction
umap_model = UMAP(n_components=5, random_state=42)

# GPU-accelerated HDBSCAN for clustering
hdbscan_model = HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)

# Initialize and fit BERTopic with GPU-accelerated models
topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)
topics, probs = topic_model.fit_transform(documents, embeddings)

topic_model.save(f"v2_viz/large-bertopic-test-{batch_number}")
print("visualizing topics")

topic_model.visualize_topics(top_n_topics=20).write_html(f"topics-{batch_number}.html")

topics = list(range(1, 21))
print("Visualizing topics over time")

print("Calculating topics over time")
topics_over_time = topic_model.topics_over_time(
    documents, dates.to_pandas().to_list(), nr_bins=50
)

print("Visualizing topics over time")
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=50).write_html(
    f"v2_viz/topics_over_time-batch{batch_number}.html"
)

print("Calculating topics per category")
topics_per_class = topic_model.topics_per_class(docs=documents, classes=categories)
topic_model.visualize_topics_per_class(
    topics_per_class, top_n_topics=50, normalize_frequency=True
).write_html(f"v2_viz/categories-{batch_number}")

print("visualizing Hierarchy")
topic_model.visualize_hierarchy(top_n_topics=20).write_html(
    f"v2_viz/large-hierarchy-{batch_number}.html"
)

print("Visualizing heatmap")
topic_model.visualize_heatmap(top_n_topics=20).write_html(
    f"v2_viz/large-heatmap-{batch_number}.html"
)

print("Visualizing barchart")
topic_model.visualize_barchart(top_n_topics=20).write_html(
    f"v2_viz/large-barchart-{batch_number}.html"
)
