import os
from gensim.models.word2vec import Word2Vec
import tensorflow as tf
from tensorboard.plugins.projector import ProjectorConfig

# Group documents by decade
documents_by_decade = {}

# Read corpus and dates


with (
    open("./corpus-6.txt", "r") as corpus_file,
    open("./categories-6.txt", "r") as dates_file,
):
    for line, date in zip(corpus_file, dates_file):
        year = date.strip().split("-")[0]  # Extract the year from the date
        decade = (int(year) // 10) * 10  # Calculate the decade
        if decade not in documents_by_decade:
            documents_by_decade[decade] = []
        documents_by_decade[decade].append(line.split())

# Create output directory for embeddings and metadata
output_dir = "./word2vec_outputs"
os.makedirs(output_dir, exist_ok=True)

log_dir = "./tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)

# Train and save Word2Vec models, embeddings, and metadata
for decade, documents in documents_by_decade.items():
    model = Word2Vec(
        sentences=documents, vector_size=100, window=5, min_count=1, workers=4
    )
    model_path = os.path.join(output_dir, f"word2vec_{decade}.model")
    model.save(model_path)

    # Save embeddings and metadata
    embeddings_path = os.path.join(log_dir, f"embeddings_{decade}.tsv")
    metadata_path = os.path.join(log_dir, f"metadata_{decade}.tsv")
    with (
        open(embeddings_path, "w") as embeddings_file,
        open(metadata_path, "w") as metadata_file,
    ):
        for word in model.wv.index_to_key:
            vector = model.wv[word]
            embeddings_file.write("\t".join(map(str, vector)) + "\n")
            metadata_file.write(word + "\n")

    # Create a TensorFlow variable for embeddings
    embeddings_var = tf.Variable(model.wv.vectors, name=f"embeddings_{decade}")

    # Create a TensorFlow SummaryWriter
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        tf.summary.experimental.write_raw_pb(
            tf.convert_to_tensor(embeddings_var), step=0
        )

    # Configure the projector
    config = ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings_var.name
    embedding.metadata_path = f"metadata_{decade}.tsv"

    # Save the projector configuration
    projector_config_path = os.path.join(log_dir, "projector_config.pbtxt")
    with open(projector_config_path, "w") as f:
        f.write(str(config))
        # Create a TensorFlow SummaryWriter
        writer = tf.summary.create_file_writer(log_dir)
        config = ProjectorConfig()

        # Add embedding to the projector
        embedding = config.embeddings.add()
        embedding.tensor_name = f"embeddings_{decade}.tsv"
        embedding.metadata_path = f"metadata_{decade}.tsv"

        # Save the projector configuration
        projector_config_path = os.path.join(log_dir, "projector_config.pbtxt")
        with writer.as_default():
            tf.summary.text("projector_config", str(config), step=0)
