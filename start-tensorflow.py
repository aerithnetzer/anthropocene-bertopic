import tensorflow as tf
from tensorboard.plugins.projector import ProjectorConfig
import os

# Create a directory for TensorBoard logs
log_dir = "./tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)

if not os.path.exists(log_dir):
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

        # Create a TensorFlow SummaryWriter
        writer = tf.summary.create_file_writer(log_dir)
        config = ProjectorConfig()

        # Add embedding to the projector
        embedding = config.embeddings.add()
        embedding.tensor_name = f"embeddings_{decade}"
        embedding.metadata_path = f"metadata_{decade}.tsv"

        # Save the projector configuration
        projector_config_path = os.path.join(log_dir, "projector_config.pbtxt")
        with writer.as_default():
            tf.summary.text("projector_config", str(config), step=0)
