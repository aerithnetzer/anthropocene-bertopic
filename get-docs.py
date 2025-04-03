from bertopic import BERTopic
import argparse
import pandas as pd
import cudf

# Set up argument parsing
parser = argparse.ArgumentParser(prog="Get representative docs")
parser.add_argument(
    "-m", "--model-number", required=True, type=str, help="Model number"
)
parser.add_argument("-t", "--topic-num", required=True, type=int, help="Topic number")


def main():
    args = parser.parse_args()
    model_number = args.model_number
    topic_number = args.topic_num

    # Load the BERTopic model
    topic_model = BERTopic(verbose=True).load(f"./topic_model_batch_{model_number}")

    # Get representative documents
    representative_docs = topic_model.get_representative_docs(topic_number)
    print(representative_docs)

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

    for doc in representative_docs:
        if representative_docs in documents:
            index = documents.index(doc)
            print(titles[index])
    # Read the corpus file and find line numbers of representative documents


main()
