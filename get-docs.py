from bertopic import BERTopic
import argparse

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

    # Read the corpus file and find line numbers of representative documents
    corpus_file = f"corpus-{model_number}.txt"
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_lines = f.readlines()

    # Find the line numbers of the representative documents
    doc_line_numbers = [
        i for i, line in enumerate(corpus_lines) if line.strip() in representative_docs
    ]
    print(doc_line_numbers)

    # Read titles file and get corresponding titles
    titles_file = f"titles-{model_number}.txt"
    with open(titles_file, "r", encoding="utf-8") as f:
        title_lines = f.readlines()

    # Print the titles corresponding to the representative documents
    print("\nRepresentative Document Titles:")


main()
