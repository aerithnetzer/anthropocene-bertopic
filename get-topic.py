from bertopic import BERTopic
import argparse
import pandas as pd

# Set up argument parsing
parser = argparse.ArgumentParser(prog="Get representative docs")
parser.add_argument(
    "-m", "--model-number", required=True, type=str, help="Model number"
)
parser.add_argument("-t", "--topic-num", required=True, type=int, help="Topic number")


def main():
    args = parser.parse_args()
    topic_number = args.topic_num

    # Load the BERTopic model
    topic_model = BERTopic(verbose=True).load("./v2_viz/big-merged-model")

    print(topic_model.get_topic(topic_number))


main()
