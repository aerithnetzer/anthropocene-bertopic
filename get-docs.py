from bertopic import BERTopic
import argparse


parser = argparse.ArgumentParser(prog="Get representative docs")
parser.add_argument("-m", "--model-number")
parser.add_argument("-t", "--topic_num")


def main():
    args = parser.parse_args()
    model_number = args.model_number
    topic_number = args.topic_num

    topic_model = BERTopic(verbose=True).load(f"./topic_model_batch_{model_number}")
    representative_docs = topic_model.get_representative_docs(topic_number)
    print(representative_docs)


main()

