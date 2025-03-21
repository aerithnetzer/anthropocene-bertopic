import boto3
from bertopic import BERTopic
from tqdm import tqdm
import json
from cudf import pd


def load_first_jsonl_from_s3(
    bucket_name: str = "anthropocene-data", prefix: str = "constellate/"
):
    """
    Recursively loads all JSONL files from an S3 bucket with the given prefix.
    Args:
        bucket_name: The name of the S3 bucket
        prefix: The prefix to filter objects by (default: "constellate/")
    Returns:
        A list of dictionaries containing the data from all JSONL files
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    jsonl_data = []
    for response in response_iterator:
        if "Contents" in response:
            for obj in response["Contents"]:
                if obj["Key"].endswith(".jsonl"):
                    # Load the JSONL file
                    obj_response = s3_client.get_object(
                        Bucket=bucket_name, Key=obj["Key"]
                    )
                    jsonl_content = obj_response["Body"].read().decode("utf-8")
                    df = pd.read_json(jsonl_content, lines=True)
                    print(df.columns())
    return jsonl_data


def main():
    load_first_jsonl_from_s3()
