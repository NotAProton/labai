"""Simple Amazon Bedrock Moonshot example using the Converse API."""

import os
import sys

import boto3
from botocore.exceptions import ClientError


REGION = os.getenv("BEDROCK_REGION", "ap-south-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "moonshotai.kimi-k2.5")
PROMPT = os.getenv(
    "BEDROCK_PROMPT",
    "Describe the purpose of a 'hello world' program in one line.",
)


def extract_text(response):
    content = response["output"]["message"].get("content", [])
    return "\n".join(block["text"] for block in content if "text" in block).strip()


def main():
    client = boto3.client("bedrock-runtime", region_name=REGION)

    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": PROMPT}],
                }
            ],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.5,
            },
        )
    except ClientError as error:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {error}")
        sys.exit(1)
    except Exception as error:
        print(f"ERROR: Unexpected failure while calling Bedrock. Reason: {error}")
        sys.exit(1)

    print(extract_text(response))


if __name__ == "__main__":
    main()


