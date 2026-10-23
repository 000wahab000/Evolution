import logging
import os
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

logger = logging.getLogger(__name__)

BUCKET = os.environ.get("S3_BUCKET", "")


def get_s3_client():
    if not BUCKET or boto3 is None:
        return None
    try:
        return boto3.client("s3")
    except Exception as e:
        logger.warning("Failed to initialize boto3 S3 client: %s", e)
        return None


def pull_models():
    """Pull saved model checkpoints and metadata from S3 if configured."""
    if not BUCKET:
        return

    s3 = get_s3_client()
    if not s3:
        return

    os.makedirs("models", exist_ok=True)
    target_keys = [
        "best_model.pth",
        "universal_best.pth",
        "universal_best.json",
        "top_0.pth",
        "top_1.pth",
        "top_2.pth",
        "top_3.pth",
        "top_4.pth",
    ]

    for filename in target_keys:
        s3_key = f"models/{filename}"
        local_path = os.path.join("models", filename)
        try:
            s3.download_file(BUCKET, s3_key, local_path)
            logger.info("Downloaded %s from s3://%s/%s", local_path, BUCKET, s3_key)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey"):
                # Object does not exist yet in S3
                continue
            logger.warning("ClientError downloading s3://%s/%s: %s", BUCKET, s3_key, e)
        except Exception as e:
            logger.warning("Error downloading s3://%s/%s: %s", BUCKET, s3_key, e)


def push_model(local_path: str, s3_key: str):
    """Upload a saved model checkpoint or metadata file to S3."""
    if not BUCKET or not os.path.exists(local_path):
        return

    s3 = get_s3_client()
    if not s3:
        return

    try:
        s3.upload_file(local_path, BUCKET, s3_key)
        logger.info("Uploaded %s to s3://%s/%s", local_path, BUCKET, s3_key)
    except Exception as e:
        logger.warning("Failed to upload %s to s3://%s/%s: %s", local_path, BUCKET, s3_key, e)
