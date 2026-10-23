import os
from unittest.mock import MagicMock, patch
from src.s3_sync import pull_models, push_model


def test_s3_sync_noop_when_no_bucket():
    with patch.dict(os.environ, {"S3_BUCKET": ""}, clear=False):
        # Should not raise any error
        pull_models()
        push_model("dummy_path.pth", "models/dummy_path.pth")


def test_s3_sync_push_and_pull_mocked(tmp_path):
    mock_client = MagicMock()
    with patch("src.s3_sync.BUCKET", "test-bucket"), \
         patch("src.s3_sync.get_s3_client", return_value=mock_client):
        
        # Test push
        test_file = tmp_path / "test_model.pth"
        test_file.write_text("weights")
        push_model(str(test_file), "models/test_model.pth")
        mock_client.upload_file.assert_called_once_with(str(test_file), "test-bucket", "models/test_model.pth")

        # Test pull
        pull_models()
        assert mock_client.download_file.call_count > 0
