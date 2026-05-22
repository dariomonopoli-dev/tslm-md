try:
    from sagemaker.inputs import TrainingInput
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False
    TrainingInput = None

def get_training_input(s3_uri: str):
    """Create TrainingInput with FastFile mode for streaming data from S3."""
    if not SAGEMAKER_AVAILABLE:
        raise ImportError("SageMaker SDK not available. Install with: pip install sagemaker")
    return TrainingInput(
        s3_data=s3_uri,
        input_mode="FastFile"
    )

# Default S3 URI for the dataset
DEFAULT_S3_URI = "s3://sagemaker-us-west-2-094487995066/datasets/MD.hdf5"