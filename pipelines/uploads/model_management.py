import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component, Model, Output, Dataset, Metrics, Input
from google.cloud import aiplatform

from load_config.config import PROJECT_ID, REGION, BUCKET_NAME

@component
def get_latest_model(model: Output[Model], metrics: Output[Metrics]):
    """Gets the latest model from the models directory."""
    from google.cloud import storage
    import json
    from datetime import datetime

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # List all model directories
    model_dirs = []
    for blob in bucket.list_blobs(prefix="models/"):
        if blob.name.endswith('/model.keras'):
            model_dir = blob.name.split('/model.keras')[0]
            model_dirs.append(model_dir)
    
    if not model_dirs:
        raise Exception("No models found in the bucket")
    
    # Get the latest model directory (sorted by timestamp)
    latest_model_dir = sorted(model_dirs)[-1]
    model.uri = f"gs://{BUCKET_NAME}/{latest_model_dir}/model.keras"
    
    # Get the metrics for this model
    metrics_blob = bucket.blob(f"{latest_model_dir}/metrics.json")
    metrics_data = json.loads(metrics_blob.download_as_string())
    
    for key, value in metrics_data.items():
        metrics.log_metric(key, value)
    
    print(f"Found latest model: {model.uri}")
    print(f"Model metrics: {metrics_data}")

@component
def compare_and_save_best(new_model: Input[Model], metrics: Input[Metrics]):
    """Compares the new model with the best and updates if better."""
    import json
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Get new model metrics
    new_accuracy = metrics.metadata.get('val_accuracy', metrics.metadata.get('accuracy', 0))
    best_model_path = "models/best/model.keras"
    best_metrics_path = "models/best/metrics.json"

    # Load best model metrics if exists
    try:
        blob = bucket.blob(best_metrics_path)
        best_metrics_data = json.loads(blob.download_as_string())
        best_accuracy = best_metrics_data.get('val_accuracy', best_metrics_data.get('accuracy', 0))
    except Exception as e:
        print(f"No existing best model found: {str(e)}")
        best_accuracy = 0

    print(f"New Model Accuracy: {new_accuracy}, Best Model Accuracy: {best_accuracy}")

    if new_accuracy > best_accuracy:
        print("New model is better! Updating best model...")
        
        # Copy model file
        new_model_path = new_model.uri.replace(f"gs://{BUCKET_NAME}/", "")
        source_blob = bucket.blob(new_model_path)
        bucket.copy_blob(source_blob, bucket, best_model_path)

        # Update metrics
        metrics_blob = bucket.blob(best_metrics_path)
        metrics_data = {k: v for k, v in metrics.metadata.items()}
        metrics_blob.upload_from_string(json.dumps(metrics_data))
        
        print("Best model updated successfully")
    else:
        print("Current best model remains superior")

@dsl.pipeline(
    name="model-management-pipeline",
    pipeline_root=f"gs://{BUCKET_NAME}/pipelines"
)
def model_management_pipeline():
    # Get the latest trained model
    latest_model_op = get_latest_model()
    
    # Compare with best model and update if better
    compare_and_save_best(
        new_model=latest_model_op.outputs["model"],
        metrics=latest_model_op.outputs["metrics"]
    )
