import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component, Model, Output, Dataset, Metrics, Input
from google.cloud import aiplatform

PROJECT_ID = "cat-dog-detection-449208"
REGION = "europe-west3"
BUCKET_NAME = "original-dataset-cat-dog"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

@component
def load_latest_dataset(dataset: Output[Dataset]):
    """Finds the latest dataset and outputs the path."""
    from google.cloud import storage
    import datetime

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="datasets/"))
    
    dataset_folders = sorted(set(blob.name.split('/')[1] for blob in blobs if "dataset" in blob.name), reverse=True)
    latest_dataset = dataset_folders[0] if dataset_folders else None

    dataset.uri = f"gs://{BUCKET_NAME}/datasets/{latest_dataset}"
    print(f"Using dataset: {dataset.uri}")

@component
def train_model(dataset: Input[Dataset], model: Output[Model], metrics: Output[Metrics]):  # ✅ FIXED: Input[Dataset]
    """Trains a model on the dataset and saves it."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import json
    from google.cloud import storage

    # Load dataset
    datagen = ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(dataset.uri + "/train", target_size=(128, 128), batch_size=8, class_mode='binary')
    valid_data = datagen.flow_from_directory(dataset.uri + "/valid", target_size=(128, 128), batch_size=8, class_mode='binary')

    # Define model
    model_tf = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_tf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model_tf.fit(train_data, validation_data=valid_data, epochs=5)

    # Evaluate and store metrics
    test_loss, test_acc = model_tf.evaluate(valid_data)
    metrics.log_metric("accuracy", test_acc)

    # Save model to GCS
    model_save_path = f"gs://{BUCKET_NAME}/models/cat_dog_model"
    model_tf.save(model_save_path)

    model.uri = model_save_path
    print(f"Model saved at: {model.uri}")

@component
def compare_and_save_best(new_model: Input[Model], metrics: Input[Metrics]):
    """Compares the new model with the best and updates if better."""
    import json
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    new_accuracy = metrics.metadata['accuracy']
    best_model_path = "models/best/model"
    best_metrics_path = "models/best/metrics.json"

    # Load best model accuracy if exists
    try:
        blob = bucket.blob(best_metrics_path)
        best_metrics_data = json.loads(blob.download_as_text())
        best_accuracy = best_metrics_data.get("accuracy", 0)
    except:
        best_accuracy = 0

    print(f"New Model Accuracy: {new_accuracy}, Best Model Accuracy: {best_accuracy}")

    if new_accuracy > best_accuracy:
        print("Updating best model...")
        new_model_blob = bucket.blob(new_model.uri)
        new_model_blob.copy_to(bucket, best_model_path)

        best_metrics_blob = bucket.blob(best_metrics_path)
        best_metrics_blob.upload_from_string(json.dumps({"accuracy": new_accuracy}))

@dsl.pipeline(name="cat-dog-training-pipeline", pipeline_root=f"gs://{BUCKET_NAME}/pipelines")
def training_pipeline():
    dataset = load_latest_dataset()

    model_and_metrics = train_model(dataset=dataset.output)

    compare_and_save_best(
        new_model=model_and_metrics.outputs["model"], 
        metrics=model_and_metrics.outputs["metrics"]
    )

