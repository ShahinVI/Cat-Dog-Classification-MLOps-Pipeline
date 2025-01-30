import tensorflow as tf
from google.cloud import storage
import os
import json
from datetime import datetime
import shutil
from load_config.config import PROJECT_ID, BUCKET_NAME
class LocalTrainingPipeline:
    def __init__(self, project_id=PROJECT_ID, bucket_name=BUCKET_NAME):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        # Add status file for tracking progress
        self.status_file = 'json_files/training_status.json'
        self.update_status("initializing")

    def update_status(self, status, message=None, metrics=None):
        """Update the status file with current progress"""
        status_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        if metrics:
            status_data["metrics"] = metrics
        
        print(f"Status Update: {status} - {message if message else ''}")
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f)

    def download_dataset(self, local_path="dataset"):
        """Download dataset from GCS to local storage"""
        
        try:
            print("Downloading dataset...")
            self.update_status("downloading_dataset")
            
            if not os.path.exists(local_path):
                os.makedirs(local_path)

            blobs = list(self.bucket.list_blobs(prefix="datasets/"))
            print(f"Found {len(blobs)} total blobs")
            
            dataset_folders = sorted(set(blob.name.split('/')[1] 
                                      for blob in blobs if "dataset" in blob.name), 
                                  reverse=True)
            print(f"Found dataset folders: {dataset_folders}")
            
            if not dataset_folders:
                raise Exception("No dataset found in GCS bucket")
                
            latest_dataset = dataset_folders[0]
            dataset_prefix = f"datasets/{latest_dataset}"
            print(f"Using dataset: {dataset_prefix}")
            
            # Download dataset files
            downloaded_files = 0
            for blob in self.bucket.list_blobs(prefix=dataset_prefix):
                local_file = os.path.join(local_path, blob.name.replace(dataset_prefix + '/', ''))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                blob.download_to_filename(local_file)
                downloaded_files += 1
                if downloaded_files % 10 == 0:
                    self.update_status("downloading_dataset", f"Downloaded {downloaded_files} files")
            
            print(f"Successfully downloaded {downloaded_files} files")
            return local_path
            
        except Exception as e:
            error_msg = f"Error downloading dataset: {str(e)}"
            print(error_msg)
            self.update_status("error", error_msg)
            raise

    def train_model(self, dataset_path, batch_size=8):
        """Train the model using pre-split dataset"""
        try:
            print("Preparing data generators...")
            self.update_status("preparing_data")
            
            # Check if directories exist
            train_dir = os.path.join(dataset_path, 'train')
            valid_dir = os.path.join(dataset_path, 'valid')
            test_dir = os.path.join(dataset_path, 'test')
            
            for dir_path in [train_dir, valid_dir, test_dir]:
                if not os.path.exists(dir_path):
                    raise Exception(f"Directory not found: {dir_path}")
            
            # List contents of directories
            for dir_path in [train_dir, valid_dir, test_dir]:
                print(f"\nContents of {dir_path}:")
                for root, dirs, files in os.walk(dir_path):
                    print(f"Directory: {root}")
                    print(f"Subdirectories: {dirs}")
                    print(f"Files: {len(files)} files")

            # Create data generator with augmentation
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Create separate generator for validation/test without augmentation
            valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255  # Only rescaling for validation/test
            )

            self.update_status("loading_data")
            # Load training data
            train_generator = datagen.flow_from_directory(
                train_dir,
                target_size=(128, 128),
                batch_size=batch_size,
                class_mode='binary'
            )

            # Load validation data
            validation_generator = valid_test_datagen.flow_from_directory(
                valid_dir,
                target_size=(128, 128),
                batch_size=batch_size,
                class_mode='binary'
            )

            print("Creating model...")
            self.update_status("creating_model")
            model = self.create_model()

            print("Training model...")
            self.update_status("training")
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=20,  # Increase epochs but let early stopping decide when to stop
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: self.update_status(
                            "training",
                            f"Completed epoch {epoch + 1}",
                            logs
                        )
                    )
                ]
            )

            # Load and evaluate on test set
            print("\nEvaluating on test set...")
            test_generator = valid_test_datagen.flow_from_directory(
                test_dir,
                target_size=(128, 128),
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False
            )
            
            test_results = model.evaluate(test_generator, verbose=1)
            
            # Add test metrics to history dictionary
            history.history['test_accuracy'] = [test_results[1]]  # Accuracy is typically the second metric
            history.history['test_loss'] = [test_results[0]]     # Loss is typically the first metric

            return model, history
                
        except Exception as e:
            error_msg = f"Error in model training: {str(e)}"
            print(error_msg)
            self.update_status("error", error_msg)
            raise
    
    def evaluate_model(self, model, test_generator):
        """Evaluate model with detailed metrics"""
        # Get predictions
        predictions = model.predict(test_generator)
        predictions_binary = (predictions > 0.5).astype(int)
        
        # Get true labels
        true_labels = test_generator.classes
        
        # Calculate confusion matrix
        confusion = tf.math.confusion_matrix(true_labels, predictions_binary)
        
        print("\nConfusion Matrix:")
        print("           Predicted Cat  Predicted Dog")
        print(f"Actual Cat     {confusion[0][0]}           {confusion[0][1]}")
        print(f"Actual Dog     {confusion[1][0]}           {confusion[1][1]}")
        
        # Calculate per-class metrics
        cat_precision = confusion[0][0] / (confusion[0][0] + confusion[1][0])
        cat_recall = confusion[0][0] / (confusion[0][0] + confusion[0][1])
        dog_precision = confusion[1][1] / (confusion[1][1] + confusion[0][1])
        dog_recall = confusion[1][1] / (confusion[1][1] + confusion[1][0])
        
        print("\nPer-class Metrics:")
        print(f"Cat - Precision: {cat_precision:.4f}, Recall: {cat_recall:.4f}")
        print(f"Dog - Precision: {dog_precision:.4f}, Recall: {dog_recall:.4f}")
        
        return confusion, predictions

    def run_pipeline(self):
        """Run the complete training pipeline"""
        try:
            print("Starting training pipeline...")
            self.update_status("started", "Pipeline initializing...")
            
            # Download dataset
            dataset_path = self.download_dataset()
            
            # Train model
            model, history = self.train_model(dataset_path)
            
            # Calculate metrics including test results
            metrics = {
                # Training and validation metrics
                "accuracy": float(history.history['accuracy'][-1]),
                "val_accuracy": float(history.history['val_accuracy'][-1]),
                "loss": float(history.history['loss'][-1]),
                "val_loss": float(history.history['val_loss'][-1]),
                # Test metrics
                "test_accuracy": float(history.history['test_accuracy'][0]),
                "test_loss": float(history.history['test_loss'][0])
            }
            
            # Save model and metrics
            self.update_status("saving_model", "Saving trained model...")
            model_path = self.save_model(model, metrics)
            
            print(f"Training completed successfully!")
            print(f"Model saved at: {model_path}")
            print(f"Final metrics:")
            print(f"  Training:    Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
            print(f"  Validation:  Accuracy: {metrics['val_accuracy']:.4f}, Loss: {metrics['val_loss']:.4f}")
            print(f"  Test:        Accuracy: {metrics['test_accuracy']:.4f}, Loss: {metrics['test_loss']:.4f}")
            
            self.update_status("completed", "Training completed successfully", metrics)
            
            return model_path, metrics
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            print(error_msg)
            self.update_status("failed", error_msg)
            
            
    def create_model(self, input_shape=(128, 128, 3)):
        """Create a more robust model"""
        model = tf.keras.Sequential([
            # First Convolution Block
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            
            # Second Convolution Block
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            
            # Third Convolution Block
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            
            # Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),  # Add dropout to prevent overfitting
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]  # Add more metrics
        )
        
        return model
    
    def save_model(self, model, metrics):
        """Save model and metrics both locally and to GCS"""
        print("Saving model and metrics...")
        self.update_status("saving_model", "Saving trained model...")
        
        # Generate timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Create local models directory if it doesn't exist
            local_models_dir = os.path.join('local_models', timestamp)
            os.makedirs(local_models_dir, exist_ok=True)
            
            # Save model locally
            local_model_path = os.path.join(local_models_dir, 'model.keras')
            local_metrics_path = os.path.join(local_models_dir, 'metrics.json')
            
            # Save model file
            model.save(local_model_path)
            print(f"Model saved locally at {local_model_path}")
            
            # Save metrics locally
            with open(local_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved locally at {local_metrics_path}")
            
            # Save to GCS (as backup)
            model_blob_path = f"models/{timestamp}/model.keras"
            model_blob = self.bucket.blob(model_blob_path)
            model_blob.upload_from_filename(local_model_path)
            print(f"Uploaded model to {model_blob_path}")
            
            metrics_blob = self.bucket.blob(f"models/{timestamp}/metrics.json")
            metrics_blob.upload_from_string(json.dumps(metrics))
            print(f"Uploaded metrics to models/{timestamp}/metrics.json")
            
            # Check if this is the best model
            self.update_best_model(local_model_path, metrics)
            
            return {
                "local_path": local_models_dir,
                "cloud_path": f"gs://{self.bucket_name}/models/{timestamp}",
                "timestamp": timestamp
            }
                
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            print(error_msg)
            self.update_status("error", error_msg)
            raise

    def update_best_model(self, model_path, new_metrics):
        """Update best model if current model is better"""
        try:
            best_model_dir = 'local_models/best'
            best_metrics_path = os.path.join(best_model_dir, 'metrics.json')
            
            # Get new model accuracy
            new_accuracy = new_metrics.get('val_accuracy', new_metrics.get('accuracy', 0))
            
            # Check if there's an existing best model
            if os.path.exists(best_metrics_path):
                with open(best_metrics_path, 'r') as f:
                    best_metrics = json.load(f)
                    best_accuracy = best_metrics.get('val_accuracy', best_metrics.get('accuracy', 0))
            else:
                best_accuracy = 0
                os.makedirs(best_model_dir, exist_ok=True)
            
            print(f"New model accuracy: {new_accuracy:.4f}")
            print(f"Best model accuracy: {best_accuracy:.4f}")
            
            # Update if better
            if new_accuracy > best_accuracy:
                print(f"New best model! Improvement: {(new_accuracy - best_accuracy)*100:.2f}%")
                
                # Copy model file
                best_model_path = os.path.join(best_model_dir, 'model.keras')
                shutil.copy2(model_path, best_model_path)
                
                # Save metrics
                with open(best_metrics_path, 'w') as f:
                    json.dump(new_metrics, f, indent=2)
                
                print("Best model updated successfully")
                
                # Also update in cloud
                model_blob = self.bucket.blob('models/best/model.keras')
                model_blob.upload_from_filename(best_model_path)
                
                metrics_blob = self.bucket.blob('models/best/metrics.json')
                metrics_blob.upload_from_string(json.dumps(new_metrics))
            else:
                print("Current best model remains superior")
        except Exception as e:
            error_msg = f"Error updating model: {str(e)}"
            print(error_msg)
            self.update_status("error", error_msg)
            raise
                
if __name__ == "__main__":
    pipeline = LocalTrainingPipeline()
    pipeline.run_pipeline()