import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from google.cloud import storage
import os
from datetime import datetime
from load_config.config import BUCKET_NAME

class ImageProcessor:
    def __init__(self, model_path="local_models/best/model.keras", image_size=(128, 128)):
        try:
            # Check if model file exists before loading
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            else:
                print(f"No pre-trained model found at {model_path}. Model will need to be trained.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        self.image_size = image_size
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(BUCKET_NAME)

    def preprocess_for_prediction(self, image_bytes):
        """Preprocess image for model prediction"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize for model
        img = cv2.resize(img, self.image_size)
        img = img / 255.0  # Normalize
        
        return np.expand_dims(img, axis=0)

    def predict(self, image_bytes):
        """Make prediction on image"""
        if self.model == None:
            if os.path.exists("local_models/best/model.keras"):
                self.model = tf.keras.models.load_model("local_models/best/model.keras")
            else:
                print(f"No pre-trained model found at {"local_models/best/model.keras"}. Model will need to be trained.")
                self.model = None
                return {
                    'prediction': None,
                    'confidence': 0
                }

        img = self.preprocess_for_prediction(image_bytes)
        prediction = self.model.predict(img)[0][0]
        
        # Convert to label and confidence
        label = 'cat' if prediction < 0.5 else 'dog'
        confidence = 1 - prediction if prediction < 0.5 else prediction
        
        return {
            'prediction': label,
            'confidence': float(confidence)
        }

    def save_original_image(self, image_bytes, image_name):
        """Save original image locally"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = f"temp_images/{timestamp}_{image_name}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Save original image
        with open(local_path, 'wb') as f:
            f.write(image_bytes)
            
        return local_path

    def upload_to_bucket(self, image_path, validation_label):
        """Upload validated image to the appropriate bucket folder"""
        if validation_label not in ['cat', 'dog']:
            return None
            
        # Determine destination path
        filename = os.path.basename(image_path)
        blob_path = f"{validation_label}/{filename}"
        
        # Upload to GCS
        blob = self.bucket.blob(blob_path)
        blob.upload_from_filename(image_path)
        
        return f"gs://{self.bucket.name}/{blob_path}"

    def cleanup_local_image(self, image_path):
        """Remove temporary local image"""
        try:
            os.remove(image_path)
            return True
        except Exception as e:
            print(f"Error cleaning up {image_path}: {str(e)}")
            return False

class ImageUtils:
    @staticmethod
    def is_valid_image(image_bytes):
        """Check if image is valid and not corrupted"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            return True
        except:
            return False

    @staticmethod
    def get_image_format(image_bytes):
        """Get image format (extension)"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.format.lower()
        except:
            return None