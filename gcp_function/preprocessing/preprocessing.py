import os
import functions_framework
from google.cloud import storage
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import traceback
import logging
import time
from load_config.config import BUCKET_NAME

def process_dataset(label_list, train_split, valid_split, test_split, 
                   image_size, rotate_90, rotate_180, rotate_270, greyscale):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_folder = f"datasets/{timestamp}_dataset"

    # Total images tracking
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    # Count total images first
    for label in label_list:
        blobs = list(bucket.list_blobs(prefix=f"{label}/"))
        total_images += len(blobs)
        if rotate_90: total_images += len(blobs)
        if rotate_180: total_images += len(blobs)
        if rotate_270: total_images += len(blobs)
    
    logger.warning(f"Total images to process: {total_images}")

    image_counts = {'cats': 0, 'dogs': 0}
    
    for label_idx, label in enumerate(label_list):
        blobs = list(bucket.list_blobs(prefix=f"{label}/"))
        if not blobs:
            raise ValueError(f"No images found for label: {label}")

        train_size = train_split / 100
        valid_size = valid_split / (100 - train_split)
        
        train_blobs, temp_blobs = train_test_split(blobs, train_size=train_size, random_state=42)
        valid_blobs, test_blobs = train_test_split(temp_blobs, train_size=valid_size, random_state=42)
        
        for split_name, split_blobs in [('train', train_blobs), 
                                      ('valid', valid_blobs), 
                                      ('test', test_blobs)]:
            for blob in split_blobs:
                try:
                    # Retry mechanism for blob download
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            image_data = blob.download_as_bytes()
                            break
                        except Exception as download_error:
                            if attempt == max_retries - 1:
                                raise
                            logger.warning(f"Download attempt {attempt + 1} failed for {blob.name}: {download_error}")
                            time.sleep(2 ** attempt)  # Exponential backoff
                    
                    if not image_data:
                        logger.warning(f"Empty data for image: {blob.name}")
                        failed_images += 1
                        continue
                        
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        logger.warning(f"Failed to decode image: {blob.name}")
                        failed_images += 1
                        continue
                    
                    # Process image
                    img = cv2.resize(img, (image_size, image_size))
                    if greyscale:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = np.expand_dims(img, axis=-1)
                    
                    # Save processed image
                    image_counts[label] += 1
                    filename = f"{label}_{image_counts[label]}.jpg"
                    save_path = f"{dataset_folder}/{split_name}/{label}/{filename}"
                    
                    # Retry mechanism for uploading
                    max_upload_retries = 3
                    for attempt in range(max_upload_retries):
                        try:
                            processed_blob = bucket.blob(save_path)
                            success, buffer = cv2.imencode('.jpg', img)
                            if success:
                                processed_blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
                            break
                        except Exception as upload_error:
                            if attempt == max_upload_retries - 1:
                                raise
                            logger.warning(f"Upload attempt {attempt + 1} failed for {save_path}: {upload_error}")
                            time.sleep(2 ** attempt)  # Exponential backoff
                    
                    # Save augmentations with retry mechanism
                    augmentations = [
                        (rotate_90, cv2.ROTATE_90_CLOCKWISE, '90'),
                        (rotate_180, cv2.ROTATE_180, '180'),
                        (rotate_270, cv2.ROTATE_90_COUNTERCLOCKWISE, '270')
                    ]
                    
                    for aug_flag, rotate_code, aug_name in augmentations:
                        if aug_flag:
                            rotated = cv2.rotate(img.copy(), rotate_code)
                            aug_path = f"{dataset_folder}/{split_name}/{label}/images/aug_{aug_name}_{filename}"
                            
                            for attempt in range(max_upload_retries):
                                try:
                                    aug_blob = bucket.blob(aug_path)
                                    success, buffer = cv2.imencode('.jpg', rotated)
                                    if success:
                                        aug_blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
                                    break
                                except Exception as aug_error:
                                    if attempt == max_upload_retries - 1:
                                        raise
                                    logger.warning(f"Augmentation upload attempt {attempt + 1} failed for {aug_path}: {aug_error}")
                                    time.sleep(2 ** attempt)
                    
                    processed_images += 1
                    if processed_images % 100 == 0:
                        logger.warning(f"Processed {processed_images}/{total_images} images")
                    
                except Exception as e:
                    logger.error(f"Error processing {blob.name}: {str(e)}")
                    failed_images += 1
                    continue
    
    logger.warning(f"Completed processing: {processed_images}/{total_images} images")
    logger.warning(f"Failed images: {failed_images}")

    # Save metadata
    metadata = {
        'label_list': label_list,
        'splits': {
            'train': train_split,
            'valid': valid_split,
            'test': test_split
        },
        'image_size': image_size,
        'augmentations': {
            'rotate_90': rotate_90,
            'rotate_180': rotate_180,
            'rotate_270': rotate_270,
            'greyscale': greyscale
        },
        'processed_images': processed_images,
        'total_images': total_images,
        'failed_images': failed_images
    }
    
    metadata_blob = bucket.blob(f"{dataset_folder}/metadata.json")
    metadata_blob.upload_from_string(json.dumps(metadata), content_type='application/json')
    
    return {
        'dataset_path': dataset_folder,
        'metadata': metadata
    }

@functions_framework.http
def preprocess_dataset(request):
    if request.method != 'POST':
        return ('Only POST requests allowed', 405)
    
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return json.dumps({
                "status": "error",
                "error": "Invalid JSON data",
                "error_type": "input_error"
            }), 400

        result = process_dataset(
            label_list=request_json['label_list'],
            train_split=request_json['train_split'],
            valid_split=request_json['valid_split'],
            test_split=request_json['test_split'],
            image_size=request_json['image_size'],
            rotate_90=request_json['rotate_90'],
            rotate_180=request_json['rotate_180'],
            rotate_270=request_json['rotate_270'],
            greyscale=request_json['greyscale']
        )
        
        return json.dumps({
            "status": "success",
            "metadata": result
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "error_type": "processing_error",
            "details": {
                "error_class": e.__class__.__name__,
                "traceback": traceback.format_exc()
            }
        }), 500