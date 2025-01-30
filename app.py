from flask import Flask, request, render_template, jsonify
from google.cloud import storage, aiplatform
from werkzeug.utils import secure_filename
import cv2
import os
from PIL import Image
import io
import requests
import json
import kfp
from kfp.v2 import compiler
import importlib.util
from datetime import datetime
from utils.support_function import allowed_file, check_category
from database.schema import get_db, PredictionRecord
from utils.image_processing import ImageProcessor, ImageUtils
from sqlalchemy.orm import Session
import base64

from pipelines.model_management import model_management_pipeline
from pipelines.pipeline import training_pipeline
from utils.local_training import LocalTrainingPipeline
from concurrent.futures import ThreadPoolExecutor

from load_config.config import (
    GOOGLE_APPLICATION_CREDENTIALS, PROJECT_ID, BUCKET_NAME, PIPELINE_ROOT, 
    UPLOAD_FOLDER, CLOUD_FUNCTION_URL, TRAIN_LOCAL, REGION, 
    COMPILED_JSON_PATH_LOCAL, COMPILED_JSON_PATH_VERTEX
)

executor = ThreadPoolExecutor(1)

app = Flask(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipelines/"


if TRAIN_LOCAL:
    COMPILED_JSON_PATH = COMPILED_JSON_PATH_LOCAL
else:
    COMPILED_JSON_PATH = COMPILED_JSON_PATH_VERTEX

PROJECT_ID=PROJECT_ID

# Region: europe-west3 
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)



@app.route('/upload_file', methods=['POST'])
def upload_file():
   if 'file' not in request.files:
       return {"status": "error", "filename": "", "error": "No file in request"}

   file = request.files['file']
   category = request.form['category']

   # Validate category
   if not check_category(category):
       return {"status": "error", "filename": file.filename, 
               "error": "Invalid category. Choose 'cats' or 'dogs'."}

   # Validate file
   if file.filename == '':
       return {"status": "error", "filename": "", 
               "error": "No file selected."}
   if not allowed_file(file.filename):
       return {"status": "error", "filename": file.filename,
               "error": "Unsupported file type. Allowed: png, jpg, jpeg."}

   try:
       # Verify Image Integrity
       image = Image.open(io.BytesIO(file.read()))
       image.verify()
       file.seek(0)
       
       filename = secure_filename(file.filename)
       blob_path = f"{category}/{filename}"
       
       blob = bucket.blob(blob_path)
       blob.upload_from_string(
           file.read(),
           content_type=file.content_type
       )
       return {"status": "success", "filename": filename}
   
   except (IOError, SyntaxError) as e:
       return {"status": "error", "filename": file.filename, 
               "error": f"Invalid or corrupted image: {str(e)}"}
   except Exception as e:
       return {"status": "error", "filename": file.filename,
               "error": f"Upload failed: {str(e)}"}

@app.route('/', methods=['GET', 'POST'])
def upload():
   return render_template('interface.html')

@app.route('/process_dataset', methods=['POST'])
def process_dataset():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['label_list', 'train_split', 'valid_split', 'test_split', 
                         'image_size', 'rotate_90', 'rotate_180', 'rotate_270', 'greyscale']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "status": "error",
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Validate label list
        if not isinstance(data['label_list'], list) or not all(isinstance(label, str) for label in data['label_list']):
            return jsonify({
                "status": "error",
                "error": "label_list must be a list of strings"
            }), 400

        # Validate splits
        splits = [data['train_split'], data['valid_split'], data['test_split']]
        if not all(isinstance(split, (int, float)) for split in splits):
            return jsonify({
                "status": "error",
                "error": "Split values must be numbers"
            }), 400
        
        if sum(splits) != 100:
            return jsonify({
                "status": "error",
                "error": "Split percentages must sum to 100%"
            }), 400
        
        if any(split <= 0 for split in splits):
            return jsonify({
                "status": "error",
                "error": "Split percentages must be positive"
            }), 400

        # Validate image size
        if not isinstance(data['image_size'], int) or data['image_size'] < 32 or data['image_size'] > 512:
            return jsonify({
                "status": "error",
                "error": "image_size must be an integer between 32 and 512"
            }), 400

        # Validate boolean flags
        bool_fields = ['rotate_90', 'rotate_180', 'rotate_270', 'greyscale']
        if not all(isinstance(data[field], bool) for field in bool_fields):
            return jsonify({
                "status": "error",
                "error": "Rotation and greyscale flags must be boolean"
            }), 400

        response = requests.post(CLOUD_FUNCTION_URL, json=data)
        
        if response.status_code != 200:
            error_data = response.json()
            return jsonify(error_data), response.status_code
            
        return response.json()
        
    except requests.RequestException as e:
        return jsonify({
            "status": "error",
            "error": "Failed to connect to processing service",
            "details": str(e)
        }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/preprocess')
def preprocess():
    return render_template('preprocessing.html')

# vertex AI pipeline
@app.route('/upload-pipeline', methods=['POST'])
def upload_pipeline():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith('.py'):
            return jsonify({"error": "Only Python (.py) files are allowed"}), 400

        # Save the uploaded file locally
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(file_path)
        

        # Load the pipeline function dynamically
        spec = importlib.util.spec_from_file_location("pipeline_module", file_path)
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)

        if TRAIN_LOCAL:
            # if you want to train locally but upload and see metrics using vertex ai
            if not hasattr(pipeline_module, "model_management_pipeline"):
                return jsonify({"error": "Pipeline script does not define 'model_management' function"}), 400
        
            compiler.Compiler().compile(
                pipeline_func=pipeline_module.model_management_pipeline,
                package_path=COMPILED_JSON_PATH
                )
            
            blob = bucket.blob("pipelines/model_management.json")
            blob.upload_from_filename(COMPILED_JSON_PATH)

            return jsonify({
                "status": "success",
                "message": "Pipeline compiled and uploaded successfully!",
                "file_uploaded": file.filename,
                "storage_path": f"{PIPELINE_ROOT}model_management.json"
            })
        else:
            # for vertex ai
            if not hasattr(pipeline_module, "training_pipeline"):
                return jsonify({"error": "Pipeline script does not define 'training_pipeline' function"}), 400

            # Step 1: Compile pipeline.py to pipeline.json dynamically
            compiler.Compiler().compile(
                pipeline_func=pipeline_module.training_pipeline,
                package_path=COMPILED_JSON_PATH
            )

            # Step 2: Upload pipeline.json to Google Cloud Storage
            blob = bucket.blob("pipelines/pipeline.json")
            blob.upload_from_filename(COMPILED_JSON_PATH)

            #  Return a success message including a timestamp
            return jsonify({
                "status": "success",
                "message": "Pipeline compiled and uploaded successfully!",
                "file_uploaded": file.filename,
                "storage_path": f"{PIPELINE_ROOT}pipeline.json"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/start-training', methods=['POST'])
def start_training():
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION)

        if TRAIN_LOCAL:
            pipeline_job = aiplatform.PipelineJob(
            display_name="cat-dog-training-pipeline",
            template_path=f"{PIPELINE_ROOT}model_management.json",
            pipeline_root=PIPELINE_ROOT
        )
        else:
            pipeline_job = aiplatform.PipelineJob(
                display_name="cat-dog-training-pipeline",
                template_path=f"{PIPELINE_ROOT}pipeline.json",
                pipeline_root=PIPELINE_ROOT
            )

        pipeline_job.submit()
        print("✅ Vertex AI Pipeline triggered!")

        return jsonify({"message": "Pipeline started in Vertex AI!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/evaluate-model', methods=['POST'])
def evaluate_model():
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(COMPILED_JSON_PATH), exist_ok=True)
        if TRAIN_LOCAL:
            pipeline_func=model_management_pipeline
            file_json="model_management"
        else:
            pipeline_func=training_pipeline
            file_json="pipeline"


        # Compile the pipeline
        compiler.Compiler().compile(
            pipeline_func=pipeline_func,
            package_path=COMPILED_JSON_PATH
        )
        print("✅ Pipeline compiled successfully")
        
        # Upload the compiled pipeline
        blob = bucket.blob(f"pipelines/{file_json}.json")
        blob.upload_from_filename(COMPILED_JSON_PATH)
        print("✅ Pipeline uploaded to GCS")
        
        # Run the pipeline
        aiplatform.init(project=PROJECT_ID, location=REGION)
        pipeline_job = aiplatform.PipelineJob(
            display_name=file_json,
            template_path=f"gs://{BUCKET_NAME}/pipelines/{file_json}.json",
            pipeline_root=PIPELINE_ROOT
        )
        pipeline_job.submit()
        print("✅ Pipeline job submitted")
        
        return jsonify({
            "status": "success",
            "message": "Model evaluation pipeline started",
            "job_id": pipeline_job.resource_name
        })
        
    except Exception as e:
        print(f"❌ Error in evaluate_model: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/pipeline')
def pipeline():
    return render_template('pipeline.html')  # Serving the correct HTML template

#########################
#### TEMP ######

@app.route('/start-local-training', methods=['POST'])
def start_local_training():
    try:
        # Clear any existing status file
        if os.path.exists('json_files/training_status.json'):
            os.remove('json_files/training_status.json')
            
        # Start training in a background thread
        executor.submit(run_training_pipeline)
        
        return jsonify({
            "status": "success",
            "message": "Training started successfully"
        })
    except Exception as e:
        print(f"Error starting training: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def run_training_pipeline():
    try:
        print("Initializing training pipeline...")
        pipeline = LocalTrainingPipeline()
        
        print("Starting pipeline execution...")
        save_paths = pipeline.run_pipeline()
        
        print("Training completed successfully")
        print(f"Local model path: {save_paths['local_path']}")
        print(f"Cloud model path: {save_paths['cloud_path']}")
        
        # Get metrics from the saved file
        metrics_path = os.path.join(save_paths['local_path'], 'metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # Add test metrics to status update
        status_data = {
            "status": "completed",
            "local_path": save_paths['local_path'],
            "cloud_path": save_paths['cloud_path'],
            "metrics": {
                "accuracy": metrics['accuracy'],
                "val_accuracy": metrics['val_accuracy'],
                "loss": metrics['loss'],
                "val_loss": metrics['val_loss'],
                "test_accuracy": metrics.get('test_accuracy', None),
                "test_loss": metrics.get('test_loss', None)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Update status file
        with open('json_files/training_status.json', 'w') as f:
            json.dump(status_data, f)
        
    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in training pipeline: {error_message}")
        print("Traceback:")
        print(traceback_str)
        
        # Update status file with error
        with open('json_files/training_status.json', 'w') as f:
            json.dump({
                "status": "failed",
                "error": error_message,
                "traceback": traceback_str,
                "timestamp": datetime.now().isoformat()
            }, f)

@app.route('/training-status', methods=['GET'])
def get_training_status():
    try:
        if os.path.exists('json_files/training_status.json'):
            with open('json_files/training_status.json', 'r') as f:
                status_data = json.load(f)
                print(f"Current status: {status_data}")
                return jsonify(status_data)
        else:
            return jsonify({
                "status": "not_started"
            })
    except Exception as e:
        print(f"Error reading status: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    
@app.route('/local_training')
def local_training():
    return render_template('local_training.html')
##################################################

image_processor = ImageProcessor()

@app.route('/predict_manage')
def predict_manage():
    return render_template('predict_manage.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "error": "No image provided"})

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"status": "error", "error": "No image selected"})

        # Read image bytes
        image_bytes = image_file.read()
        
        # Validate image
        if not ImageUtils.is_valid_image(image_bytes):
            return jsonify({"status": "error", "error": "Invalid or corrupted image"})

        # Save original image
        local_path = image_processor.save_original_image(image_bytes, image_file.filename)
        
        # Make prediction
        result = image_processor.predict(image_bytes)
        
        # Save to database
        db = next(get_db())
        record = PredictionRecord(
            image_path=local_path,
            image_name=image_file.filename,
            prediction=result['prediction'],
            confidence=result['confidence']
        )
        db.add(record)
        db.commit()
        
        return jsonify({
            "status": "success",
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "image_id": record.id
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/get_unvalidated_image')
def get_unvalidated_image():
    try:
        db = next(get_db())
        record = db.query(PredictionRecord).filter_by(
            is_validated=False,
            is_processed=False
        ).first()
        
        if not record:
            return jsonify({
                "status": "error",
                "error": "No unvalidated images found"
            })
        
        # Create temporary URL or data URL for image
        with open(record.image_path, 'rb') as f:
            image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode()
        
        return jsonify({
            "status": "success",
            "image_id": record.id,
            "image_url": f"data:image/jpeg;base64,{image_b64}",
            "prediction": record.prediction,
            "confidence": record.confidence
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/validate_image', methods=['POST'])
def validate_image():
    try:
        data = request.json
        image_id = data.get('image_id')
        label = data.get('label')
        
        if not image_id or not label:
            return jsonify({"status": "error", "error": "Missing image_id or label"})
        
        if label not in ['cat', 'dog', 'neither']:
            return jsonify({"status": "error", "error": "Invalid label"})
        
        db = next(get_db())
        record = db.query(PredictionRecord).filter_by(id=image_id).first()
        
        if not record:
            return jsonify({"status": "error", "error": "Image record not found"})
        
        record.is_validated = True
        record.validation_label = label
        db.commit()
        
        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/process_validated_images', methods=['POST'])
def process_validated_images():
    try:
        db = next(get_db())
        records = db.query(PredictionRecord).filter_by(
            is_validated=True,
            is_processed=False
        ).all()
        
        processed_images = []
        processed_count = 0
        
        for record in records:
            if record.validation_label in ['cats', 'dogs']:
                # Upload to bucket
                cloud_path = image_processor.upload_to_bucket(
                    record.image_path,
                    record.validation_label
                )
                
                if cloud_path:
                    record.is_processed = True
                    record.processed_timestamp = datetime.utcnow()
                    processed_count += 1
                    
                    processed_images.append({
                        "url": f"data:image/jpeg;base64,{get_image_base64(record.image_path)}",
                        "prediction": record.prediction,
                        "validation": record.validation_label,
                        "status": "Uploaded"
                    })
                    
                    # Cleanup local file
                    image_processor.cleanup_local_image(record.image_path)
        
        db.commit()
        
        return jsonify({
            "status": "success",
            "processed_count": processed_count,
            "processed_images": processed_images
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/get_stats')
def get_stats():
    try:
        db = next(get_db())
        
        unprocessed_count = db.query(PredictionRecord).filter_by(is_processed=False).count()
        validated_count = db.query(PredictionRecord).filter_by(
            is_validated=True,
            is_processed=False
        ).count()
        
        return jsonify({
            "status": "success",
            "unprocessed_count": unprocessed_count,
            "validated_count": validated_count
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

def get_image_base64(image_path):
    """Helper function to convert image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()





######################################################

if __name__ == '__main__':
    app.run(debug=True)

#gcloud functions deploy preprocess_dataset \
#  --runtime python312 \
#  --trigger-http \
#  --memory 2048MB \
#  --timeout 540s \
#  --region europe-west3 \
# --service-account service_account_email \
#  --allow-unauthenticated


#gcloud functions deploy preprocess_dataset \
#  --runtime python312 \
#  --trigger-http \
#  --memory=8192MB \
#  --timeout=900s \
#  --region=europe-west3 \
#  --service-account=service_account email \
#  --allow-unauthenticated