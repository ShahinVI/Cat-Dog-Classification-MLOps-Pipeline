import json
import os

# Load configuration from JSON file
CONFIG_PATH = "config.json"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file `{CONFIG_PATH}` not found!")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Expose configuration as a dictionary
DATABASE_URL = config.get("DATABASE_URL")
GOOGLE_APPLICATION_CREDENTIALS = config.get("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = config.get("PROJECT_ID")
BUCKET_NAME = config.get("BUCKET_NAME")
PIPELINE_ROOT = config.get("PIPELINE_ROOT")
UPLOAD_FOLDER = config.get("UPLOAD_FOLDER")
CLOUD_FUNCTION_URL = config.get("CLOUD_FUNCTION_URL")
TRAIN_LOCAL = config.get("TRAIN_LOCAL")
REGION = config.get("REGION")
COMPILED_JSON_PATH_LOCAL = config.get("COMPILED_JSON_PATH_LOCAL")
COMPILED_JSON_PATH_VERTEX = config.get("COMPILED_JSON_PATH_VERTEX")
