# Cat & Dog Classification System 🐱🐶

A comprehensive **machine learning system** for **classifying, managing, and improving a cat and dog image dataset**. This system provides an **end-to-end** pipeline from **data collection** to **model training, evaluation, and deployment** while supporting **local and cloud-based operations**.

![Interface](interface.png)
My cat is 67.73% a cat

## 📌 System Overview

This project implements a **complete MLOps pipeline**, including:
- **🌐 Web interface** for **image uploads, predictions, and validation**
- **📊 Training options**:
  - **Local training** (on your machine)
  - **Cloud training** using **Vertex AI** (not available yet because not debugged)
- **🖼️ Image preprocessing** and **augmentation**
- **📈 Model tracking and performance evaluation**
- **💾 Cloud storage and database integration**
- **🚀 Dynamic MLOps workflow for pipeline automation**

---

## 🛠️ Tech Stack

| Component  | Technology Used  |
|------------|----------------|
| **Backend**  | Flask (Python) |
| **Frontend** | HTML, JavaScript |
| **Cloud Services** | Google Cloud Platform (GCP) |
| **Database** | Cloud SQL (MySQL), SQLAlchemy |
| **ML Framework** | TensorFlow |
| **Image Processing** | OpenCV, PIL |
| **Pipeline Orchestration** | Kubeflow Pipelines (KFP), Vertex AI |
| **Other Libraries** | Pandas, FastAPI, Uvicorn, kfp |

---

## 🏛️ System Architecture

This project follows a **modular architecture** focusing on **scalability, performance, and automation**.

![System Architecture](architecture-diagram.png)

### 📌 Key Components

#### 🌐 Web Layer (Frontend & API)
- **Flask Web App**: Interface for users to upload images and manage predictions.
- **FastAPI Endpoints**: High-performance REST API for managing training and predictions.

#### 💻 Local Processing
- **Image Processing Module** (`image_processing.py`): Handles **image transformations, resizing, and normalization**.
- **Local Training Module** (`local_training.py`): Executes **model training** with a **hardcoded batch size, epochs, and image size**.

#### ☁️ Cloud Processing
- **Cloud Function (`preprocessing.py`)**: Handles dataset preparation and augmentation.
- **Vertex AI Pipeline** (`pipeline.py`)**: Automates training and deployment. (not available yet because not debugged)

#### 📊 Data Management
- **Cloud Storage (GCS)**: Stores images and trained models.
- **Cloud SQL (MySQL)**: Tracks predictions and validations.
- **SQLAlchemy ORM**: Manages database connections.

---

## 🔄 Operational Workflow

This system follows a **dynamic data pipeline**:

![Operational Flow](sequence-diagram.png)

### **1️⃣ Image Upload & Preprocessing**
- Users upload images through the web interface.
- Images are stored in **Google Cloud Storage**.
- Preprocessing pipeline **normalizes** and **augments** the dataset.

### **2️⃣ Training Options**
Training can be performed **locally** or in **the cloud**.

#### **🖥️ Local Training**
- Loads dataset from Cloud Storage.
- Runs training **with hardcoded parameters** (batch size, epochs, image size).
- Saves the trained model locally and in data bucket.

#### **☁️ Cloud Training (Vertex AI)**
- Trains the model using **scalable resources**.
- Handles large datasets efficiently.
- Automatically selects the **best model** based on evaluation metrics.

### **3️⃣ Prediction & Validation**
- Users can request **predictions**.
- Predictions are **stored in Cloud SQL**.
- Users can **validate** predictions.
- **Validated images** improve the dataset.
- User then push the validated images into the images directory where a new dataset can be generated.

### **4️⃣ Model & Dataset Management**
- The **best model is dynamically selected** based on validation accuracy.
- Data is **updated** automatically based on user feedback.

---

## 📌 Configuration Adjustments

**Currently, some parameters in the system are hardcoded. To improve flexibility, a better configuration file (`config.json`) should be introduced.**

### 🔧 **Parameters That Should Be Made Configurable**
| Parameter | Current State | Suggested Improvement |
|-----------|-------------|----------------------|
| **Image Size** | Hardcoded (`128x128`) | Define in `config.json` |
| **Batch Size** | Hardcoded (`8`) | Allow user-defined value |
| **Epochs** | Hardcoded (`5`) | Allow dynamic control |
| **Dataset Split** | Hardcoded (70-20-10) | Make adjustable |
| **Augmentations** | Fixed options disabled (rotate, grayscale) | Enable user selection |
| **Cloud vs. Local Training** | Manually triggered | Enable dynamic switch |


---

## 🚀 Getting Started

first of all you need to fill in or create a config.json file with the following structure:
```json
{
    "DATABASE_URL": "mysql+mysqlconnector://user:password@host:port/database_name",
    "GOOGLE_APPLICATION_CREDENTIALS": "json_files/service_user_api_key.json",
    "PROJECT_ID": "project_id",
    "BUCKET_NAME": "bucket_name",
    "PIPELINE_ROOT": "gs://bucket_name/pipelines/",
    "UPLOAD_FOLDER": "pipelines/uploads",
    "CLOUD_FUNCTION_URL": "https://europe-west3-project_id.cloudfunctions.net/preprocess_dataset",
    "TRAIN_LOCAL": true,
    "REGION": "europe-west3",
    "COMPILED_JSON_PATH_LOCAL": "pipelines/json_pipeline/model_management.json",
    "COMPILED_JSON_PATH_VERTEX": "pipelines/json_pipeline/pipeline.json"
  }

```
- create a project on GCP. when creating a project you will get project id.
- so you have to create an sql instance on cloud.
- create a service account and provide all necessary rights
- create a bucket in GCP storage
- upload the preprocessing.py file under gcp_function/preprocessing to a gcp cloud function. you can use one of the following codes for deploying. use the second one it is faster.

  ```sh
    gcloud functions deploy preprocess_dataset \
      --runtime python312 \
      --trigger-http \
      --memory 2048MB \
      --timeout 540s \
      --region europe-west3 \
    --service-account service_account_email \
      --allow-unauthenticated

    gcloud functions deploy preprocess_dataset \
      --runtime python312 \
      --trigger-http \
      --memory=8192MB \
      --timeout=900s \
      --region=europe-west3 \
      --service-account=service_account email \
      --allow-unauthenticated
  ```
- make sure the region is set correctly.


### 1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 2 Initialize Database
create a sql database on google cloud console
take the credentials and add them to the config.json
```bash
cd database
python test_db.py
```

### 3 Run the Application
```bash
python app.py
```

---

## 📌 Key Components

### 🔹 `app.py`
- **Flask backend** handling:
  - **Image uploads**
  - **Model predictions**
  - **Training triggers**
  - **Validation & database operations**

### 🔹 `local_training.py`
- **Manages local training**:
  - **Loads dataset**
  - **Runs model training**
  - **Tracks performance**
  - **Saves model versioning**
  
### 🔹 `image_processing.py`
- **Handles image operations**:
  - **Preprocessing**
  - **Predictions**
  - **Validation & storage management**

### 🔹 `preprocessing.py (Cloud Function)`
- **Manages dataset preparation**:
  - **Image resizing**
  - **Data augmentation**
  - **Dataset splitting**

### 🔹 `schema.py`
- **Defines database structure**:
  - **Prediction records**
  - **Validation status**
  - **Processing states**

---

one of the things I suggest to modify is the model architecture.  play around with it and see the model different result. but keep using keras and extract the model as .keras file :)

There is alot to improve upon in this code. the vertex ai training model was not tested because the free tier doesnt have access to these functions. I still need to work on it later.

