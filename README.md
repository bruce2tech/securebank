# SecureBank Fraud Detection System

A comprehensive fraud detection system built with Flask, scikit-learn, and Docker for real-time transaction analysis and model training.

## 🚀 Quick Start

### Prerequisites
- Docker
- Python 3.9+ (for local development)
- bash (for running test scripts)
- data sources: customer_data.csv , transaction_data.parquet, fraud_truth_data.json

### Initial setup
From project root:
* 1. Clone repo and store data sources in securebank/data_sources

* 2. Create engineered dataset in securebank/storage/datasets
     Run: python engineer_from_raw.py

* 3. Create build docker image
     Run: docker build -t securebank  .

* 4. Start container
     Run: docker run -d -p 5000:5000 securebank_app

### Running the System

1. **Build and Start the Server**
   ```bash
   cd securebank/executables
   chmod +x *.sh
   ./run_server.sh
   ```

2. **Test the Prediction Endpoint**
   ```bash
   ./predict.sh
   ```

3. **Test Dataset Creation**
   ```bash
   ./create_dataset.sh
   ```

4. **Test Model Training**
   ```bash
   ./train_model.sh
   ```

### List datasets in the container
docker exec securebank-app ls -la storage/datasets/

### Copy datasets to local machine
docker cp securebank-app:/app/storage/datasets/filename ./storage/datasets/filename

### List models in the container
docker exec securebank-app ls -la storage/models/

### Copy models to local machine
docker cp securebank-app:/app/storage/models/filename ./storage/models/filename

## 📁 Project Structure

```
securebank/
├── Dockerfile                      # Docker configuration
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── app.py                         # Main Flask application
├── test.json                      # Sample test data
├── engineer_from_raw.py           # create engineered dataset from sources and test with xgboost
├── sampled_engineered_dataset.py  # create smaller dataset from engineered dataset
├── train_lightgbm.py              # dev script for training model
├── train_with_adasyn.py           # dev script for training model
├── train_model_docker.py          # dev script for training model    
├── modules/                       # Core modules
│   ├── data/                      # Data processing
│   │   └── raw_data_handler.py    # Data handling utilities
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       ├── advanced_logging.py
│       ├── data_drift_detector.py # detect drift on new data.
│       ├── logging_utils.py       # Logging system
│       └── model_utils.py         # Model management
├── logs/                          # System logs (JSON files)
├── storage/
│   └── datasets/                  # Generated training datasets
│   └── models/                    # Trained models
├── data_sources/                  # Raw data files
└── executables/                   # Test scripts
    ├── run_server.sh             # Build and run Docker container
    ├── predict.sh                # Test prediction endpoint
    ├── create_dataset.sh         # Test dataset creation
    └── train_model.sh            # Test model training
```

##  API Endpoints

### POST `/predict`
Predict fraud for a single transaction.

**Commandline Test**
```
curl -s -X POST http://localhost:5000/predict \
        -H 'Content-Type: application/json' \
        -d '{
    "trans_date_trans_time": "2021-10-07 12:01:55",
    "cc_num": "4059294504000000",
    "unix_time": 1633608115,
    "merchant": "Walmart",
    "category": "grocery_pos",
    "amt": 45.23,
    "merch_lat": 40.7589,
    "merch_long": -73.9851
}'

```

**Request Body:**
```json
{
    "trans_date_trans_time": "2021-10-07 12:01:55",
    "cc_num": "4059294504000000", 
    "unix_time": 1633608115,
    "merchant": "Walmart",
    "category": "grocery_pos",
    "amt": 45.23,
    "merch_lat": 40.7589,
    "merch_long": -73.9851
}
```

**Response:**
```json
{"confidence":0.6266993938524225,"features_used":58,"fraud_probability":0.6266993938524225,"model_type":"lightgbm","predict":"fraudulent"}
```

### POST `/create_dataset`
Generate high-quality training dataset from raw data sources.

**Commandline Test**
```
./executables/create_dataset.sh
```
**Response**
```
Testing /create_dataset endpoint...
URL: http://127.0.0.1:5000/create_dataset
----------------------------------------
Response Body:
{
    "columns": 23,
    "dataset_path": "storage/datasets/dataset_train_20250925_011305.csv",
    "fraud_ratio": 0.010027058490769887,
    "rows": 1647542,
    "status": "success",
    "timestamp": "20250925_011305"
}
```

### POST `/train_model`
Train a new fraud detection model on the latest dataset.

**Commandline Test**
```
./executables/train_model.sh
```

**Response:**
```json
Testing /train_model endpoint...
URL: http://127.0.0.1:5000/train_model
Timeout: 300s
Note: Training takes 1-2 minutes
----------------------------------------
Response Body:
{
    "adasyn_used": true,
    "configuration": "LightGBM-Aggressive",
    "features_used": 58,
    "meets_requirements": true,
    "message": "Model MEETS 70/70 requirements",
    "model_path": "output/fraud_model_20250925_011438.pkl",
    "model_type": "lightgbm",
    "performance": {
        "f1_score": 0.8019,
        "precision": 0.7634,
        "recall": 0.8445
    },
    "processing_time_seconds": 61.46,
    "status": "success",
    "test_samples": 329509,
    "threshold": 0.5,
    "training_samples": 1353333
}
```

### GET `/health`
Health check endpoint for monitoring.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2023-12-01T14:35:00",
    "model_loaded": true
}
```



## Testing

The system includes comprehensive bash scripts for testing all functionality:

```bash
# Start the system
./executables/run_server.sh

# Test prediction (requires trained model)
./executables/predict.sh [json_file] [host] [port]

# Generate training dataset
./executables/create_dataset.sh [host] [port] 

# Train new model
./executables/train_model.sh [host] [port]
```

This project is part of the SecureBank fraud detection system.