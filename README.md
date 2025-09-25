# SecureBank Fraud Detection System

A comprehensive fraud detection system built with Flask, scikit-learn, and Docker for real-time transaction analysis and model training.

## 🚀 Quick Start

### Prerequisites
- Docker
- Python 3.9+ (for local development)
- bash (for running test scripts)

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

## 🔌 API Endpoints

### POST `/predict`
Predict fraud for a single transaction.

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
{
    "predict": "legitimate",
    "probability": 0.15
}
```

### POST `/create_dataset`
Generate high-quality training dataset from raw data sources.

**Response:**
```json
{
    "status": "success",
    "dataset_path": "storage/datasets/dataset_train_20231201_143022.csv",
    "records": 50000,
    "fraud_rate": 0.05
}
```

### POST `/train_model`
Train a new fraud detection model on the latest dataset.

**Response:**
```json
{
    "status": "success",
    "model_id": "fraud_model_20231201_143500_p0.752_r0.689",
    "performance": {
        "precision": 0.752,
        "recall": 0.689,
        "f1_score": 0.719
    }
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

## 📊 Model Requirements

- **Performance**: ≥70% precision and ≥70% recall on production datasets
- **Features**: Uses both transaction and customer data
- **Training Time**: Completes in under 15 minutes without GPU
- **Architecture**: Logistic regression with preprocessing pipeline

## 📝 Logging and Monitoring

All requests and operations are logged as JSON files in the `logs/` directory with timestamps:

- **Prediction requests**: Request data, predictions, response times
- **Training activities**: Model performance, training metrics
- **Dataset generation**: Dataset statistics, processing time
- **System errors**: Error details and context

Example log file: `logs/log_20231201_143022_123.json`

## 🧪 Testing

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

## 🐛 Development Status

**Phase 1: ✅ Complete** - Project structure and basic Flask app
- ✅ Organized project structure
- ✅ Basic Flask application with `/predict` endpoint
- ✅ Logging and model management utilities
- ✅ Docker configuration
- ✅ Test scripts

**Phase 2: 🚧 In Progress** - Enhanced Flask endpoints
**Phase 3: 📋 Planned** - Complete logging system
**Phase 4: 📋 Planned** - Dataset generation service  
**Phase 5: 📋 Planned** - Model training service
**Phase 6: 📋 Planned** - Final integration and testing

## 🔧 Local Development

For local development without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Test endpoints
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d @test.json
```

## 📄 License

This project is part of the SecureBank fraud detection system assignment.