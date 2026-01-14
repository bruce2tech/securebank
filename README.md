# SecureBank: Production Fraud Detection with Operational ML

> Real-time transaction fraud detection achieving **76.3% precision / 84.5% recall** while handling extreme class imbalance (256:1), demonstrating production ML system design with drift detection and automated retraining.

## The Problem

Financial fraud detection presents a challenging ML engineering problem:

1. **Extreme class imbalance**: Only 0.39% of transactions are fraudulent—standard classifiers predict "legitimate" for everything and achieve 99.6% accuracy while catching zero fraud.

2. **Asymmetric error costs**: A false negative (missed fraud) costs the bank money. A false positive (blocked legitimate transaction) costs customer trust. Neither is acceptable at high rates.

3. **Distribution shift**: Fraud patterns evolve. A model trained on 2023 data degrades on 2024 fraud tactics without monitoring and retraining infrastructure.

This project addresses all three through strategic algorithm selection, class balancing techniques, and production monitoring architecture.

## Approach and Key Decisions

### Why LightGBM + ADASYN?

I evaluated four approaches on the same dataset:

| Algorithm | Precision | Recall | Decision |
|-----------|----------:|-------:|----------|
| Logistic Regression | 45% | 72% | Underfit—insufficient capacity for 58 features |
| Random Forest | 69% | 78% | Below 70/70 threshold |
| XGBoost + SMOTE | 69% | **95%** | High recall but too many false alarms |
| **LightGBM + ADASYN** | **76.3%** | **84.5%** | Optimal balance—meets requirements |

**Strategic Decision**: ADASYN (Adaptive Synthetic Sampling) outperformed SMOTE because it generates more synthetic samples near decision boundaries where the classifier struggles. Combined with LightGBM's DART boosting (dropout regularization), this achieved the precision/recall balance required for production.

### Feature Engineering Strategy

The 58 engineered features fall into 6 categories, each targeting specific fraud patterns:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Velocity** | 12 | Rapid transactions signal card compromise |
| **Customer Behavior** | 15 | Deviation from spending patterns indicates fraud |
| **Merchant Risk** | 8 | Historical fraud rate by merchant/category |
| **Time Patterns** | 10 | Fraud peaks at specific hours/days |
| **Transaction Patterns** | 9 | Round amounts, unusual sizes |
| **Composite Risk** | 4 | Weighted combination scores |

**Top predictive features** (by importance):
1. `amt` (31.2%) — Transaction amount
2. `daily_amount_sum` (18.4%) — Cumulative daily spending
3. `amt_zscore` (12.3%) — Deviation from customer norm
4. `merchant_fraud_rate` (8.7%) — Historical merchant risk
5. `seconds_since_last_trans` (6.2%) — Transaction velocity

## Performance Results

**Test Set**: 329,509 transactions (0.39% fraud rate)

| Metric | Value | Requirement |
|--------|------:|-------------|
| Precision | 76.34% | ≥70% |
| Recall | 84.45% | ≥70% |
| F1-Score | 80.19% | — |
| Latency | <10ms | Production-ready |

**Confusion Matrix**:

|               | Predicted Legit | Predicted Fraud |
|---------------|---------------:|----------------:|
| Actual Legit | 327,206 | 573 |
| Actual Fraud | 198 | 1,532 |

**Interpretation**: Of 1,730 actual fraud cases, the model catches 1,532 (84.5%). Of 2,105 fraud predictions, 1,532 are correct (76.3%). The 573 false positives represent 0.17% of legitimate transactions—low enough for production deployment.

## Production Architecture

### System Design

![System Architecture](System_Architecture.png)

The production system includes:
- **REST API**: Flask-based prediction service with <10ms latency
- **Model Versioning**: Timestamped artifacts with performance metadata
- **Structured Logging**: JSON logs for every prediction (PII anonymized)
- **Drift Detection**: Statistical monitoring of feature distributions
- **Automated Retraining**: Performance-triggered model updates

### Drift Detection Pipeline

![Drift Detection](Drift_Detection_Pipeline.png)

**Statistical Tests**:
- **Numerical features**: Kolmogorov-Smirnov test (detects distribution shifts)
- **Categorical features**: Chi-square test (detects frequency changes)
- **Monitoring cadence**: Real-time comparison against training baseline

**Automated Retraining Triggers**:
1. Precision drops below 72% (2% buffer from requirement)
2. Recall drops below 72%
3. Feature drift detected in >25% of features
4. Fraud rate changes >50% from baseline

### Logging and Compliance

![Logging Architecture](Logging_Architecture.png)

**PII Handling**:
- Card numbers: Hashed (SHA-256)
- Coordinates: Rounded to 2 decimal places
- Retention: 90 days
- Compliance: GDPR and PCI-DSS aligned

## Quick Start

### Prerequisites
- Docker
- Raw data files: `customer_release.csv`, `transactions_release.parquet`, `fraud_release.json`

### Setup

```bash
# Clone and prepare data
git clone https://github.com/bruce2tech/securebank.git
cd securebank
# Place data files in securebank/data_sources/

# Generate engineered dataset
python engineer_from_raw.py

# Build and run Docker container
docker build -t securebank .
docker run -d -p 5000:5000 --name securebank-app securebank
```

### Test Endpoints

```bash
cd executables
chmod +x *.sh

./predict.sh          # Test single prediction
./create_dataset.sh   # Generate training data
./train_model.sh      # Train new model
```

## API Reference

### POST /predict
Score a transaction for fraud probability.

```bash
curl -X POST http://localhost:5000/predict \
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

**Response**:
```json
{
  "predict": "fraudulent",
  "fraud_probability": 0.627,
  "confidence": 0.627,
  "model_type": "lightgbm",
  "features_used": 58
}
```

### POST /train_model
Train a new model on the latest dataset.

```json
{
  "status": "success",
  "performance": {
    "precision": 0.7634,
    "recall": 0.8445,
    "f1_score": 0.8019
  },
  "model_path": "output/fraud_model_20250925_011438.pkl",
  "processing_time_seconds": 61.46
}
```

### GET /health
Health check for monitoring.

## Technical Insights

### Key Observations

1. **Class balancing strategy matters**: ADASYN's adaptive sampling outperformed SMOTE by 7% precision because it focuses synthetic generation on difficult boundary cases rather than uniform oversampling.

2. **Feature engineering dominates**: The top 5 engineered features account for 76% of model importance. Raw transaction fields alone achieve only 52% precision.

3. **Threshold tuning is critical**: The default 0.5 threshold optimizes F1. Adjusting to 0.6 increases precision to 82% at the cost of recall (76%)—a business decision based on false positive tolerance.

### Production Considerations

For enterprise deployment:

- **Real-time feature computation**: Velocity features require maintaining customer state. Consider Redis for low-latency lookups.
- **Model serving**: Current Flask implementation is single-threaded. Production requires Gunicorn workers or model serving infrastructure (TensorFlow Serving, Triton).
- **A/B testing**: New models should shadow-score before promotion to catch edge cases.
- **Explainability**: SHAP values for individual predictions support fraud analyst review.

### Known Limitations

- Dataset is synthetic (simulated from real patterns); production performance may differ
- Assumes batch feature computation; streaming features would require infrastructure changes
- No multi-model ensemble; production systems often combine rule-based and ML approaches

## Project Structure

```
securebank/
├── app.py                      # Flask API server
├── engineer_from_raw.py        # Feature engineering pipeline
├── train_lightgbm.py           # Model training with ADASYN
├── System_Report.md            # Comprehensive technical analysis
├── modules/
│   ├── data/                   # Data processing and drift detection
│   ├── features/               # Feature engineering
│   ├── model/                  # Model training and prediction
│   └── utils/                  # Logging, monitoring, utilities
├── storage/
│   ├── datasets/               # Engineered training data
│   ├── models/                 # Trained model artifacts
│   └── lineage/                # Dataset versioning
└── executables/                # Test scripts
```

## Technologies

- **ML Framework**: LightGBM, scikit-learn, imbalanced-learn (ADASYN)
- **API**: Flask
- **Data Processing**: pandas, NumPy, pyarrow
- **Containerization**: Docker
- **Statistical Testing**: scipy (KS-test, Chi-square)
- **Monitoring**: JSON structured logging

## Author

Patrick Bruce

## License

This project is for educational and portfolio purposes.
