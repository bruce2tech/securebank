# SecureBank Fraud Detection System Dockerfile with Model Training
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Update system packages and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p logs storage/datasets output data_sources

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --verbose -r requirements.txt

# Copy application code and data
COPY . .

# Copy engineered dataset
COPY storage/datasets/dataset_engineered_raw.csv storage/datasets/dataset_engineered_raw.csv

# Train a model during build using external script
RUN python3 train_model_docker.py

# Verify model was created
RUN test -f output/best_lgb_model.pkl && \
    echo "✅ Model file verified: $(ls -lh output/best_lgb_model.pkl | awk '{print $5}')" || \
    (echo "⚠️ Model file not found" && exit 1)

# Create non-root user for security
RUN useradd -m -u 1000 securebank && chown -R securebank:securebank /app

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER securebank

# Expose port for Flask application
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]