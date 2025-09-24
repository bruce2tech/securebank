#!/bin/bash

# train_model.sh - Test the /train_model endpoint
# Usage: ./train_model.sh [host] [port]

# Default values
HOST=${1:-"127.0.0.1"}
PORT=${2:-"5000"}
URL="http://${HOST}:${PORT}/train_model"
TIMEOUT=300  # This line was missing

echo "Testing /train_model endpoint..."
echo "URL: $URL"
echo "Timeout: ${TIMEOUT}s"
echo "Note: Training takes 1-2 minutes"
echo "----------------------------------------"

# Make the POST request with extended timeout
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}\nTIME_TOTAL:%{time_total}" \
     --max-time ${TIMEOUT} \
     -X POST \
     -H "Content-Type: application/json" \
     -d '{}' \
     "$URL")

# Extract HTTP status and time
http_status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
time_total=$(echo "$response" | grep "TIME_TOTAL:" | cut -d: -f2)
response_body=$(echo "$response" | sed '/HTTP_STATUS:/d; /TIME_TOTAL:/d')

# Display results
echo "Response Body:"
echo "$response_body" | python3 -m json.tool 2>/dev/null || echo "$response_body"
echo ""
echo "HTTP Status: $http_status"
echo "Training Time: ${time_total}s"

# Check if request was successful
if [ "$http_status" -eq 200 ]; then
    echo "✓ Model training endpoint test PASSED"
    exit 0
else
    echo "✗ Model training endpoint test FAILED"
    exit 1
fi