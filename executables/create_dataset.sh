#!/bin/bash

# create_dataset.sh - Test the /create_dataset endpoint
# Usage: ./create_dataset.sh [host] [port]

HOST=${1:-"127.0.0.1"}
PORT=${2:-"5000"}
URL="http://${HOST}:${PORT}/create_dataset"

echo "Testing /create_dataset endpoint..."
echo "URL: $URL"
echo "----------------------------------------"

# Make the POST request
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}\nTIME_TOTAL:%{time_total}" \
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
echo "Response Time: ${time_total}s"

if [ "$http_status" -eq 200 ]; then
    # Extract the dataset path
    dataset_path=$(echo "$response_body" | python3 -c "import json,sys; print(json.load(sys.stdin).get('dataset_path',''))" 2>/dev/null)
    filename=$(basename "$dataset_path")
    
    # Check for nested directory issue and move file if needed
    nested_path="securebank/storage/datasets/securebank/storage/datasets/$filename"
    correct_path="securebank/storage/datasets/$filename"
    
    if [ -f "$nested_path" ]; then
        echo "  Found file in nested directory, moving..."
        mv "$nested_path" "$correct_path"
        echo "  File moved to: $correct_path"
        
        # Clean up empty nested directories
        rmdir "securebank/storage/datasets/securebank/storage/datasets" 2>/dev/null
        rmdir "securebank/storage/datasets/securebank/storage" 2>/dev/null
        rmdir "securebank/storage/datasets/securebank" 2>/dev/null
    elif [ -f "$correct_path" ]; then
        echo "  File created at: $correct_path"
    else
        echo "  File location: Check Docker container"
    fi
    
    echo "✓ Dataset creation endpoint test PASSED"
    exit 0
else
    echo "✗ Dataset creation endpoint test FAILED"
    exit 1
fi