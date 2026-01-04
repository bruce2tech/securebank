#!/bin/bash

# predict.sh - Test the /predict endpoint with sample transaction data
# Usage: ./predict.sh [json_file] [host] [port]

# Default values
JSON_FILE=${1:-"../test.json"}
HOST=${2:-"127.0.0.1"}
PORT=${3:-"5000"}
URL="http://${HOST}:${PORT}/predict"

echo "Testing /predict endpoint..."
echo "URL: $URL"
echo "JSON file: $JSON_FILE"
echo "----------------------------------------"

# Check if JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found at $JSON_FILE"
    echo "Creating sample test.json file..."
    
    # Create sample JSON if it doesn't exist
    cat > "$JSON_FILE" << 'EOF'
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
EOF
    echo "Sample test.json created."
fi

echo "Request payload:"
cat "$JSON_FILE" | python3 -m json.tool 2>/dev/null || cat "$JSON_FILE"
echo ""

# Make the POST request
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}\nTIME_TOTAL:%{time_total}" \
     -X POST \
     -H "Content-Type: application/json" \
     -d @"$JSON_FILE" \
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

# Check if request was successful
if [ "$http_status" -eq 200 ]; then
    echo "✓ Prediction endpoint test PASSED"
    exit 0
else
    echo "✗ Prediction endpoint test FAILED"
    exit 1
fi