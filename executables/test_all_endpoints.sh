#!/bin/bash

# test_all_endpoints.sh - Comprehensive test of all SecureBank endpoints
# Usage: ./test_all_endpoints.sh [host] [port]

# Default values
HOST=${1:-"127.0.0.1"}
PORT=${2:-"5000"}
BASE_URL="http://${HOST}:${PORT}"

echo "🧪 SecureBank Endpoint Testing Suite"
echo "======================================"
echo "Base URL: $BASE_URL"
echo "Timestamp: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0

run_test() {
    local test_name="$1"
    local url="$2"
    local method="$3"
    local data="$4"
    local expected_status="$5"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${BLUE}Test $TOTAL_TESTS: $test_name${NC}"
    echo "URL: $url"
    echo "Method: $method"
    
    if [ "$data" != "" ]; then
        echo "Data: $data"
    fi
    
    # Make request
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}\nTIME_TOTAL:%{time_total}" "$url")
    else
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}\nTIME_TOTAL:%{time_total}" \
                   -X "$method" \
                   -H "Content-Type: application/json" \
                   -d "$data" \
                   "$url")
    fi
    
    # Extract status and time
    http_status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
    time_total=$(echo "$response" | grep "TIME_TOTAL:" | cut -d: -f2)
    response_body=$(echo "$response" | sed '/HTTP_STATUS:/d; /TIME_TOTAL:/d')
    
    # Display response
    echo "Response:"
    echo "$response_body" | python3 -m json.tool 2>/dev/null || echo "$response_body"
    echo "Status: $http_status | Time: ${time_total}s"
    
    # Check result
    if [ "$http_status" -eq "$expected_status" ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAILED (Expected: $expected_status, Got: $http_status)${NC}"
    fi
    
    echo "--------------------------------------"
    echo ""
}

# Wait for server to be ready
echo "⏳ Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready!${NC}"
        echo ""
        break
    fi
    echo -n "."
    sleep 2
done

# Test 1: Health Check
run_test "Health Check" "$BASE_URL/health" "GET" "" "200"

# Test 2: Create Dataset (default parameters)
run_test "Create Dataset (default)" "$BASE_URL/create_dataset" "POST" '{}' "200"

# Test 3: Create Dataset (custom parameters)
run_test "Create Dataset (custom)" "$BASE_URL/create_dataset" "POST" \
'{"partition_type": "train", "balance_strategy": "undersample"}' "200"

# Test 4: Train Model (default parameters)
run_test "Train Model (default)" "$BASE_URL/train_model" "POST" '{}' "200"

# Test 5: Prediction (should work after training)
run_test "Fraud Prediction" "$BASE_URL/predict" "POST" \
'{
    "trans_date_trans_time": "2021-10-07 12:01:55",
    "cc_num": "4059294504000000",
    "unix_time": 1633608115,
    "merchant": "Walmart",
    "category": "grocery_pos",
    "amt": 45.23,
    "merch_lat": 40.7589,
    "merch_long": -73.9851
}' "200"

# Test 6: Error handling - Invalid JSON
run_test "Invalid JSON" "$BASE_URL/predict" "POST" '{invalid json}' "400"

# Test 7: Error handling - Missing required fields
run_test "Missing Fields" "$BASE_URL/predict" "POST" '{"amt": 100}' "400"

# Test 8: Error handling - Invalid partition type
run_test "Invalid Partition Type" "$BASE_URL/create_dataset" "POST" \
'{"partition_type": "invalid"}' "400"

echo "========================================"
echo "🏁 Test Results Summary"
echo "========================================"
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$((TOTAL_TESTS - PASSED_TESTS))${NC}"

if [ "$PASSED_TESTS" -eq "$TOTAL_TESTS" ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    exit 1
fi