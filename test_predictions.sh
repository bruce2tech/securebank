#!/bin/bash
# save as: test_predictions.sh

echo "Generating prediction activity..."

# Make 10 test predictions
for i in {1..10}; do
    curl -X POST http://localhost:5000/predict \
        -H 'Content-Type: application/json' \
        -d '{
            "trans_date_trans_time": "2024-01-15 10:30:00",
            "cc_num": 1234567890123456,
            "unix_time": 1705318200,
            "merchant": "store_'$i'",
            "category": "grocery_pos",
            "amt": '$((RANDOM % 500 + 50))',
            "merch_lat": 40.7128,
            "merch_long": -74.0060
        }'
    echo ""
done

echo "Done! Now check metrics again:"
curl "http://localhost:5000/system_metrics?hours=1" | python3 -m json.tool