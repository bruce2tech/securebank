#!/bin/bash

# run_server.sh - Build and run SecureBank Docker container (SIMPLE FIX)
# Usage: ./run_server.sh

# Define image name and port
IMAGE_NAME="securebank"
PORT="5000"  # Hard-coded to avoid parsing issues

echo "🏦 Starting SecureBank Docker Container"
echo "======================================="

# Check if the Docker image already exists
if docker images | grep -q "^securebank "; then
    echo "✅ SecureBank Docker image found"
    echo "Using existing image..."
else
    echo "🔨 Building SecureBank Docker image..."
    
    # Check if Dockerfile exists
    if [ ! -f "../Dockerfile" ]; then
        echo "❌ Error: Dockerfile not found at ../Dockerfile"
        exit 1
    fi

    # Build the Docker image
    echo "Building Docker image: $IMAGE_NAME..."
    docker build -t "$IMAGE_NAME" -f "../Dockerfile" ..

    # Check if the build was successful
    if [ $? -eq 0 ]; then
        echo "✅ Docker image '$IMAGE_NAME' built successfully."
    else
        echo "❌ Error: Failed to build Docker image."
        exit 1
    fi
fi

# Stop any existing container with the same name
echo "🧹 Cleaning up existing containers..."
docker stop securebank-container 2>/dev/null || true
docker rm securebank-container 2>/dev/null || true

echo "🚀 Starting SecureBank container..."
echo "Container name: securebank-container"
echo "Port mapping: $PORT:$PORT"
echo "----------------------------------------"

# Run the Docker container
docker run -d \
    --name securebank-container \
    -p "$PORT:$PORT" \
    -v "$(pwd)/../logs:/app/logs" \
    -v "$(pwd)/../storage:/app/storage" \
    -v "$(pwd)/../output:/app/output" \
    "$IMAGE_NAME"

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo "✅ SecureBank container started successfully!"
    
    # Wait for the container to be ready
    echo ""
    echo "⏳ Waiting for container to be ready..."
    for i in {1..30}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "✅ Container is ready and responding!"
            break
        fi
        echo -n "."
        sleep 2
        if [ $i -eq 30 ]; then
            echo ""
            echo "⚠️  Container may be slow to start. Check status manually:"
        fi
    done
    
    echo ""
    echo "📊 Container Status:"
    docker ps | grep securebank-container
    echo ""
    echo "🌐 Application URLs:"
    echo "- Health check: http://localhost:$PORT/health"
    echo "- API endpoints: http://localhost:$PORT/"
    echo "- Enhanced logging: http://localhost:$PORT/logs/status"
    echo ""
    echo "📋 Useful Commands:"
    echo "- View logs: docker logs securebank-container"
    echo "- Stop container: docker stop securebank-container"
    echo "- Test Phase 3 logging: ../test_enhanced_logging.sh"
    echo ""
    
    # Test the health endpoint
    health_response=$(curl -s "http://localhost:$PORT/health" 2>/dev/null)
    if [ ! -z "$health_response" ]; then
        echo "💚 Health Check Response:"
        echo "$health_response" | python3 -m json.tool 2>/dev/null || echo "$health_response"
    fi
    
else
    echo "❌ Error: Failed to start container."
    echo ""
    echo "🔍 Troubleshooting:"
    echo "- Check if port $PORT is already in use: lsof -i :$PORT"
    echo "- Check Docker logs: docker logs securebank-container"
    echo "- Try manual start: docker run -p $PORT:$PORT securebank"
    exit 1
fi