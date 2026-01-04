#!/bin/bash

# setup.sh - Setup SecureBank development environment
# Usage: ./setup.sh

echo "🏦 SecureBank Setup Script"
echo "=========================="

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p logs
mkdir -p storage/datasets
mkdir -p output
mkdir -p data_sources
mkdir -p executables

echo "✓ Directories created"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x executables/*.sh
chmod +x setup.sh

echo "✓ Scripts are now executable"

# Check if Python requirements are installed
echo "🐍 Checking Python dependencies..."
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found"
    
    # Check if pip is available
    if command -v pip3 &> /dev/null; then
        echo "📦 Installing Python dependencies..."
        pip3 install -r requirements.txt
        echo "✓ Dependencies installed"
    else
        echo "⚠️  pip3 not found. Please install dependencies manually:"
        echo "   pip3 install -r requirements.txt"
    fi
else
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

# Check if Docker is available
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✓ Docker found"
    
    # Test Docker access
    if docker ps &> /dev/null; then
        echo "✓ Docker is accessible"
    else
        echo "⚠️  Docker found but not accessible. You may need to:"
        echo "   - Start Docker service"
        echo "   - Add your user to docker group"
        echo "   - Run with sudo"
    fi
else
    echo "⚠️  Docker not found. Docker is required for containerized deployment."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your data files to data_sources/:"
echo "   - customer_release.csv"
echo "   - transactions_release.parquet" 
echo "   - fraud_release.json"
echo ""
echo "2. Run the system:"
echo "   cd executables"
echo "   ./run_server.sh      # Start Docker container"
echo "   ./test_all_endpoints.sh  # Test all functionality"
echo ""
echo "3. Or run locally for development:"
echo "   python3 app.py       # Start Flask server"
echo ""
echo "For more information, see README.md"