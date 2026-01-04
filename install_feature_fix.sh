#!/bin/bash

# Installation script for feature mismatch fix
# This script backs up existing files and installs the fixed versions

echo "🔧 SECUREBANK FEATURE MISMATCH FIX INSTALLER"
echo "============================================="
echo ""

# Configuration
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create backup directory
echo "📦 Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Function to backup file
backup_file() {
    local file=$1
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/$(basename $file).bak"
        echo "  ✓ Backed up: $file"
    fi
}

# Function to create directory if it doesn't exist
ensure_dir() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  ✓ Created directory: $dir"
    fi
}

echo ""
echo "🔄 Backing up existing files..."
echo "--------------------------------"

# Backup existing files
backup_file "app.py"
backup_file "modules/model/fraud_model.py"
backup_file "modules/features/feature_engineer.py"

echo ""
echo "📁 Ensuring directory structure..."
echo "-----------------------------------"

# Ensure all necessary directories exist
ensure_dir "modules/model"
ensure_dir "modules/features"
ensure_dir "modules/data"
ensure_dir "storage/models"
ensure_dir "storage/datasets"
ensure_dir "logs"

echo ""
echo "📝 Creating fixed components..."
echo "-------------------------------"

# Create __init__.py files
touch modules/__init__.py
touch modules/model/__init__.py
touch modules/features/__init__.py
touch modules/data/__init__.py

# Create the fixed feature_engineer.py
cat > modules/features/feature_engineer.py << 'EOF'
# AUTO-GENERATED: Fixed Feature Engineering Module
# This file contains the corrected feature engineering pipeline
# that prevents duplicate columns and ensures consistent features

[INSERT CONTENT FROM fixed_feature_engineer ARTIFACT HERE]
EOF

# Create the fixed fraud_model.py
cat > modules/model/fraud_model.py << 'EOF'
# AUTO-GENERATED: Fixed Fraud Model Module
# This file contains the corrected model that properly handles features
# without falling back to rule-based predictions

[INSERT CONTENT FROM fixed_fraud_model ARTIFACT HERE]
EOF

# Create the fixed app.py
cat > app.py << 'EOF'
# AUTO-GENERATED: Fixed Flask Application
# This file contains the corrected API that uses the fixed components

[INSERT CONTENT FROM fixed_app ARTIFACT HERE]
EOF

echo -e "${YELLOW}  ⚠ Note: Replace [INSERT CONTENT...] markers with actual code${NC}"

echo ""
echo "📋 Creating test script..."
echo "-------------------------"

# Create the test script
cat > test_feature_fix.sh << 'EOF'
[INSERT CONTENT FROM test_feature_fix ARTIFACT HERE]
EOF

chmod +x test_feature_fix.sh

echo "  ✓ Created test_feature_fix.sh"

echo ""
echo "🔍 Verification Steps"
echo "--------------------"
echo ""
echo "1. Copy the fixed code from the artifacts into the files:"
echo "   - modules/features/feature_engineer.py"
echo "   - modules/model/fraud_model.py"
echo "   - app.py"
echo ""
echo "2. Start your application:"
echo "   python app.py"
echo ""
echo "3. Run the test script:"
echo "   ./test_feature_fix.sh"
echo ""
echo "4. Check the test results for:"
echo "   - Features used > 9 (indicates engineered features)"
echo "   - Probabilistic outputs (not just 0 or 1)"
echo "   - Consistent feature counts"
echo ""
echo "✅ Installation preparation complete!"
echo "📁 Original files backed up in: $BACKUP_DIR/"
echo ""
echo "If tests fail, you can restore from backup:"
echo "  cp $BACKUP_DIR/*.bak ."
echo ""