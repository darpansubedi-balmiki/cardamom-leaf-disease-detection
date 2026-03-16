#!/bin/bash

# Setup script for Cardamom Leaf Disease Detection Backend
# This script installs all required dependencies and verifies the installation

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "Setting up Cardamom Disease Detection Backend"
echo "============================================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python $PYTHON_VERSION detected"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}❌ Error: Python 3.9 or higher is required${NC}"
    echo "Current version: $PYTHON_VERSION"
    echo "Please upgrade Python and try again"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -gt 13 ]; then
    echo -e "${YELLOW}⚠️  Warning: Python $PYTHON_VERSION is newer than tested versions (3.9-3.13)${NC}"
    echo "Installation may work, but compatibility is not guaranteed"
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION - Compatible!${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip upgraded to latest version${NC}"
echo ""

# Install requirements
echo "Installing dependencies from requirements.txt..."
echo "This may take 3-5 minutes depending on your internet connection..."
echo ""

if pip install -r requirements.txt; then
    echo ""
    echo -e "${GREEN}✅ All dependencies installed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Error installing dependencies${NC}"
    echo "Please check the error messages above and try:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "For help, see: INSTALL_DEPENDENCIES.md"
    exit 1
fi

echo ""
echo "============================================================"
echo "Running diagnostic check..."
echo "============================================================"
echo ""

# Run diagnostic check if it exists
if [ -f "check_training_setup.py" ]; then
    if python check_training_setup.py; then
        echo ""
        echo "============================================================"
        echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
        echo "============================================================"
        echo ""
        echo "Next steps:"
        echo "1. Verify setup: python check_training_setup.py"
        echo "2. Start training: python train.py"
        echo "   (Make sure you have the dataset in dataset/ folder)"
        echo "3. Or start API: uvicorn app.main:app --reload"
        echo ""
        echo "For more information:"
        echo "- Training guide: ../START_TRAINING_NOW.md"
        echo "- Documentation: ../DOCUMENTATION_INDEX.md"
        echo "============================================================"
    else
        echo ""
        echo "============================================================"
        echo -e "${YELLOW}⚠️  Setup completed with warnings${NC}"
        echo "============================================================"
        echo ""
        echo "Dependencies are installed, but diagnostic check found issues."
        echo "Review the messages above and see:"
        echo "- TRAIN_PY_ERRORS.md for troubleshooting"
        echo "- INSTALL_DEPENDENCIES.md for installation help"
        echo "============================================================"
    fi
else
    echo ""
    echo "============================================================"
    echo -e "${GREEN}✅ DEPENDENCIES INSTALLED!${NC}"
    echo "============================================================"
    echo ""
    echo "Note: check_training_setup.py not found"
    echo "You can still proceed with training or running the API"
    echo ""
    echo "Next steps:"
    echo "1. Start training: python train.py"
    echo "2. Or start API: uvicorn app.main:app --reload"
    echo "============================================================"
fi
