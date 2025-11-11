#!/bin/bash

# Plant Disease Detection System - Quick Start Script
# This script sets up and runs a demonstration of the system

set -e  # Exit on error

echo "=================================="
echo "Plant Disease Detection System"
echo "Quick Start Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python installation
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install dependencies
echo ""
echo "${YELLOW}Installing dependencies...${NC}"
echo "This may take 5-10 minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Dependencies installed"

# Check for Kaggle credentials
echo ""
echo "Checking Kaggle API credentials..."
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "${GREEN}✓${NC} Kaggle credentials found"
else
    echo -e "${YELLOW}⚠${NC} Kaggle credentials not found"
    echo ""
    echo "To download the dataset, you need Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Save kaggle.json to ~/.kaggle/"
    echo ""
    read -p "Do you have kaggle.json ready? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p ~/.kaggle
        echo "Please enter the path to your kaggle.json file:"
        read KAGGLE_PATH
        cp "$KAGGLE_PATH" ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
        echo -e "${GREEN}✓${NC} Kaggle credentials configured"
    else
        echo ""
        echo "You can set up Kaggle credentials later and run:"
        echo "  python main.py --step data"
        echo ""
    fi
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed
mkdir -p models/classical_ml models/deep_learning models/deployment
mkdir -p results visualizations logs docs
echo -e "${GREEN}✓${NC} Directories created"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Run full pipeline (2-8 hours):"
echo "   ${GREEN}python main.py --step all${NC}"
echo ""
echo "2. Or run individual steps:"
echo "   ${GREEN}python main.py --step data${NC}         # Download dataset"
echo "   ${GREEN}python main.py --step preprocess${NC}   # Preprocess images"
echo "   ${GREEN}python main.py --step classical${NC}    # Train classical ML"
echo "   ${GREEN}python main.py --step dl${NC}           # Train deep learning"
echo "   ${GREEN}python main.py --step eval${NC}         # Evaluate models"
echo ""
echo "3. Make predictions on an image:"
echo "   ${GREEN}python inference_demo.py --image path/to/image.jpg --model-path models/deep_learning/mobilenet_v2_final.h5${NC}"
echo ""
echo "For more information, see README.md and docs/USER_GUIDE.md"
echo ""


