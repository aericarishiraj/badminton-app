#!/bin/bash
echo "Installing Badminton Analysis App dependencies..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher first."
    echo "You can download it from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Dependencies installed successfully!"
echo "You can now run the app with: python3 app.py"
echo "Or build the distributable with: python3 build_mac.py"
