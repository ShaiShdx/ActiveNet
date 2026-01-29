#!/bin/bash
# Active Network Simulation - Setup Script
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "=================================================="
echo "Active Network Simulation - Setup"
echo "=================================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Remove it? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Optional: Install PyTorch for CUDA support
echo ""
echo "Install PyTorch for CUDA support? (y/n)"
read -r response
if [ "$response" = "y" ]; then
    echo "Installing PyTorch with CUDA 11.8..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118
fi

# Verify installation
echo ""
echo "=================================================="
echo "Verifying installation..."
echo "=================================================="
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python -c "import networkx; print(f'NetworkX: {networkx.__version__}')"

# Check PyTorch if installed
if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
fi

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the simulation:"
echo "  python main.py              # CPU only"
echo "  python main_cuda.py         # GPU if available"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
