#!/bin/bash

# ============================================================================
# TFT Quant Strategy - Complete Setup Script
# ============================================================================
# This script sets up the environment and verifies everything is working
# Run this with Git Bash on Windows
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# Step 0: Check Prerequisites
# ============================================================================
print_header "Step 0: Checking Prerequisites"

# Check Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python not found! Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    print_error "Python 3.8 or higher required. Current: $PYTHON_VERSION"
    exit 1
fi

# Check pip
if command -v pip &> /dev/null; then
    print_success "pip found: $(pip --version)"
else
    print_error "pip not found!"
    exit 1
fi

# Check CUDA (optional)
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n 1
    GPU_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected. Will use CPU (training will be slower)"
    GPU_AVAILABLE=false
fi

# ============================================================================
# Step 1: Create Virtual Environment
# ============================================================================
print_header "Step 1: Creating Virtual Environment"

if [ -d "venv" ]; then
    print_warning "Virtual environment 'venv' already exists"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf venv
        print_info "Creating new virtual environment..."
        python -m venv venv
        print_success "Virtual environment created"
    else
        print_info "Using existing virtual environment"
    fi
else
    print_info "Creating virtual environment..."
    python -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/Scripts/activate || source venv/bin/activate

print_success "Virtual environment activated"

# ============================================================================
# Step 2: Upgrade pip and Install Requirements
# ============================================================================
print_header "Step 2: Installing Dependencies"

print_info "Upgrading pip..."
python -m pip install --upgrade pip

print_info "Installing PyTorch (this may take a few minutes)..."
if [ "$GPU_AVAILABLE" = true ]; then
    print_info "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio
    #pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_info "Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio
fi

print_success "PyTorch installed"

print_info "Installing other dependencies..."
pip install -r requirements.txt

print_success "All dependencies installed"

# ============================================================================
# Step 3: Verify Installation
# ============================================================================
print_header "Step 3: Verifying Installation"

print_info "Checking imports..."

python << EOF
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: {e}")
        return False

print("\nCore Libraries:")
check_import("torch", "PyTorch")
check_import("numpy", "NumPy")
check_import("pandas", "Pandas")
check_import("dask", "Dask")

print("\nOptimization:")
check_import("optuna", "Optuna")

print("\nData Processing:")
check_import("pyarrow", "PyArrow")

print("\nChecking CUDA availability:")
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("⚠ CUDA not available (will use CPU)")

print("\nChecking cuDF availability (optional):")
try:
    import cudf
    print("✓ cuDF available (GPU acceleration enabled)")
except ImportError:
    print("⚠ cuDF not available (will use Pandas)")

print("\nPython Environment:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Location: {sys.executable}")
EOF

print_success "Import verification complete"

# ============================================================================
# Step 4: Check Directory Structure
# ============================================================================
print_header "Step 4: Checking Directory Structure"

# Check for required files
REQUIRED_FILES=(
    "config.py"
    "data_converter.py"
    "data_processor_streaming.py"
    "tft_model_enhanced.py"
    "trainer_enhanced.py"
    "trading_strategy_enhanced.py"
    "hyperparameter_tuning.py"
    "main_enhanced.py"
    "test.py"
)

print_info "Checking for required Python files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file"
    else
        print_error "$file NOT FOUND!"
        exit 1
    fi
done

# Check for data directory
if [ -d "EBY" ]; then
    CSV_COUNT=$(find EBY -name "*.csv" -type f | wc -l)
    print_success "Data directory 'EBY' found with $CSV_COUNT CSV files"
else
    print_warning "Data directory 'EBY' not found"
    print_info "Please ensure your CSV files are in the 'EBY' directory"
    print_info "Expected structure: EBY/day0.csv, EBY/day1.csv, etc."
fi

# Check/Create output directories
REQUIRED_DIRS=("models" "results" "optuna_studies" "cache" "EBY_parquet")

print_info "Checking/creating output directories..."
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created: $dir"
    else
        print_success "Exists: $dir"
    fi
done

# ============================================================================
# Step 5: Quick Test (Optional)
# ============================================================================
print_header "Step 5: Run Quick Test?"

echo "Would you like to run a quick test to verify everything works?"
echo "This will test the complete workflow on minimal data (2-5 minutes)."
echo ""
read -p "Run quick test? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Running quick test..."
    echo ""
    
    python test.py
    
    if [ $? -eq 0 ]; then
        print_success "Quick test passed! System is ready."
    else
        print_error "Quick test failed. Please check the errors above."
        exit 1
    fi
else
    print_info "Skipping quick test"
fi

# ============================================================================
# Final Summary
# ============================================================================
print_header "Setup Complete!"

echo -e "${GREEN}Your environment is ready!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment (if not already active):"
echo "   ${YELLOW}source venv/Scripts/activate${NC}  # Git Bash"
echo "   ${YELLOW}venv\\Scripts\\activate${NC}         # Windows CMD"
echo ""
echo "2. Run the quick test:"
echo "   ${YELLOW}python test.py${NC}"
echo ""
echo "3. Run the full pipeline:"
echo "   ${YELLOW}python main_enhanced.py${NC}"
echo ""
echo "4. Or skip certain steps:"
echo "   ${YELLOW}python main_enhanced.py --skip-conversion${NC}"
echo "   ${YELLOW}python main_enhanced.py --skip-tuning${NC}"
echo ""
echo "Documentation:"
echo "  - See MODIFICATIONS_SUMMARY.md for detailed changes"
echo "  - Check config.py for configuration options"
echo ""
echo -e "${GREEN}Happy trading! 🚀${NC}"