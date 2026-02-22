# Python Version Compatibility Guide

## Supported Python Versions

This project supports **Python 3.9 - 3.13**.

### Recommended Version
- **Python 3.11** or **Python 3.12** (best balance of stability and performance)

### Version-Specific Notes

#### Python 3.13 (Latest) ✅
- **Status**: Fully supported
- **Requirements**: Updated dependencies (see requirements.txt)
- **Benefits**: Latest features, improved performance
- **Notes**: All dependencies have wheels for Python 3.13

#### Python 3.12 ✅
- **Status**: Fully supported
- **Requirements**: Standard dependencies
- **Benefits**: Stable, good performance
- **Recommended**: Yes

#### Python 3.11 ✅
- **Status**: Fully supported (recommended)
- **Requirements**: Standard dependencies
- **Benefits**: Very stable, excellent performance
- **Recommended**: Yes

#### Python 3.10 ✅
- **Status**: Fully supported
- **Requirements**: Standard dependencies
- **Benefits**: Stable, widely used

#### Python 3.9 ⚠️
- **Status**: Supported but nearing end-of-life
- **Requirements**: Standard dependencies
- **Notes**: Consider upgrading to 3.11 or 3.12

#### Python 3.8 and below ❌
- **Status**: NOT supported
- **Reason**: Missing required features and type hints

## Installation Issues by Platform

### macOS (Apple Silicon / M1/M2/M3)

#### Python 3.13
**Issue**: Pillow build errors
**Solution**: Use requirements.txt with Pillow>=10.4.0

```bash
pip install -r requirements.txt
```

If you still have issues:
```bash
# Install system dependencies via Homebrew
brew install libjpeg zlib

# Then retry
pip install -r requirements.txt
```

#### PyTorch on Apple Silicon
PyTorch works great on Apple Silicon with MPS (Metal Performance Shaders) acceleration:

```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

### macOS (Intel)

Works with all supported Python versions. Use:
```bash
pip install -r requirements.txt
```

### Linux

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip libjpeg-dev zlib1g-dev

# Install Python requirements
pip install -r requirements.txt
```

#### CentOS/RHEL
```bash
# Install system dependencies
sudo yum install python3-devel python3-pip libjpeg-turbo-devel zlib-devel

# Install Python requirements
pip install -r requirements.txt
```

### Windows

Works with all supported Python versions:
```bash
pip install -r requirements.txt
```

## Checking Your Python Version

```bash
python --version
# or
python3 --version
```

## Creating a Virtual Environment

### Recommended Method (venv)

```bash
# Create virtual environment
python -m venv venv

# Activate
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Alternative (conda)

```bash
# Create environment with specific Python version
conda create -n cardamom python=3.12

# Activate
conda activate cardamom

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

### Problem: Pillow won't build

**Symptoms:**
```
error: subprocess-exited-with-error
Getting requirements to build wheel did not run successfully
```

**Solution 1**: Update to Python 3.11 or 3.12
```bash
# Using pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Or using conda
conda create -n cardamom python=3.12
```

**Solution 2**: Install system dependencies
```bash
# macOS
brew install libjpeg zlib

# Ubuntu/Debian
sudo apt-get install libjpeg-dev zlib1g-dev

# CentOS/RHEL
sudo yum install libjpeg-turbo-devel zlib-devel
```

**Solution 3**: Use updated requirements.txt
Make sure you have the latest requirements.txt with Pillow>=10.4.0

### Problem: NumPy compatibility errors

**Symptoms:**
```
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
pip install --upgrade numpy
```

### Problem: PyTorch CUDA not working

**For NVIDIA GPU users:**

Check CUDA version:
```bash
nvidia-smi
```

Install matching PyTorch version:
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Problem: Import errors after installation

**Solution:**
```bash
# Reinstall in clean environment
pip uninstall -y torch torchvision pillow numpy opencv-python
pip install -r requirements.txt
```

## Performance Tips

### GPU Acceleration

**NVIDIA GPU (CUDA):**
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Apple Silicon (MPS):**
```python
import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

### Training Speed by Python Version

Approximate relative performance (Python 3.11 = 100%):
- Python 3.13: ~105% (fastest)
- Python 3.12: ~103%
- Python 3.11: 100% (baseline)
- Python 3.10: ~95%
- Python 3.9: ~92%

## Dependency Version Matrix

| Python | FastAPI | Pillow | PyTorch | NumPy | Status |
|--------|---------|--------|---------|-------|--------|
| 3.13   | 0.109.1+ | 10.4.0+ | 2.1.0+ | 1.26.4+ | ✅ |
| 3.12   | 0.109.0+ | 10.2.0+ | 2.0.0+ | 1.24.0+ | ✅ |
| 3.11   | 0.100.0+ | 10.0.0+ | 2.0.0+ | 1.24.0+ | ✅ |
| 3.10   | 0.100.0+ | 9.5.0+ | 1.13.0+ | 1.23.0+ | ✅ |
| 3.9    | 0.100.0+ | 9.5.0+ | 1.13.0+ | 1.22.0+ | ⚠️ |

## Getting Help

If you continue to have Python version issues:

1. Check this guide for your specific Python version
2. Try Python 3.11 or 3.12 (most stable)
3. Make sure you're in a virtual environment
4. Check system dependencies are installed
5. Open an issue with:
   - Python version (`python --version`)
   - Operating system
   - Error message
   - Output of `pip list`

## Quick Reference

**Best Python versions for this project:**
1. **Python 3.12** - Best overall choice
2. **Python 3.11** - Most stable
3. **Python 3.13** - Latest features (if you need them)

**Installation command:**
```bash
pip install -r requirements.txt
```

**If you have issues:**
1. Update pip: `pip install --upgrade pip`
2. Install system dependencies (see platform sections above)
3. Try a different Python version (3.11 or 3.12 recommended)
4. Use a fresh virtual environment
