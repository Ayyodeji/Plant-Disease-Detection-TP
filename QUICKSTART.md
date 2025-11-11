# 🚀 Quick Start Guide

Get the Plant Disease Detection system up and running in minutes!

## Prerequisites

- Python 3.8+ installed
- 5GB free disk space
- Internet connection (for initial download)
- (Optional) GPU for faster training

## Installation (5 minutes)

### macOS/Linux

```bash
# 1. Navigate to project directory
cd plant_disease_detection

# 2. Run quick start script
chmod +x quick_start.sh
./quick_start.sh
```

### Windows

```bash
# 1. Open Command Prompt or PowerShell
cd plant_disease_detection

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Set Up Kaggle API (2 minutes)

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save the downloaded `kaggle.json`:

```bash
# macOS/Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

## Running the System

### Option A: Complete Demo (Minimal Training)

For a quick demo without full training:

```bash
# Download and prepare a small sample
python demo_quick.py
```

This will:
- Download 1000 sample images
- Train a lightweight model (5 minutes)
- Show example predictions

### Option B: Full Pipeline (Production Quality)

For complete, production-ready models:

```bash
# Run everything (2-8 hours depending on hardware)
python main.py --step all
```

This includes:
1. ✅ Download full dataset (~1.5GB)
2. ✅ Preprocess 50,000+ images
3. ✅ Train 6 models (3 classical + 3 deep learning)
4. ✅ Comprehensive evaluation
5. ✅ Export deployment-ready models

### Option C: Step-by-Step

Run individual components:

```bash
# Step 1: Download dataset (5-10 minutes)
python main.py --step data

# Step 2: Preprocess images (30-60 minutes)
python main.py --step preprocess

# Step 3: Train classical ML (30 minutes)
python main.py --step classical

# Step 4: Train deep learning (1-4 hours)
python main.py --step dl

# Step 5: Evaluate models (10 minutes)
python main.py --step eval

# Step 6: Convert for deployment (5 minutes)
python main.py --step deploy
```

## Making Your First Prediction

Once training is complete:

```bash
python inference_demo.py \
    --image test_images/tomato_leaf.jpg \
    --model-path models/deep_learning/mobilenet_v2_final.h5 \
    --model-type keras
```

**Output:**
```
Top Prediction:
  Disease: Tomato_Late_blight
  Confidence: 97.3%
```

## Viewing Results

After training, check these directories:

```
📁 results/
   ├── model_comparison.csv           # Compare all models
   └── *_evaluation.json              # Detailed metrics

📁 visualizations/
   ├── *_confusion_matrix.png         # Confusion matrices
   ├── *_roc_curves.png              # ROC curves
   └── *_training_curves.png         # Training history

📁 models/
   ├── classical_ml/                  # Traditional ML models
   ├── deep_learning/                 # Neural networks
   └── deployment/                    # Mobile-ready models
```

## What's Next?

### Learn More
- 📖 [Full Documentation](README.md)
- 👥 [User Guide](docs/USER_GUIDE.md)
- 🤝 [Ethics Guidelines](docs/ETHICS.md)

### Try Advanced Features

**Custom Dataset:**
```python
# Train on your own images
python main.py --config my_config.yaml
```

**Batch Processing:**
```python
# Process entire folders
python batch_process.py --input-dir my_images/
```

**Mobile Deployment:**
```python
# Convert to TFLite for Android/iOS
from src.deployment import ModelConverter
converter = ModelConverter()
converter.convert_to_tflite(model, 'mobilenet_v2', quantize=True)
```

## Troubleshooting

### Common Issues

**"No module named..."**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**"Kaggle API error"**
```bash
# Check credentials
cat ~/.kaggle/kaggle.json
# Should show your username and key
```

**"Out of memory"**
```bash
# Reduce batch size in config.yaml
# Change batch_size: 32 to batch_size: 8
```

**"CUDA not available"**
```bash
# Normal! Training will use CPU (slower but works)
# To use GPU, install: pip install tensorflow[and-cuda]
```

## Getting Help

- 💬 Open an issue on GitHub
- 📧 Email: support@example.com
- 📚 Check [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

## Expected Timeline

| Task | Time | Hardware |
|------|------|----------|
| Installation | 5 min | Any |
| Data download | 10 min | Fast internet |
| Preprocessing | 1 hour | Any |
| Classical ML training | 30 min | Any |
| Deep learning training | 2-4 hours | GPU recommended |
| **Total (first time)** | **4-6 hours** | Mid-range PC |

Subsequent runs are faster as data is cached!

## System Requirements

**Minimum:**
- 4GB RAM
- 10GB disk space
- Dual-core CPU

**Recommended:**
- 16GB RAM
- 20GB disk space
- GPU with 4GB+ VRAM
- SSD storage

## Success Indicators

You'll know it's working when you see:

1. ✅ `data/processed/X_train.npy` exists
2. ✅ `models/deep_learning/mobilenet_v2_final.h5` created
3. ✅ `results/model_comparison.csv` shows >95% accuracy
4. ✅ `visualizations/` contains plots

## Next Steps

Congratulations! Your system is ready. Now:

1. 📝 Review model performance in `results/`
2. 🎨 Check visualizations
3. 🧪 Test on your own images
4. 📱 Deploy to mobile devices
5. 🌍 Share with agricultural communities

---

**Happy plant disease detecting! 🌱**



