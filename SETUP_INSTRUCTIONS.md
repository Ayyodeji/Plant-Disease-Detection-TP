# Plant Disease Detection - Setup & Run Instructions

**Simple guide for running this plant disease detection system**

---

## What This Does

This system trains AI models to identify plant diseases from leaf images. It:
- Downloads a plant disease dataset
- Trains multiple machine learning models
- Creates predictions models you can use

**Warning:** This is resource-intensive and may take 2-8 hours on a good computer.

---

## System Requirements

**Minimum:**
- **RAM:** 8GB (16GB+ recommended)
- **Storage:** 10GB free space
- **Internet:** Stable connection for downloading dataset (~2GB)
- **Time:** 2-8 hours to complete

**Operating System:**
- macOS, Linux, or Windows with WSL

---

## Setup Instructions

### Step 1: Check Python Installation

Open Terminal and check Python version:

```bash
python3 --version
```

You need **Python 3.8 or higher**. If not installed, download from [python.org](https://www.python.org/downloads/)

---

### Step 2: Navigate to Project Folder

```bash
cd /path/to/plant_disease_detection
```

Replace `/path/to/` with wherever you saved this folder.

---

### Step 3: Create Virtual Environment

This keeps dependencies isolated:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal.

---

### Step 4: Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all necessary libraries. May take 5-10 minutes.

---

### Step 5: Setup Kaggle API (For Dataset Download)

The system downloads data from Kaggle. You need an API key:

1. Go to [kaggle.com](https://www.kaggle.com) and sign in (create free account if needed)
2. Click your profile picture → **Account**
3. Scroll to **API** section → Click **Create New API Token**
4. This downloads `kaggle.json`

Now install it:

```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded file there
# (Replace ~/Downloads with wherever it downloaded)
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

---

## Running The System

### Option 1: Run Everything (Simplest)

```bash
python main.py --step all
```

**This will:**
1. Download dataset (~2GB, takes 10-30 min)
2. Process all images
3. Train 3 classical machine learning models
4. Train 3 deep learning models
5. Evaluate everything
6. Save results

**Expected time:** 2-8 hours depending on your computer

---

### Option 2: Run Step-by-Step (If Crashes Occur)

If your computer crashes or runs out of memory, run one step at a time:

```bash
# Step 1: Download data
python main.py --step data

# Step 2: Preprocess images
python main.py --step preprocess

# Step 3: Extract features
python main.py --step features

# Step 4: Train classical ML (lighter on memory)
python main.py --step classical

# Step 5: Train deep learning (heavy on memory - might cause crashes)
python main.py --step dl

# Step 6: Evaluate models
python main.py --step eval

# Step 7: Prepare for deployment
python main.py --step deploy
```

**Tip:** If Step 5 (deep learning) crashes, skip it. Classical ML models from Step 4 work fine!

---

### Option 3: Skip Deep Learning (Recommended for Lower-End PCs)

To avoid crashes, train only classical ML models:

```bash
# Just run steps 1-4
python main.py --step data
python main.py --step preprocess
python main.py --step features
python main.py --step classical
```

You'll get working models without the heavy deep learning training.

---

## If Computer Crashes / Out of Memory

### Reduce Memory Usage

Edit `config.yaml` and change these values:

```yaml
data:
  max_classes: 5              # Reduce from 38 to 5 (fewer plant types)
  max_images_per_class: 500   # Reduce from unlimited to 500

deep_learning:
  batch_size: 16              # Reduce from 32 to 16 or even 8
```

Then try running again.

---

### Monitor Memory Usage

Before running, check available RAM:

**Mac/Linux:**
```bash
# Check memory
free -h   # Linux
vm_stat   # Mac
```

**Windows:**
- Open Task Manager (Ctrl+Shift+Esc)
- Watch Memory usage

Close other programs before running.

---

## What Gets Created

After running, you'll see:

```
plant_disease_detection/
├── data/
│   ├── raw/                    # Downloaded dataset
│   └── processed/              # Processed images (.npy files)
│
├── models/
│   ├── classical_ml/           # Trained ML models (.pkl files)
│   │   ├── random_forest.pkl
│   │   ├── svm.pkl
│   │   └── gradient_boosting.pkl
│   └── deep_learning/          # Trained neural networks (.h5 files)
│       ├── mobilenet_v2_final.h5
│       └── ...
│
├── results/
│   ├── model_comparison.csv    # Performance comparison
│   └── *_evaluation.json       # Detailed metrics
│
└── visualizations/
    ├── *_confusion_matrix.png  # Visual results
    └── *_roc_curves.png        # Performance graphs
```

---

## How to Know It Worked

Look for these signs of success:

1. **No error messages** in terminal
2. **Files created** in `models/`, `results/`, `visualizations/`
3. **Final message:**
   ```
   PIPELINE COMPLETE!
   All models trained, evaluated, and ready for deployment.
   ```

---

## Testing A Model

Once trained, test a model on a single image:

```bash
python inference_demo.py \
    --image /path/to/leaf_image.jpg \
    --model-path models/classical_ml/random_forest.pkl \
    --model-type sklearn
```

Or for deep learning model:

```bash
python inference_demo.py \
    --image /path/to/leaf_image.jpg \
    --model-path models/deep_learning/mobilenet_v2_final.h5 \
    --model-type keras
```

---

## Common Problems & Solutions

### Problem: "ModuleNotFoundError"

**Solution:** Activate virtual environment and reinstall:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### Problem: "Kaggle API credentials not found"

**Solution:** Check kaggle.json is in correct location:
```bash
ls ~/.kaggle/kaggle.json
```
If missing, repeat Step 5.

---

### Problem: "Out of memory" / Computer freezes

**Solution:** 
1. Edit `config.yaml` (see "If Computer Crashes" section above)
2. Run step-by-step instead of all at once
3. Close other applications
4. Skip deep learning: only run classical ML

---

### Problem: "CUDA out of memory" (GPU error)

**Solution:** Add this to reduce GPU memory:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
python main.py --step classical
```
Or just train classical ML models (they don't use GPU).

---

### Problem: Takes too long / seems stuck

**Check if it's actually working:**
- Look for progress bars in terminal
- Check CPU usage (should be high)
- Check if files are being created in `data/` folder

**Normal timeframes:**
- Data download: 10-30 minutes
- Preprocessing: 30-60 minutes  
- Training: 1-6 hours

---

## Understanding Results

After completion, check these files:

**`results/model_comparison.csv`**
- Shows which model performed best
- Look for highest "accuracy" and "f1_score"

**`visualizations/model_comparison.png`**
- Visual comparison of all models
- Taller bars = better performance

**`results/*_evaluation.json`**
- Detailed metrics for each model

---

## Backing Up Results

After successful run, backup these important folders:

```bash
# Create backup
tar -czf plant_disease_results.tar.gz models/ results/ visualizations/

# This creates a single compressed file with all outputs
```

---

## Starting Fresh

If something goes wrong and you want to restart:

```bash
# Remove generated files
rm -rf data/ models/ results/ visualizations/ logs/

# Then run again
python main.py --step all
```

---

## Getting Help

If stuck:

1. Check the error message carefully
2. Look in the "Common Problems" section above
3. Check RAM usage (close other apps)
4. Try reducing dataset size in `config.yaml`
5. Try classical ML only (skip deep learning)

---

## Quick Command Reference

```bash
# Full run (all steps)
python main.py --step all

# Individual steps
python main.py --step data         # Download dataset
python main.py --step preprocess   # Process images
python main.py --step features     # Extract features
python main.py --step classical    # Train ML models (lighter)
python main.py --step dl           # Train neural networks (heavy)
python main.py --step eval         # Evaluate models
python main.py --step deploy       # Export for production

# Test a model
python inference_demo.py --image leaf.jpg --model-path models/classical_ml/random_forest.pkl --model-type sklearn
```

---

## Expected Timeline

| Task | Time | Memory Usage |
|------|------|--------------|
| Dataset download | 10-30 min | Low |
| Preprocessing | 30-60 min | Medium |
| Feature extraction | 20-40 min | Medium |
| Classical ML training | 30-90 min | Medium-High |
| Deep learning training | 2-6 hours | **Very High** |
| Evaluation | 10-20 min | Medium |

**Total:** 4-8 hours for complete pipeline

---

## Recommended Approach

**For regular computers (8-16GB RAM):**
```bash
# 1. Try classical ML only first
python main.py --step data
python main.py --step preprocess  
python main.py --step features
python main.py --step classical

# 2. Check results
ls -lh models/classical_ml/
cat results/model_comparison.csv

# 3. If successful and you have time/resources, try deep learning
python main.py --step dl
```

**For powerful computers (16GB+ RAM):**
```bash
# Just run everything
python main.py --step all
```

---

**Good luck! The models will be ready to identify plant diseases once training completes.**


