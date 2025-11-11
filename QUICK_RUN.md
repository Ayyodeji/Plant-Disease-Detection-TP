# Quick Run Guide - Plant Disease Detection

## TL;DR - Just Run It

### 1. Setup (One Time Only)

```bash
cd plant_disease_detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR: venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt

# Setup Kaggle (for dataset)
# Get kaggle.json from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

### 2. Run Training

**Option A - Full Pipeline (8+ hours, needs 16GB RAM):**
```bash
python main.py --step all
```
---

### 3. Check Results

```bash
# View results
cat results/model_comparison.csv

# Models saved here
ls models/classical_ml/
ls models/deep_learning/
```

---

## If It Crashes

Edit `config.yaml` and change:

```yaml
data:
  max_classes: 5              # Use only 5 plant types
  max_images_per_class: 300   # Limit images

deep_learning:
  batch_size: 8               # Use less memory
```

Then run again.

---

## What You'll Get

- **Models:** Trained AI models in `models/` folder
- **Results:** Performance metrics in `results/` folder  
- **Graphs:** Visual comparisons in `visualizations/` folder

Done!


