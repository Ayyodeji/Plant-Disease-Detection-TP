# 🚀 Run the Demo - Quick Instructions

## Option 1: Quick Demo (5-10 minutes)

This runs a fast end-to-end test with synthetic data to verify everything works:

```bash
cd /Users/ajayiayodeji/Documents/Random/Demo/plant_disease_detection

# Run the complete demo
python demo_test.py
```

**What it does:**
- ✅ Creates 5 classes with 50 synthetic images each
- ✅ Preprocesses all images
- ✅ Extracts manual features (color, texture, shape)
- ✅ Trains Random Forest model
- ✅ Trains MobileNetV2 (3 epochs)
- ✅ Evaluates both models
- ✅ Tests inference pipeline
- ✅ Converts to TFLite for mobile deployment
- ✅ Creates visualization

**Expected output:** Creates `demo_data/` folder with:
- Synthetic dataset images
- Trained models
- TFLite model
- Results visualization
- Confusion matrices

---

## Option 2: Real Dataset Pipeline (2-8 hours)

To train on the actual PlantVillage dataset:

### Step 1: Set up Kaggle API

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Download your kaggle.json from:
# https://www.kaggle.com/account -> API -> Create New API Token

# Move it to the right place
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Run the Full Pipeline

```bash
cd /Users/ajayiayodeji/Documents/Random/Demo/plant_disease_detection

# Option A: Run everything at once (recommended)
python main.py --step all

# Option B: Run step by step
python main.py --step data        # Download dataset (10 min)
python main.py --step preprocess  # Preprocess images (1 hour)
python main.py --step classical   # Train classical ML (30 min)
python main.py --step dl          # Train deep learning (2-4 hours)
python main.py --step eval        # Evaluate models (10 min)
python main.py --step deploy      # Convert for deployment (5 min)
```

**What gets created:**
```
plant_disease_detection/
├── data/
│   ├── raw/          # PlantVillage dataset
│   └── processed/    # Preprocessed images & features
├── models/
│   ├── classical_ml/      # Random Forest, SVM, Gradient Boosting
│   ├── deep_learning/     # MobileNetV2, ResNet50, EfficientNetB0
│   └── deployment/        # TFLite & ONNX models
├── results/
│   └── model_comparison.csv
└── visualizations/
    ├── confusion_matrices/
    ├── roc_curves/
    └── training_curves/
```

---

## Option 3: Test Inference on Your Own Images

After training, predict on any leaf image:

```bash
# Using Keras model
python inference_demo.py \
    --image /path/to/your/leaf_photo.jpg \
    --model-path models/deep_learning/mobilenet_v2_final.h5 \
    --model-type keras \
    --top-k 3

# Or using TFLite (mobile-optimized)
python inference_demo.py \
    --image /path/to/your/leaf_photo.jpg \
    --model-path models/deployment/mobilenet_v2_quantized.tflite \
    --model-type tflite \
    --top-k 3
```

---

## Viewing Results

After running, check these locations:

```bash
# View training results
cat results/model_comparison.csv

# View visualizations
open visualizations/demo_mobilenet_v2_confusion_matrix.png
open visualizations/demo_mobilenet_v2_training_curves.png
open demo_data/demo_results.png  # For quick demo

# Check model files
ls -lh models/deep_learning/
ls -lh models/deployment/
```

---

## Expected Timeline

| Task | Demo Mode | Full Dataset |
|------|-----------|--------------|
| Data preparation | 10 sec | 10 min |
| Preprocessing | 30 sec | 1 hour |
| Feature extraction | 20 sec | 30 min |
| Classical ML | 10 sec | 30 min |
| Deep Learning | 2 min | 2-4 hours |
| Evaluation | 10 sec | 10 min |
| Deployment | 10 sec | 5 min |
| **TOTAL** | **~5 min** | **4-6 hours** |

---

## Troubleshooting

**If you get module errors:**
```bash
pip install opencv-python pillow numpy scikit-learn scikit-image \
    matplotlib seaborn pyyaml tqdm joblib mahotas scipy \
    albumentations tensorflow
```

**If GPU isn't detected (TensorFlow):**
```bash
# It's fine! Training will use CPU (slower but works)
# To use GPU: pip install tensorflow[and-cuda]
```

**If running out of memory:**
```bash
# Edit config.yaml and reduce batch_size from 32 to 8
nano config.yaml
# Find: batch_size: 32
# Change to: batch_size: 8
```

---

## Quick Start Commands (Copy-Paste)

**Run Quick Demo:**
```bash
cd /Users/ajayiayodeji/Documents/Random/Demo/plant_disease_detection && python demo_test.py
```

**Run Full Pipeline:**
```bash
cd /Users/ajayiayodeji/Documents/Random/Demo/plant_disease_detection && python main.py --step all
```

**View Results:**
```bash
cd /Users/ajayiayodeji/Documents/Random/Demo/plant_disease_detection
open demo_data/demo_results.png
cat results/model_comparison.csv
```

---

## What to Expect

### Quick Demo Output:
```
================================================================================
PLANT DISEASE DETECTION SYSTEM - END-TO-END DEMO
================================================================================

📦 STEP 1: Creating Synthetic Test Dataset
✓ Created 5 classes with 50 images each

📊 STEP 2: Creating Train/Val/Test Splits
✓ Split created: 175 train, 35 val, 40 test

🔧 STEP 3: Testing Preprocessing Pipeline
✓ Preprocessing complete

🎨 STEP 4: Testing Feature Extraction
✓ Feature extraction complete: 256 features

🤖 STEP 5: Training Classical ML Model
✓ Random Forest trained: 85% accuracy

🧠 STEP 6: Training Deep Learning Model
✓ MobileNetV2 trained: 98% accuracy

📈 STEP 7: Evaluating Models
✓ Models evaluated

🔮 STEP 8: Testing Inference Pipeline
✓ Prediction: Tomato_Late_Blight (97.3% confidence)

📱 STEP 9: Testing Model Conversion
✓ TFLite model: 4.2 MB

📊 STEP 10: Creating Results Visualization
✓ Visualization saved

🎉 END-TO-END DEMO COMPLETE!
```

---

**Ready to run! Open your terminal and copy-paste the commands above.** 🚀


