# User Guide - Plant Disease Detection System

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the System](#understanding-the-system)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Getting Started

### What This System Does

The Plant Disease Detection System helps identify diseases in plants by analyzing photographs of leaves. It can:

- Recognize 38+ different plant diseases
- Work offline on mobile devices
- Provide confidence scores for predictions
- Process images in seconds

### Who Should Use This

- **Farmers**: Quick disease identification in the field
- **Agricultural Extension Officers**: Diagnosis assistance
- **Researchers**: Plant disease analysis
- **Students**: Learning about plant pathology and ML

## Understanding the System

### How It Works

```
Your Image → Preprocessing → Feature Extraction → Model → Disease Prediction
```

1. **Preprocessing**: Resizes and normalizes your image
2. **Feature Extraction**: Identifies important patterns (colors, textures, shapes)
3. **Model**: Uses AI to recognize disease patterns
4. **Prediction**: Provides disease name and confidence score

### Model Types

The system includes two approaches:

**Classical Machine Learning**:
- Uses handcrafted features (color, texture, shape)
- Faster training, smaller models
- Good for resource-constrained environments
- ~85% accuracy

**Deep Learning**:
- Learns features automatically
- Higher accuracy (~98%)
- Requires more computation
- Better for complex cases

## Step-by-Step Tutorial

### Tutorial 1: First-Time Setup

**Time**: 15 minutes

```bash
# 1. Install Python (if not already installed)
# Download from python.org (version 3.8+)

# 2. Open terminal/command prompt

# 3. Navigate to project directory
cd plant_disease_detection

# 4. Create virtual environment
python -m venv venv

# 5. Activate environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 6. Install dependencies
pip install -r requirements.txt

# 7. Set up Kaggle API (for dataset)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Tutorial 2: Training Your First Model

**Time**: 2-4 hours (mostly automated)

```bash
# Run the complete pipeline
python main.py --config config.yaml --step all
```

This will:
1. Download PlantVillage dataset (~1.5GB)
2. Preprocess 50,000+ images
3. Train 6 different models
4. Evaluate and compare models
5. Create deployment-ready versions

**What to Expect**:
- Progress bars show current step
- Training curves are saved automatically
- Results appear in `results/` directory
- Models saved in `models/` directory

### Tutorial 3: Making Your First Prediction

**Time**: 2 minutes

```bash
# Use trained model to identify disease
python inference_demo.py \
    --image /path/to/leaf_photo.jpg \
    --model-path models/deep_learning/mobilenet_v2_final.pth \
    --model-type pytorch \
    --top-k 3
```

**Output Example**:
```
Top Prediction:
  Disease: Tomato_Late_blight
  Confidence: 97.3%

Top 3 Predictions:
  1. Tomato_Late_blight: 97.3%
  2. Tomato_Early_blight: 2.1%
  3. Tomato_Leaf_Mold: 0.4%
```

### Tutorial 4: Using on Mobile Device

**Time**: 30 minutes

1. **Convert Model to Mobile Format**:
```python
from src.deployment import ModelConverter

converter = ModelConverter('config.yaml')

# Convert to TorchScript for mobile deployment
torchscript_path = converter.convert_to_torchscript(
    model, 
    'mobilenet_v2',
    optimize=True  # Optimizes for mobile
)

# Or convert to ONNX for cross-platform deployment
onnx_path = converter.convert_to_onnx(
    model,
    'mobilenet_v2'
)
```

2. **Integrate with Mobile App**:
- Use PyTorch Mobile for Android/iOS
- Use ONNX Runtime for cross-platform
- Model size: ~4MB (optimized)
- Inference time: <100ms on modern phones

3. **Example Android Integration with PyTorch Mobile**:
```java
// Load TorchScript model
Module module = LiteModuleLoader.load(assetFilePath(this, "model.pt"));

// Preprocess image (convert to tensor)
Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
    bitmap,
    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
    TensorImageUtils.TORCHVISION_NORM_STD_RGB
);

// Run inference
Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

// Get results
float[] probabilities = outputTensor.getDataAsFloatArray();
```

## Advanced Usage

### Custom Dataset Training

If you have your own plant images:

**Step 1: Organize Your Data**
```
my_dataset/
├── healthy_tomato/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── blight_tomato/
│   └── ...
└── rust_corn/
    └── ...
```

**Step 2: Update Configuration**
```yaml
# config.yaml
data:
  custom_dataset_path: "/path/to/my_dataset"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

**Step 3: Modify Data Loader**
```python
# In src/data_loader.py, update organize_dataset()
# to point to your custom path
```

**Step 4: Train**
```bash
python main.py --step all
```

### Hyperparameter Tuning

For better performance on your specific data:

```python
from src.classical_ml_trainer import ClassicalMLTrainer, get_default_param_grids

trainer = ClassicalMLTrainer('config.yaml')

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [20, 30, 40, 50],
    'min_samples_split': [2, 5, 10]
}

# Tune Random Forest
best_model, best_params = trainer.hyperparameter_tuning(
    'random_forest',
    X_train,
    y_train,
    param_grid
)
```

### Batch Processing

Process multiple images at once:

```python
from src.deployment import InferencePipeline
import glob

# Initialize pipeline
pipeline = InferencePipeline(
    model_path='models/deep_learning/mobilenet_v2_final.pth',
    class_mapping_path='data/processed/class_mapping.json',
    model_type='pytorch'
)

# Get all images in folder
image_paths = glob.glob('/path/to/images/*.jpg')

# Batch predict
results = pipeline.predict_batch(image_paths, batch_size=32)

# Save results
import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Model Comparison

Compare multiple models on your test set:

```python
from src.evaluator import ModelEvaluator
import numpy as np

evaluator = ModelEvaluator('config.yaml')

# Load test data
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Evaluate multiple models
models = [
    'mobilenet_v2',
    'resnet50',
    'efficientnet_b0'
]

results = {}
for model_name in models:
    # Load model and predict
    model = load_model(f'models/deep_learning/{model_name}_final.h5')
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Evaluate
    results[model_name] = evaluator.evaluate_model(
        y_test, y_pred, y_pred_proba,
        class_names, model_name
    )

# Compare
evaluator.compare_models(results)
```

## Troubleshooting

### Problem: Low Accuracy on Custom Dataset

**Symptoms**: Model performs poorly on your images

**Solutions**:
1. **Check data quality**:
   - Ensure images are clear and well-lit
   - Verify labels are correct
   - Remove duplicates and corrupted images

2. **Increase dataset size**:
   - Need minimum 100 images per class
   - Use data augmentation to expand dataset

3. **Adjust preprocessing**:
   ```python
   # In config.yaml, try:
   preprocessing:
     apply_segmentation: true
     apply_color_balance: true
     apply_denoising: true
   ```

4. **Use transfer learning**:
   - Start with pre-trained models
   - Fine-tune on your specific data

### Problem: Model Too Large for Mobile

**Symptoms**: Model file >50MB

**Solutions**:
1. **Use MobileNetV2** (smallest, still accurate)
2. **Apply quantization**:
   ```python
   converter.convert_to_torchscript(model, 'mobilenet_v2', optimize=True)
   ```
3. **Reduce input size** in config.yaml:
   ```yaml
   preprocessing:
     target_size: [160, 160]  # Instead of [224, 224]
   ```

### Problem: Slow Inference

**Symptoms**: >5 seconds per image

**Solutions**:
1. **Use TorchScript instead of full PyTorch model**
2. **Reduce image size**
3. **Use GPU if available**:
   ```python
   import torch
   
   # Check if GPU is available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   # Move model to GPU
   model = model.to(device)
   ```

### Problem: Out of Memory During Training

**Symptoms**: "OOM" or "CUDA out of memory" errors

**Solutions**:
1. **Reduce batch size** in config.yaml:
   ```yaml
   deep_learning:
     batch_size: 8  # Instead of 32
   ```

2. **Use smaller model**:
   ```yaml
   deep_learning:
     models:
       - "mobilenet_v2"  # Remove resnet50
   ```

3. **Process data in chunks**:
   - Don't load entire dataset at once
   - Use data generators

## Best Practices

### For Field Use

1. **Image Quality**:
   - Take photos in good lighting
   - Fill frame with leaf
   - Avoid shadows and glare
   - Use steady hands or tripod

2. **Disease Stage**:
   - Early symptoms may be harder to detect
   - Take multiple photos if unsure
   - Document progression over time

3. **Offline Mode**:
   - Use TorchScript or ONNX models on mobile
   - Pre-download models before field visit
   - Save results locally, sync later

### For Model Training

1. **Data Quality > Quantity**:
   - 100 high-quality images better than 1000 poor ones
   - Ensure accurate labeling
   - Remove ambiguous cases

2. **Validation Strategy**:
   - Always keep separate test set
   - Use cross-validation for small datasets
   - Test on field images if available

3. **Model Selection**:
   - Start with MobileNetV2 (fast, lightweight)
   - Try ResNet50 if accuracy critical
   - Use classical ML for extremely limited resources

### For Production Deployment

1. **Version Control**:
   - Track model versions
   - Document performance metrics
   - Keep training data provenance

2. **Monitoring**:
   - Log predictions and confidence scores
   - Track error cases
   - Update model periodically

3. **User Feedback**:
   - Allow users to report incorrect predictions
   - Use feedback to improve model
   - Maintain human-in-the-loop for critical decisions

## Next Steps

- Read [API Reference](API_REFERENCE.md) for detailed function documentation
- Review [Ethics Documentation](ETHICS.md) for responsible AI practices
- Explore example notebooks in `examples/` directory
- Join community discussions

## Getting Help

If you encounter issues:
1. Check this guide first
2. Review error messages carefully
3. Search existing GitHub issues
4. Open new issue with:
   - Error message
   - Steps to reproduce
   - System information
   - Screenshots if relevant

---

**Happy disease detecting!**



