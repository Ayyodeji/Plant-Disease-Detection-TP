# 🌿 Plant Disease Detection System

A comprehensive, production-ready end-to-end system for detecting plant diseases using classical machine learning and deep learning approaches. Built with simplicity, portability, and local relevance in mind.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Design Philosophy](#system-design-philosophy)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This system implements a complete pipeline for plant disease detection, from data acquisition to model deployment. It uses the PlantVillage dataset and implements both classical machine learning (Random Forest, SVM, Gradient Boosting) and deep learning approaches (MobileNet, ResNet, EfficientNet).

### Key Capabilities

- **Automated Data Pipeline**: Downloads, organizes, and preprocesses PlantVillage dataset
- **Dual Approach**: Both handcrafted features (color, texture, shape) and deep learning features
- **Multiple Models**: 6+ trained models (3 classical ML + 3 deep learning)
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1, confusion matrices, ROC curves
- **Production-Ready**: TFLite and ONNX export for edge device deployment
- **Offline Capable**: Designed for use in low-connectivity environments

## ✨ Features

### Data Processing
- ✅ Automated dataset download from Kaggle
- ✅ Stratified train/validation/test splits
- ✅ Image preprocessing (resize, normalize, color standardization)
- ✅ Advanced data augmentation (rotation, flip, color jitter, noise, blur, etc.)
- ✅ Optional leaf segmentation for field images

### Feature Extraction
- ✅ **Manual Features**: Color histograms, color moments, Haralick texture, LBP, Gabor filters, shape features
- ✅ **Deep Features**: Pre-trained CNN feature extraction
- ✅ Feature scaling and normalization

### Models
- ✅ **Classical ML**: Random Forest, SVM, Gradient Boosting
- ✅ **Deep Learning**: MobileNetV2, ResNet50, EfficientNetB0
- ✅ Transfer learning with fine-tuning
- ✅ Hyperparameter optimization

### Evaluation
- ✅ Comprehensive metrics (accuracy, precision, recall, F1)
- ✅ Per-class performance analysis
- ✅ Confusion matrices and ROC curves
- ✅ Model comparison visualizations
- ✅ Cross-validation

### Deployment
- ✅ TensorFlow Lite conversion with quantization
- ✅ ONNX export
- ✅ Inference pipeline for production use
- ✅ Mobile-optimized models (<50MB)

## 🎨 System Design Philosophy

This system follows three core principles:

1. **Simplicity**: Clean, modular code with clear separation of concerns
2. **Portability**: Lightweight models suitable for edge devices and mobile phones
3. **Local Relevance**: Designed for use by farmers and agricultural workers in resource-constrained environments

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA for faster training

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd plant_disease_detection
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Kaggle API (for dataset download)

```bash
# Create Kaggle API directory
mkdir -p ~/.kaggle

# Copy your kaggle.json to the directory
cp /path/to/your/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

To get your `kaggle.json`:
1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"

## ⚡ Quick Start

### Option 1: Run Complete Pipeline

Train all models and evaluate them:

```bash
python main.py --config config.yaml --step all
```

This will:
1. Download PlantVillage dataset (~1.5GB)
2. Preprocess all images
3. Extract manual features
4. Train 3 classical ML models
5. Train 3 deep learning models
6. Evaluate all models
7. Convert models for deployment

**Expected runtime**: 2-8 hours (depending on hardware)

### Option 2: Run Individual Steps

```bash
# Step 1: Data preparation only
python main.py --step data

# Step 2: Preprocessing only
python main.py --step preprocess

# Step 3: Feature extraction only
python main.py --step features

# Step 4: Train classical ML only
python main.py --step classical

# Step 5: Train deep learning only
python main.py --step dl

# Step 6: Evaluation only
python main.py --step eval

# Step 7: Deployment conversion only
python main.py --step deploy
```

### Option 3: Quick Inference Demo

Make predictions on a single image:

```bash
python inference_demo.py \
    --image /path/to/plant/image.jpg \
    --model-path models/deep_learning/mobilenet_v2_final.h5 \
    --model-type keras \
    --top-k 3
```

## 📁 Project Structure

```
plant_disease_detection/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                    # Main pipeline execution
├── inference_demo.py          # Inference demonstration
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading and organization
│   ├── preprocessing.py       # Image preprocessing
│   ├── augmentation.py        # Data augmentation
│   ├── feature_extraction.py  # Manual and CNN features
│   ├── classical_ml_trainer.py # Classical ML training
│   ├── deep_learning_trainer.py # Deep learning training
│   ├── evaluator.py           # Model evaluation
│   └── deployment.py          # Model conversion and inference
│
├── data/                      # Data directory (created automatically)
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Preprocessed data and splits
│
├── models/                    # Trained models (created automatically)
│   ├── classical_ml/          # Classical ML models
│   ├── deep_learning/         # Deep learning models
│   └── deployment/            # Deployment-ready models
│
├── results/                   # Evaluation results (created automatically)
│   ├── model_comparison.csv
│   └── *_evaluation.json
│
├── visualizations/            # Generated visualizations (created automatically)
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── training_curves/
│
└── docs/                      # Documentation
    ├── USER_GUIDE.md
    ├── ETHICS.md
    └── API_REFERENCE.md
```

## 📖 Usage Guide

### Configuration

All settings are in `config.yaml`. Key parameters:

```yaml
data:
  train_split: 0.7      # Training data percentage
  val_split: 0.15       # Validation data percentage
  test_split: 0.15      # Test data percentage

preprocessing:
  target_size: [224, 224]  # Image dimensions

deep_learning:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
```

### Custom Dataset

To use your own dataset:

1. Organize images in folders by class:
```
your_dataset/
├── healthy/
│   ├── img1.jpg
│   └── img2.jpg
├── disease_1/
│   ├── img1.jpg
│   └── img2.jpg
└── disease_2/
    └── ...
```

2. Modify `config.yaml`:
```yaml
data:
  custom_dataset_path: "/path/to/your_dataset"
```

3. Update data loader to use custom path

### Training Your Own Model

```python
from src.deep_learning_trainer import DeepLearningTrainer
import numpy as np

# Load your data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

# Initialize trainer
trainer = DeepLearningTrainer('config.yaml')

# Train model
results = trainer.train_model(
    'mobilenet_v2',
    X_train, y_train,
    X_val, y_val,
    num_classes=38
)
```

### Making Predictions

```python
from src.deployment import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline(
    model_path='models/deep_learning/mobilenet_v2_final.h5',
    class_mapping_path='data/processed/class_mapping.json',
    model_type='keras'
)

# Predict
result = pipeline.predict('path/to/image.jpg', top_k=3)

print(f"Disease: {result['top_prediction']['class']}")
print(f"Confidence: {result['top_prediction']['confidence_percentage']:.2f}%")
```

## 📊 Model Performance

Expected performance on PlantVillage dataset:

| Model | Accuracy | Precision | Recall | F1-Score | Size |
|-------|----------|-----------|--------|----------|------|
| **Deep Learning** |
| MobileNetV2 | ~98% | ~98% | ~98% | ~98% | 14MB |
| ResNet50 | ~99% | ~99% | ~99% | ~99% | 98MB |
| EfficientNetB0 | ~98% | ~98% | ~98% | ~98% | 29MB |
| **Classical ML** |
| Random Forest | ~85% | ~85% | ~85% | ~85% | 50MB |
| SVM | ~82% | ~82% | ~82% | ~82% | 100MB |
| Gradient Boosting | ~83% | ~83% | ~83% | ~83% | 30MB |

*Note: Performance may vary based on dataset and training conditions*

## 📱 Deployment

### Mobile Deployment (TensorFlow Lite)

```python
from src.deployment import ModelConverter

converter = ModelConverter('config.yaml')

# Convert to TFLite with quantization
tflite_path = converter.convert_to_tflite(
    model, 
    'mobilenet_v2',
    quantize=True
)

# Result: mobilenet_v2_quantized.tflite (~4MB)
```

### Web/Server Deployment (ONNX)

```python
# Convert to ONNX
onnx_path = converter.convert_to_onnx(
    'models/deep_learning/mobilenet_v2_final.h5',
    'mobilenet_v2'
)

# Use with ONNX Runtime for faster inference
```

### Integration Examples

**Python API**:
```python
from src.deployment import InferencePipeline

pipeline = InferencePipeline(...)
result = pipeline.predict('image.jpg')
```

**Command Line**:
```bash
python inference_demo.py --image image.jpg --model-path model.h5
```

## 🤝 Ethical Considerations

This system is designed with the following ethical principles:

### Data Ethics
- ✅ Respects PlantVillage dataset licensing
- ✅ Provides guidelines for obtaining informed consent for field data
- ✅ Anonymizes location and personal information
- ✅ Ensures fair representation across crops and regions

### Model Ethics
- ✅ Transparent limitations and disclaimers
- ✅ Avoids overfitting through regularization and validation
- ✅ Reports honest performance metrics
- ✅ Tests on diverse data when available

### Deployment Ethics
- ✅ Offline-capable for low-connectivity areas
- ✅ Lightweight models for resource-constrained devices
- ✅ Multilingual support (extensible)
- ✅ Collaborates with local agricultural experts

**See [ETHICS.md](docs/ETHICS.md) for complete ethical guidelines.**

## 🐛 Troubleshooting

### Common Issues

**Issue**: Kaggle API authentication error
```
Solution: Ensure kaggle.json is in ~/.kaggle/ with correct permissions (600)
```

**Issue**: Out of memory during training
```
Solution: Reduce batch_size in config.yaml (try 16 or 8)
```

**Issue**: CUDA out of memory
```
Solution: Use smaller model (MobileNetV2) or reduce image size
```

**Issue**: Import errors
```
Solution: Ensure all dependencies installed: pip install -r requirements.txt
```

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Detailed usage instructions
- **[Ethics Documentation](docs/ETHICS.md)**: Ethical considerations and guidelines
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation

## 🔬 Research & References

This system implements techniques from:

- PlantVillage Dataset: Hughes & Salathé (2015)
- Transfer Learning: Pan & Yang (2010)
- Data Augmentation: Shorten & Khoshgoftaar (2019)
- Mobile Models: Howard et al. (2017) - MobileNets

## 🙏 Acknowledgments

- PlantVillage for the comprehensive plant disease dataset
- TensorFlow and scikit-learn communities
- Agricultural extension officers who provided valuable feedback

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

**Dataset License**: PlantVillage dataset is licensed under CC BY 4.0

## 📞 Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: [your-email]
- Documentation: [docs/](docs/)

## 🚧 Roadmap

Future enhancements:
- [ ] Multi-language UI support
- [ ] Real-time video processing
- [ ] Treatment recommendations
- [ ] Integration with agricultural databases
- [ ] Progressive web app (PWA)
- [ ] Explainable AI visualizations (Grad-CAM)

---

**Built with ❤️ for farmers and agricultural workers worldwide**


