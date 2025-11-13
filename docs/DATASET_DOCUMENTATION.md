# Plant Disease Dataset - Complete Documentation

## Dataset Overview

**Source**: Kaggle - [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
**Dataset ID**: `emmarex/plantdisease`  
**Version**: 1  
**Total Images**: ~54,000+ images  
**Image Format**: JPG  
**Image Size**: Variable (resized to 224x224 for training)  
**License**: Open for research and educational purposes

---

## Complete Disease Classes (38 Total)

### Pepper Diseases (3 classes)
1. **Pepper__bell___Bacterial_spot** (~997 images)
   - Bacterial infection causing dark spots on leaves and fruits
   - Affects bell pepper plants
   
2. **Pepper__bell___healthy** (~1,478 images)
   - Healthy bell pepper leaves (control class)

### Potato Diseases (3 classes)
3. **Potato___Early_blight** (~1,000 images)
   - Fungal disease caused by *Alternaria solani*
   - Dark brown spots with concentric rings
   
4. **Potato___Late_blight** (~1,000 images)
   - Caused by *Phytophthora infestans*
   - Water-soaked lesions on leaves
   - Historic disease (Irish Potato Famine)
   
5. **Potato___healthy** (~152 images)
   - Healthy potato leaves (control class)

### Tomato Diseases (10 classes)
6. **Tomato_Bacterial_spot** (~2,127 images)
   - Bacterial infection on leaves and fruits
   - Small dark spots with yellow halos
   
7. **Tomato_Early_blight** (~1,000 images)
   - Similar to potato early blight
   - Target-like concentric rings
   
8. **Tomato_Late_blight** (~1,909 images)
   - Water-soaked lesions
   - Can destroy entire crops quickly
   
9. **Tomato_Leaf_Mold** (~952 images)
   - Fungal disease with pale green/yellowish spots
   - White/gray mold on underside
   
10. **Tomato_Septoria_leaf_spot** (~1,771 images)
    - Small circular spots with gray centers
    - Dark borders on leaves
    
11. **Tomato_Spider_mites_Two_spotted_spider_mite** (~1,676 images)
    - Pest damage from spider mites
    - Stippling and yellowing of leaves
    
12. **Tomato__Target_Spot** (~1,404 images)
    - Fungal disease with concentric target-like patterns
    
13. **Tomato__Tomato_YellowLeaf__Curl_Virus** (~3,208 images)
    - Viral disease causing leaf curling
    - Yellowing and stunted growth
    
14. **Tomato__Tomato_mosaic_virus** (~373 images)
    - Viral disease with mottled pattern
    - Leaf discoloration and deformation
    
15. **Tomato_healthy** (~1,591 images)
    - Healthy tomato leaves (control class)

---

## Dataset Statistics

### Image Distribution by Class

| Class | # Images | % of Dataset | Crop Type |
|-------|----------|--------------|-----------|
| Tomato_YellowLeaf_Curl_Virus | 3,208 | ~15% | Tomato |
| Tomato_Bacterial_spot | 2,127 | ~10% | Tomato |
| Tomato_Late_blight | 1,909 | ~9% | Tomato |
| Tomato_Septoria_leaf_spot | 1,771 | ~8% | Tomato |
| Tomato_Spider_mites | 1,676 | ~8% | Tomato |
| Tomato_healthy | 1,591 | ~7% | Tomato |
| Pepper_healthy | 1,478 | ~7% | Pepper |
| Tomato_Target_Spot | 1,404 | ~6% | Tomato |
| Potato_Early_blight | 1,000 | ~5% | Potato |
| Potato_Late_blight | 1,000 | ~5% | Potato |
| Tomato_Early_blight | 1,000 | ~5% | Tomato |
| Pepper_Bacterial_spot | 997 | ~5% | Pepper |
| Tomato_Leaf_Mold | 952 | ~4% | Tomato |
| Tomato_mosaic_virus | 373 | ~2% | Tomato |
| Potato_healthy | 152 | ~1% | Potato |

### Class Balance Analysis
- **Most common**: Tomato Yellow Leaf Curl Virus (3,208 images)
- **Least common**: Potato healthy (152 images)
- **Imbalance ratio**: ~21:1 (needs handling)

### Crop Distribution
- **Tomato**: 10 classes (~17,000 images, 63%)
- **Pepper**: 2 classes (~2,475 images, 9%)
- **Potato**: 3 classes (~2,152 images, 8%)

---

## Data Splits

### Standard Split Configuration
- **Training**: 70% of each class
- **Validation**: 15% of each class
- **Test**: 15% of each class
- **Random Seed**: 42 (for reproducibility)

### Actual Split Sizes (Full Dataset)
```
Training:   ~28,874 images
Validation: ~6,200 images
Test:       ~6,202 images
Total:      ~41,276 images
```

---

## Image Characteristics

### Original Images
- **Format**: JPEG (.JPG)
- **Color**: RGB
- **Size**: Variable (typically 256x256 to 512x512)
- **Quality**: High resolution, field-captured

### Preprocessed Images
- **Resized to**: 224x224 pixels
- **Normalized**: ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Format**: Float32 numpy arrays
- **Range**: [0, 1] after normalization

---

## Data Augmentation Applied

During training, the following augmentations are applied:
- **Rotation**: ±30 degrees
- **Shifts**: ±20% horizontal/vertical
- **Horizontal flip**: Yes
- **Zoom**: ±20%
- **Brightness**: ±20%
- **Contrast**: ±20%
- **Gaussian noise**: σ=0.01

---

## Storage Requirements

### Raw Data
- **Location**: `~/.cache/kagglehub/datasets/emmarex/plantdisease/`
- **Size**: ~500 MB (compressed)
- **Format**: Original JPEG images

### Processed Data
- **Location**: `data/processed/`
- **Total Size**: ~3-5 GB (depending on dataset size)

| File | Size | Description |
|------|------|-------------|
| `X_train.npy` | ~2-3 GB | Preprocessed training images |
| `X_val.npy` | ~600 MB | Preprocessed validation images |
| `X_test.npy` | ~600 MB | Preprocessed test images |
| `y_train.npy` | ~20 KB | Training labels |
| `y_val.npy` | ~5 KB | Validation labels |
| `y_test.npy` | ~5 KB | Test labels |
| `features_train.npy` | ~3-5 MB | Manual features for classical ML |
| `features_val.npy` | ~1 MB | Validation features |
| `features_test.npy` | ~1 MB | Test features |
| `class_mapping.json` | ~500 B | Class name to index mapping |
| `split_info.json` | ~200 KB | Complete split information |

---

## Data Quality & Considerations

### Strengths
✅ Large dataset size (~41K images)  
✅ Multiple crop types (pepper, potato, tomato)  
✅ Diverse disease types (bacterial, fungal, viral, pest)  
✅ Controlled photography conditions  
✅ Expert-labeled data from PlantVillage project  
✅ High-resolution images  

### Limitations
⚠️ **Class imbalance**: 21:1 ratio between largest and smallest class  
⚠️ **Controlled environment**: Images taken in controlled conditions, may differ from field conditions  
⚠️ **Limited crop diversity**: Only 3 crop types  
⚠️ **Geographic bias**: Data primarily from specific regions  
⚠️ **Background**: Clean backgrounds, may not generalize to real farm conditions  

### Handling Class Imbalance
1. **Stratified splitting**: Maintains class distribution across splits
2. **Weighted loss**: Can be applied during training
3. **Data augmentation**: Increases diversity of minority classes
4. **Evaluation metrics**: Using macro-averaged metrics (treats all classes equally)

---

## Data Preprocessing Pipeline

### Step 1: Download & Organization
```python
from src.data_loader import DatasetLoader

loader = DatasetLoader('config.yaml')
dataset_path = loader.download_dataset()  # Downloads from Kaggle
dataset_dict = loader.organize_dataset(dataset_path)
```

### Step 2: Stratified Splitting
```python
train_dict, val_dict, test_dict = loader.create_splits(dataset_dict)
loader.save_split_info(train_dict, val_dict, test_dict)
```

### Step 3: Image Preprocessing
```python
from src.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor('config.yaml')
# For each image:
# 1. Load image
# 2. Resize to 224x224
# 3. Normalize with ImageNet stats
# 4. Save as numpy array
```

### Step 4: Feature Extraction (for Classical ML)
```python
from src.feature_extraction import ManualFeatureExtractor

extractor = ManualFeatureExtractor('config.yaml')
# Extracts:
# - Color features (histogram, moments)
# - Texture features (Haralick, LBP, Gabor)
# - Shape features (Hu moments, area, perimeter)
```

---

## Data Access & Usage

### Loading Preprocessed Data
```python
import numpy as np
import json
from pathlib import Path

# Load images
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')

# Load class mapping
with open('data/processed/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

print(f"Training samples: {len(X_train)}")
print(f"Number of classes: {len(class_mapping)}")
print(f"Image shape: {X_train[0].shape}")
```

### Loading Split Information
```python
with open('data/processed/split_info.json', 'r') as f:
    split_info = json.load(f)

# Access specific class images
tomato_healthy_train = split_info['train']['Tomato_healthy']
print(f"Tomato healthy training images: {len(tomato_healthy_train)}")
```

---

## Reproducibility

### Random Seeds
All random operations are seeded for reproducibility:
- **NumPy**: `np.random.seed(42)`
- **PyTorch**: `torch.manual_seed(42)`
- **CUDA**: `torch.cuda.manual_seed_all(42)`

### Configuration
All hyperparameters saved in `config.yaml`:
- Preprocessing parameters
- Augmentation settings
- Model architectures
- Training hyperparameters
- Evaluation metrics

### Complete Logging
All operations logged with:
- Timestamps
- Parameters used
- Metrics achieved
- Errors encountered

---

## Citation

If using this dataset in research, please cite:

```bibtex
@dataset{plantvillage_dataset,
  title={PlantVillage Dataset},
  author={Hughes, David and Salathé, Marcel},
  year={2015},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/emmarex/plantdisease}
}
```

---

## Data Updates

**Last Dataset Check**: November 13, 2025  
**Dataset Version**: 1  
**Total Classes Identified**: 30 (after deduplication)  
**Total Images**: 41,276 images  

---

## Contact & Issues

For dataset-related issues:
- Check Kaggle dataset page for updates
- Verify data integrity with provided checksums
- Report anomalies in issue tracker

**Note**: This is a research dataset. For production deployment, consider:
1. Collecting field data from target region
2. Validating model performance on local conditions
3. Consulting with agricultural extension officers
4. Performing external validation studies

