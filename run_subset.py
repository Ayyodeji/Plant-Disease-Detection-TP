"""
Run pipeline on a small subset to avoid memory issues.
"""

import sys
import os
from pathlib import Path

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 80)
print("PLANT DISEASE DETECTION - SUBSET PIPELINE")
print("=" * 80)
print("\nProcessing 5 classes with 200 images each to avoid memory issues\n")

# Step 1: Download dataset
print("STEP 1: Downloading PlantVillage dataset...")
print("-" * 80)

from data_loader import DatasetLoader
import json

loader = DatasetLoader('config.yaml')

# Check if data already exists
raw_dir = Path('data/raw')
raw_dir.mkdir(parents=True, exist_ok=True)

# Check if dataset is already downloaded
has_images = False
for root, dirs, files in os.walk(raw_dir):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            has_images = True
            break
    if has_images:
        break

if not has_images:
    print("Downloading dataset from Kaggle...")
    try:
        downloaded_path = loader.download_dataset()
        print(f"Downloaded to: {downloaded_path}")
        dataset_path = Path(downloaded_path)
    except Exception as e:
        print(f"\nERROR: Failed to download dataset: {e}")
        print("\nPlease ensure:")
        print("1. Kaggle API is configured (~/.kaggle/kaggle.json)")
        print("2. You have internet connection")
        print("\nGet kaggle.json from: https://www.kaggle.com/account")
        sys.exit(1)
else:
    print("Dataset already downloaded")
    # Find where the images are in raw_dir
    dataset_path = None
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                dataset_path = Path(root).parent
                break
        if dataset_path:
            break

# Step 2: Find and organize subset
print("\nSTEP 2: Organizing subset of data...")
print("-" * 80)

if not dataset_path:
    print("ERROR: Could not find dataset path")
    sys.exit(1)

print(f"Found dataset at: {dataset_path}")

# Manually organize with strict limits
dataset_dict = {}
image_extensions = {'.jpg', '.jpeg', '.png'}
MAX_CLASSES = 5
MAX_IMAGES = 200

for root, dirs, files in os.walk(dataset_path):
    if len(dataset_dict) >= MAX_CLASSES:
        break
    
    for file in files[:MAX_IMAGES]:  # Limit files per directory
        if Path(file).suffix.lower() in image_extensions:
            file_path = os.path.join(root, file)
            class_name = Path(root).name
            
            if class_name not in dataset_dict:
                if len(dataset_dict) >= MAX_CLASSES:
                    break
                dataset_dict[class_name] = []
            
            if len(dataset_dict[class_name]) < MAX_IMAGES:
                dataset_dict[class_name].append(file_path)

print(f"\nSelected {len(dataset_dict)} classes:")
for name, paths in dataset_dict.items():
    print(f"  {name}: {len(paths)} images")

# Step 3: Create splits
print("\nSTEP 3: Creating train/val/test splits...")
print("-" * 80)

from sklearn.model_selection import train_test_split

train_dict = {}
val_dict = {}
test_dict = {}

for class_name, image_paths in dataset_dict.items():
    train_paths, temp_paths = train_test_split(image_paths, test_size=0.3, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)
    
    train_dict[class_name] = train_paths
    val_dict[class_name] = val_paths
    test_dict[class_name] = test_paths

# Save splits
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

split_info = {
    'train': train_dict,
    'val': val_dict,
    'test': test_dict
}

with open(processed_dir / 'split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

# Save class mapping
class_to_idx = {name: idx for idx, name in enumerate(sorted(dataset_dict.keys()))}
with open(processed_dir / 'class_mapping.json', 'w') as f:
    json.dump(class_to_idx, f, indent=2)

print(f"Splits saved to {processed_dir}/split_info.json")
print(f"Class mapping saved to {processed_dir}/class_mapping.json")

# Step 4: Preprocess
print("\nSTEP 4: Preprocessing images...")
print("-" * 80)

from preprocessing import ImagePreprocessor
import numpy as np
from tqdm import tqdm

preprocessor = ImagePreprocessor('config.yaml')

def process_split(split_dict, split_name):
    images = []
    labels = []
    
    print(f"\nProcessing {split_name}...")
    for class_name, image_paths in split_dict.items():
        class_idx = class_to_idx[class_name]
        for img_path in tqdm(image_paths, desc=class_name):
            try:
                img = preprocessor.preprocess(img_path)
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error: {e}")
    
    return np.array(images), np.array(labels)

X_train, y_train = process_split(train_dict, 'train')
X_val, y_val = process_split(val_dict, 'val')
X_test, y_test = process_split(test_dict, 'test')

# Save preprocessed data
np.save(processed_dir / 'X_train.npy', X_train)
np.save(processed_dir / 'y_train.npy', y_train)
np.save(processed_dir / 'X_val.npy', X_val)
np.save(processed_dir / 'y_val.npy', y_val)
np.save(processed_dir / 'X_test.npy', X_test)
np.save(processed_dir / 'y_test.npy', y_test)

print(f"\nSaved preprocessed data:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# Step 5: Extract features
print("\nSTEP 5: Extracting features...")
print("-" * 80)

from feature_extraction import ManualFeatureExtractor

extractor = ManualFeatureExtractor('config.yaml')

X_train_denorm = [preprocessor.denormalize_image(img) for img in X_train]
X_val_denorm = [preprocessor.denormalize_image(img) for img in X_val]
X_test_denorm = [preprocessor.denormalize_image(img) for img in X_test]

features_train = extractor.extract_batch_features(X_train_denorm, show_progress=True)
features_val = extractor.extract_batch_features(X_val_denorm, show_progress=False)
features_test = extractor.extract_batch_features(X_test_denorm, show_progress=False)

np.save(processed_dir / 'features_train.npy', features_train)
np.save(processed_dir / 'features_val.npy', features_val)
np.save(processed_dir / 'features_test.npy', features_test)

print(f"Saved features: {features_train.shape[1]} dimensions")

# Step 6: Train classical ML
print("\nSTEP 6: Training classical ML models...")
print("-" * 80)

from classical_ml_trainer import ClassicalMLTrainer

trainer = ClassicalMLTrainer('config.yaml')
features_train_scaled, features_val_scaled, features_test_scaled = trainer.prepare_data(
    features_train, features_val, features_test
)

results = trainer.train_all_models(
    features_train_scaled, y_train,
    features_val_scaled, y_val,
    use_cross_validation=True
)

print("\nTraining complete!")

# Step 7: Evaluate
print("\nSTEP 7: Evaluating models...")
print("-" * 80)

from evaluator import ModelEvaluator

evaluator = ModelEvaluator('config.yaml')
class_names = sorted(class_to_idx.keys())
eval_results = {}

for model_name in trainer.models.keys():
    model = trainer.models[model_name]
    y_pred = model.predict(features_test_scaled)
    y_pred_proba = model.predict_proba(features_test_scaled)
    
    result = evaluator.evaluate_model(
        y_test, y_pred, y_pred_proba,
        class_names, model_name
    )
    eval_results[model_name] = result

evaluator.compare_models(eval_results)

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print("\nResults:")
print(f"  Models trained: {len(trainer.models)}")
print(f"  Best accuracy:  {max(r['accuracy'] for r in eval_results.values()):.1%}")
print(f"\nGenerated files:")
print(f"  models/classical_ml/*.pkl")
print(f"  results/*_evaluation.json")
print(f"  visualizations/*_confusion_matrix.png")
print("\n")

