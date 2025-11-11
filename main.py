"""
Main Execution Pipeline

End-to-end plant disease detection system.
Orchestrates data loading, preprocessing, feature extraction, training, and evaluation.
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DatasetLoader
from preprocessing import ImagePreprocessor
from augmentation import DataAugmentor
from feature_extraction import ManualFeatureExtractor, CNNFeatureExtractor
from classical_ml_trainer import ClassicalMLTrainer
from deep_learning_trainer import DeepLearningTrainer
from evaluator import ModelEvaluator
from deployment import ModelConverter, InferencePipeline


class PlantDiseaseDetectionPipeline:
    """Main pipeline for plant disease detection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 80)
        print("PLANT DISEASE DETECTION SYSTEM")
        print("=" * 80)
        print(f"\nConfiguration loaded from: {config_path}")
        
        # Initialize components
        self.data_loader = DatasetLoader(config_path)
        self.preprocessor = ImagePreprocessor(config_path)
        self.augmentor = DataAugmentor(config_path)
        self.evaluator = ModelEvaluator(config_path)
    
    def step1_prepare_data(self):
        """Step 1: Download and prepare dataset."""
        print("\n" + "=" * 80)
        print("STEP 1: DATA PREPARATION")
        print("=" * 80)
        
        # Download dataset
        dataset_path = self.data_loader.download_dataset()
        
        # Organize dataset
        dataset_dict = self.data_loader.organize_dataset(dataset_path)
        
        # Create splits
        train_dict, val_dict, test_dict = self.data_loader.create_splits(dataset_dict)
        
        # Save split information
        self.data_loader.save_split_info(train_dict, val_dict, test_dict)
        
        # Create and save class mapping
        self.class_mapping = self.data_loader.get_class_mapping(dataset_dict)
        
        print("\n✓ Data preparation complete")
        return train_dict, val_dict, test_dict
    
    def step2_preprocess_images(self, train_dict, val_dict, test_dict):
        """Step 2: Preprocess all images."""
        print("\n" + "=" * 80)
        print("STEP 2: IMAGE PREPROCESSING")
        print("=" * 80)
        
        # Load class mapping
        processed_dir = Path(self.config['data']['processed_data_dir'])
        with open(processed_dir / 'class_mapping.json', 'r') as f:
            class_to_idx = json.load(f)
        
        def preprocess_split(split_dict, split_name):
            """Preprocess a data split."""
            print(f"\nPreprocessing {split_name} set...")
            
            images = []
            labels = []
            
            for class_name, image_paths in split_dict.items():
                class_idx = class_to_idx[class_name]
                
                for img_path in tqdm(image_paths, desc=class_name):
                    try:
                        img = self.preprocessor.preprocess(img_path)
                        images.append(img)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
            
            return np.array(images), np.array(labels)
        
        # Preprocess all splits
        X_train, y_train = preprocess_split(train_dict, "training")
        X_val, y_val = preprocess_split(val_dict, "validation")
        X_test, y_test = preprocess_split(test_dict, "test")
        
        # Save preprocessed data
        np.save(processed_dir / 'X_train.npy', X_train)
        np.save(processed_dir / 'y_train.npy', y_train)
        np.save(processed_dir / 'X_val.npy', X_val)
        np.save(processed_dir / 'y_val.npy', y_val)
        np.save(processed_dir / 'X_test.npy', X_test)
        np.save(processed_dir / 'y_test.npy', y_test)
        
        print(f"\n✓ Preprocessing complete")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def step3_extract_manual_features(self, X_train, X_val, X_test):
        """Step 3: Extract manual features for classical ML."""
        print("\n" + "=" * 80)
        print("STEP 3: MANUAL FEATURE EXTRACTION")
        print("=" * 80)
        
        extractor = ManualFeatureExtractor(self.config_path)
        
        # Denormalize images for feature extraction
        X_train_denorm = [self.preprocessor.denormalize_image(img) for img in X_train]
        X_val_denorm = [self.preprocessor.denormalize_image(img) for img in X_val]
        X_test_denorm = [self.preprocessor.denormalize_image(img) for img in X_test]
        
        # Extract features
        print("\nExtracting training features...")
        features_train = extractor.extract_batch_features(X_train_denorm)
        
        print("\nExtracting validation features...")
        features_val = extractor.extract_batch_features(X_val_denorm)
        
        print("\nExtracting test features...")
        features_test = extractor.extract_batch_features(X_test_denorm)
        
        # Save features
        processed_dir = Path(self.config['data']['processed_data_dir'])
        np.save(processed_dir / 'features_train.npy', features_train)
        np.save(processed_dir / 'features_val.npy', features_val)
        np.save(processed_dir / 'features_test.npy', features_test)
        
        print(f"\n✓ Feature extraction complete")
        print(f"  Feature dimension: {features_train.shape[1]}")
        
        return features_train, features_val, features_test
    
    def step4_train_classical_ml(self, features_train, y_train, features_val, y_val):
        """Step 4: Train classical ML models."""
        print("\n" + "=" * 80)
        print("STEP 4: CLASSICAL MACHINE LEARNING TRAINING")
        print("=" * 80)
        
        trainer = ClassicalMLTrainer(self.config_path)
        
        # Scale features
        features_train_scaled, features_val_scaled, _ = trainer.prepare_data(
            features_train, features_val, features_val  # Using val for test temporarily
        )
        
        # Train all models
        results = trainer.train_all_models(
            features_train_scaled,
            y_train,
            features_val_scaled,
            y_val,
            use_cross_validation=True
        )
        
        print("\n✓ Classical ML training complete")
        return trainer, results
    
    def step5_train_deep_learning(self, X_train, y_train, X_val, y_val, num_classes):
        """Step 5: Train deep learning models."""
        print("\n" + "=" * 80)
        print("STEP 5: DEEP LEARNING TRAINING")
        print("=" * 80)
        
        trainer = DeepLearningTrainer(self.config_path)
        
        # Train all models
        results = trainer.train_all_models(
            X_train, y_train, X_val, y_val, num_classes
        )
        
        print("\n✓ Deep learning training complete")
        return trainer, results
    
    def step6_evaluate_models(
        self,
        classical_trainer,
        dl_trainer,
        features_test,
        X_test,
        y_test,
        class_names
    ):
        """Step 6: Comprehensive model evaluation."""
        print("\n" + "=" * 80)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 80)
        
        all_results = {}
        
        # Evaluate classical ML models
        print("\n--- Classical ML Models ---")
        scaler = classical_trainer.scaler
        features_test_scaled = scaler.transform(features_test)
        
        for model_name in classical_trainer.models.keys():
            print(f"\nEvaluating {model_name}...")
            model = classical_trainer.models[model_name]
            
            y_pred = model.predict(features_test_scaled)
            y_pred_proba = model.predict_proba(features_test_scaled)
            
            results = self.evaluator.evaluate_model(
                y_test, y_pred, y_pred_proba,
                class_names, f"classical_{model_name}"
            )
            all_results[f"classical_{model_name}"] = results
        
        # Evaluate deep learning models
        print("\n--- Deep Learning Models ---")
        for model_name in dl_trainer.models.keys():
            print(f"\nEvaluating {model_name}...")
            model = dl_trainer.models[model_name]
            
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            results = self.evaluator.evaluate_model(
                y_test, y_pred, y_pred_proba,
                class_names, f"dl_{model_name}"
            )
            all_results[f"dl_{model_name}"] = results
        
        # Compare all models
        self.evaluator.compare_models(all_results)
        
        print("\n✓ Model evaluation complete")
        return all_results
    
    def step7_convert_for_deployment(self, dl_trainer):
        """Step 7: Convert models for deployment."""
        print("\n" + "=" * 80)
        print("STEP 7: MODEL DEPLOYMENT PREPARATION")
        print("=" * 80)
        
        converter = ModelConverter(self.config_path)
        
        for model_name in dl_trainer.models.keys():
            model = dl_trainer.models[model_name]
            
            # Convert to TFLite
            if 'tflite' in self.config['deployment']['model_format']:
                print(f"\nConverting {model_name} to TFLite...")
                converter.convert_to_tflite(model, model_name, quantize=True)
            
            # Convert to ONNX (if tf2onnx is available)
            if 'onnx' in self.config['deployment']['model_format']:
                model_path = Path(self.config['output']['models_dir']) / 'deep_learning' / f'{model_name}_final.h5'
                if model_path.exists():
                    converter.convert_to_onnx(str(model_path), model_name)
        
        print("\n✓ Deployment conversion complete")
    
    def run_full_pipeline(self, skip_deep_learning=True):
        """Run the complete pipeline."""
        print("\n" + "=" * 80)
        print("STARTING FULL PIPELINE")
        if skip_deep_learning:
            print("(Skipping deep learning due to TensorFlow compatibility issues)")
        print("=" * 80)
        
        # Check if data is already prepared
        processed_dir = Path(self.config['data']['processed_data_dir'])
        split_info_path = processed_dir / 'split_info.json'
        
        if split_info_path.exists():
            print("\nFound existing split information. Loading...")
            train_dict, val_dict, test_dict = self.data_loader.load_split_info()
        else:
            # Step 1: Prepare data
            train_dict, val_dict, test_dict = self.step1_prepare_data()
        
        # Check if preprocessing is already done
        if (processed_dir / 'X_train.npy').exists():
            print("\nFound preprocessed data. Loading...")
            X_train = np.load(processed_dir / 'X_train.npy')
            y_train = np.load(processed_dir / 'y_train.npy')
            X_val = np.load(processed_dir / 'X_val.npy')
            y_val = np.load(processed_dir / 'y_val.npy')
            X_test = np.load(processed_dir / 'X_test.npy')
            y_test = np.load(processed_dir / 'y_test.npy')
        else:
            # Step 2: Preprocess images
            X_train, y_train, X_val, y_val, X_test, y_test = self.step2_preprocess_images(
                train_dict, val_dict, test_dict
            )
        
        # Load class mapping
        with open(processed_dir / 'class_mapping.json', 'r') as f:
            class_to_idx = json.load(f)
        class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
        num_classes = len(class_names)
        
        # Check if features are already extracted
        if (processed_dir / 'features_train.npy').exists():
            print("\nFound extracted features. Loading...")
            features_train = np.load(processed_dir / 'features_train.npy')
            features_val = np.load(processed_dir / 'features_val.npy')
            features_test = np.load(processed_dir / 'features_test.npy')
        else:
            # Step 3: Extract manual features
            features_train, features_val, features_test = self.step3_extract_manual_features(
                X_train, X_val, X_test
            )
        
        # Step 4: Train classical ML models
        classical_trainer, classical_results = self.step4_train_classical_ml(
            features_train, y_train, features_val, y_val
        )
        
        # Step 5: Train deep learning models (optional)
        if not skip_deep_learning:
            dl_trainer, dl_results = self.step5_train_deep_learning(
                X_train, y_train, X_val, y_val, num_classes
            )
        else:
            print("\n" + "=" * 80)
            print("SKIPPING DEEP LEARNING TRAINING")
            print("=" * 80)
            print("Deep learning skipped due to TensorFlow compatibility issues.")
            print("Classical ML models have been trained successfully!")
            dl_trainer = None
        
        # Step 6: Evaluate models
        if dl_trainer:
            # Evaluate both classical and deep learning
            all_results = self.step6_evaluate_models(
                classical_trainer, dl_trainer,
                features_test, X_test, y_test,
                class_names
            )
        else:
            # Evaluate only classical ML
            print("\n" + "=" * 80)
            print("STEP 6: MODEL EVALUATION (Classical ML Only)")
            print("=" * 80)
            
            scaler = classical_trainer.scaler
            features_test_scaled = scaler.transform(features_test)
            
            all_results = {}
            for model_name in classical_trainer.models.keys():
                print(f"\nEvaluating {model_name}...")
                model = classical_trainer.models[model_name]
                
                y_pred = model.predict(features_test_scaled)
                y_pred_proba = model.predict_proba(features_test_scaled)
                
                results = self.evaluator.evaluate_model(
                    y_test, y_pred, y_pred_proba,
                    class_names, f"classical_{model_name}"
                )
                all_results[f"classical_{model_name}"] = results
            
            # Compare classical models
            self.evaluator.compare_models(all_results)
        
        # Step 7: Convert for deployment (skip if no DL models)
        if dl_trainer:
            self.step7_convert_for_deployment(dl_trainer)
        else:
            print("\n" + "=" * 80)
            print("SKIPPING DEPLOYMENT CONVERSION")
            print("=" * 80)
            print("Classical ML models are already saved and ready to use!")
            print(f"Location: {self.config['output']['models_dir']}/classical_ml/")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print("\nAll models trained, evaluated, and ready for deployment.")
        print(f"Results saved in: {self.config['output']['results_dir']}")
        print(f"Models saved in: {self.config['output']['models_dir']}")
        print(f"Visualizations saved in: {self.config['output']['visualizations_dir']}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection System - End-to-End Pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'data', 'preprocess', 'features', 'classical', 'dl', 'eval', 'deploy'],
        default='all',
        help='Which step to run'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PlantDiseaseDetectionPipeline(args.config)
    
    if args.step == 'all':
        pipeline.run_full_pipeline()
    elif args.step == 'data':
        print(f"Running step: {args.step}")
        pipeline.step1_prepare_data()
    elif args.step == 'preprocess':
        print(f"Running step: {args.step}")
        train_dict, val_dict, test_dict = pipeline.data_loader.load_split_info()
        pipeline.step2_preprocess_images(train_dict, val_dict, test_dict)
    elif args.step == 'features':
        print(f"Running step: {args.step}")
        processed_dir = Path(pipeline.config['data']['processed_data_dir'])
        X_train = np.load(processed_dir / 'X_train.npy')
        X_val = np.load(processed_dir / 'X_val.npy')
        X_test = np.load(processed_dir / 'X_test.npy')
        pipeline.step3_extract_manual_features(X_train, X_val, X_test)
    elif args.step == 'classical':
        print(f"Running step: {args.step}")
        processed_dir = Path(pipeline.config['data']['processed_data_dir'])
        features_train = np.load(processed_dir / 'features_train.npy')
        features_val = np.load(processed_dir / 'features_val.npy')
        y_train = np.load(processed_dir / 'y_train.npy')
        y_val = np.load(processed_dir / 'y_val.npy')
        pipeline.step4_train_classical_ml(features_train, y_train, features_val, y_val)
    elif args.step == 'dl':
        print(f"Running step: {args.step}")
        processed_dir = Path(pipeline.config['data']['processed_data_dir'])
        X_train = np.load(processed_dir / 'X_train.npy')
        X_val = np.load(processed_dir / 'X_val.npy')
        y_train = np.load(processed_dir / 'y_train.npy')
        y_val = np.load(processed_dir / 'y_val.npy')
        with open(processed_dir / 'class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        num_classes = len(class_mapping)
        pipeline.step5_train_deep_learning(X_train, y_train, X_val, y_val, num_classes)
    elif args.step == 'eval':
        print(f"Running step: {args.step}")
        processed_dir = Path(pipeline.config['data']['processed_data_dir'])
        features_test = np.load(processed_dir / 'features_test.npy')
        X_test = np.load(processed_dir / 'X_test.npy')
        y_test = np.load(processed_dir / 'y_test.npy')
        with open(processed_dir / 'class_mapping.json', 'r') as f:
            class_to_idx = json.load(f)
        class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
        
        # Load trainers
        from classical_ml_trainer import ClassicalMLTrainer
        from deep_learning_trainer import DeepLearningTrainer
        classical_trainer = ClassicalMLTrainer(args.config)
        dl_trainer = DeepLearningTrainer(args.config)
        
        # Load models
        for model_name in pipeline.config['classical_ml']['models']:
            classical_trainer.load_model(model_name)
        for model_name in pipeline.config['deep_learning']['models']:
            dl_trainer.load_model(model_name)
        
        pipeline.step6_evaluate_models(
            classical_trainer, dl_trainer, features_test, X_test, y_test, class_names
        )
    elif args.step == 'deploy':
        print(f"Running step: {args.step}")
        from deep_learning_trainer import DeepLearningTrainer
        dl_trainer = DeepLearningTrainer(args.config)
        for model_name in pipeline.config['deep_learning']['models']:
            dl_trainer.load_model(model_name)
        pipeline.step7_convert_for_deployment(dl_trainer)
    else:
        print(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()

