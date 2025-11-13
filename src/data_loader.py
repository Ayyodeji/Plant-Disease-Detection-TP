"""
Data Loading and Dataset Management Module

This module handles downloading, loading, and organizing the PlantVillage dataset.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Dict, List
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import kagglehub


class DatasetLoader:
    """Handles dataset downloading and organization."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DatasetLoader.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.raw_data_dir = Path(self.data_config['raw_data_dir'])
        self.processed_data_dir = Path(self.data_config['processed_data_dir'])
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> str:
        """
        Download PlantVillage dataset from Kaggle.
        
        Returns:
            Path to downloaded dataset
        """
        print("Downloading PlantVillage dataset from Kaggle...")
        path = kagglehub.dataset_download(self.data_config['dataset_name'])
        print(f"Dataset downloaded to: {path}")
        return path
    
    def organize_dataset(self, source_path: str) -> Dict[str, List[str]]:
        """
        Organize dataset into structured format.
        
        Args:
            source_path: Path to downloaded dataset
            
        Returns:
            Dictionary mapping class names to image paths
        """
        print("Organizing dataset...")
        source = Path(source_path)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        dataset_dict = {}
        
        # Walk through directory structure
        for root, dirs, files in os.walk(source):
            for file in files:
                if Path(file).suffix in image_extensions:
                    file_path = os.path.join(root, file)
                    
                    # Extract class name from directory structure
                    rel_path = os.path.relpath(root, source)
                    class_name = rel_path.replace(os.sep, '_')
                    
                    # Handle case where images are in subdirectories
                    if class_name == '.':
                        class_name = Path(root).name
                    
                    if class_name not in dataset_dict:
                        dataset_dict[class_name] = []
                    dataset_dict[class_name].append(file_path)
        
        # Filter out classes with very few samples (< 10)
        dataset_dict = {k: v for k, v in dataset_dict.items() if len(v) >= 10}
        
        # Apply max_classes limit if specified
        if 'max_classes' in self.data_config and self.data_config['max_classes']:
            class_names = sorted(dataset_dict.keys())[:self.data_config['max_classes']]
            dataset_dict = {k: v for k, v in dataset_dict.items() if k in class_names}
        
        # Apply max_images_per_class limit if specified
        if 'max_images_per_class' in self.data_config and self.data_config['max_images_per_class']:
            max_imgs = self.data_config['max_images_per_class']
            dataset_dict = {k: v[:max_imgs] for k, v in dataset_dict.items()}
        
        print(f"Found {len(dataset_dict)} classes")
        for class_name, images in dataset_dict.items():
            print(f"  {class_name}: {len(images)} images")
        
        return dataset_dict
    
    def create_splits(
        self, 
        dataset_dict: Dict[str, List[str]]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Create stratified train/val/test splits.
        
        Args:
            dataset_dict: Dictionary mapping class names to image paths
            
        Returns:
            Tuple of (train_dict, val_dict, test_dict)
        """
        print("\nCreating stratified data splits...")
        
        train_split = self.data_config['train_split']
        val_split = self.data_config['val_split']
        test_split = self.data_config['test_split']
        seed = self.data_config['random_seed']
        
        train_dict = {}
        val_dict = {}
        test_dict = {}
        
        for class_name, image_paths in dataset_dict.items():
            # First split: separate test set
            train_val_paths, test_paths = train_test_split(
                image_paths,
                test_size=test_split,
                random_state=seed,
                shuffle=True
            )
            
            # Second split: separate train and validation
            val_size_adjusted = val_split / (train_split + val_split)
            train_paths, val_paths = train_test_split(
                train_val_paths,
                test_size=val_size_adjusted,
                random_state=seed,
                shuffle=True
            )
            
            train_dict[class_name] = train_paths
            val_dict[class_name] = val_paths
            test_dict[class_name] = test_paths
        
        # Print split statistics
        total_train = sum(len(v) for v in train_dict.values())
        total_val = sum(len(v) for v in val_dict.values())
        total_test = sum(len(v) for v in test_dict.values())
        total = total_train + total_val + total_test
        
        print(f"\nDataset split:")
        print(f"  Training:   {total_train} images ({total_train/total*100:.1f}%)")
        print(f"  Validation: {total_val} images ({total_val/total*100:.1f}%)")
        print(f"  Test:       {total_test} images ({total_test/total*100:.1f}%)")
        
        return train_dict, val_dict, test_dict
    
    def save_split_info(
        self, 
        train_dict: Dict, 
        val_dict: Dict, 
        test_dict: Dict
    ) -> None:
        """
        Save split information for reproducibility.
        
        Args:
            train_dict: Training split dictionary
            val_dict: Validation split dictionary
            test_dict: Test split dictionary
        """
        import json
        
        split_info = {
            'train': train_dict,
            'val': val_dict,
            'test': test_dict
        }
        
        output_path = self.processed_data_dir / 'split_info.json'
        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\nSplit information saved to: {output_path}")
    
    def load_split_info(self) -> Tuple[Dict, Dict, Dict]:
        """
        Load previously saved split information.
        
        Returns:
            Tuple of (train_dict, val_dict, test_dict)
        """
        import json
        
        split_path = self.processed_data_dir / 'split_info.json'
        
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split info not found at {split_path}. "
                "Please run data preparation first."
            )
        
        with open(split_path, 'r') as f:
            split_info = json.load(f)
        
        return (
            split_info['train'],
            split_info['val'],
            split_info['test']
        )
    
    def get_class_mapping(self, dataset_dict: Dict) -> Dict[str, int]:
        """
        Create class name to index mapping.
        
        Args:
            dataset_dict: Dictionary with class names as keys
            
        Returns:
            Dictionary mapping class names to integer indices
        """
        classes = sorted(dataset_dict.keys())
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Save class mapping
        import json
        mapping_path = self.processed_data_dir / 'class_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(class_to_idx, f, indent=2)
        
        print(f"\nClass mapping saved to: {mapping_path}")
        print(f"Number of classes: {len(class_to_idx)}")
        
        return class_to_idx

