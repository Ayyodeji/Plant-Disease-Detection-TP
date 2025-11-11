"""
Data Augmentation Module

Implements comprehensive data augmentation techniques for robust model training.
"""

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from typing import Dict, Any


class DataAugmentor:
    """Handles data augmentation for training."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataAugmentor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_config = self.config['augmentation']
        self.preprocess_config = self.config['preprocessing']
        
        # Create augmentation pipelines
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
    
    def _build_train_transform(self) -> A.Compose:
        """
        Build training augmentation pipeline.
        
        Returns:
            Albumentations Compose object
        """
        transforms = [
            # Geometric transformations
            A.Rotate(
                limit=self.aug_config['rotation_range'],
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.HorizontalFlip(p=0.5 if self.aug_config['horizontal_flip'] else 0.0),
            A.VerticalFlip(p=0.3 if self.aug_config['vertical_flip'] else 0.0),
            A.Affine(
                translate_percent={'x': (-self.aug_config['width_shift_range'], self.aug_config['width_shift_range']),
                                  'y': (-self.aug_config['height_shift_range'], self.aug_config['height_shift_range'])},
                scale=(1.0 - self.aug_config['zoom_range'], 1.0 + self.aug_config['zoom_range']),
                rotate=0,  # Already handled by Rotate
                p=0.7
            ),
            
            # Cropping
            A.RandomResizedCrop(
                size=(self.preprocess_config['target_size'][0], self.preprocess_config['target_size'][1]),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),
            
            # Color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            
            # Blur and noise
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            
            A.GaussNoise(
                std_limit=(10.0, 50.0),
                p=0.3
            ),
            
            # Optical distortions
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.3, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            ], p=0.2),
            
            # Occlusion simulation
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 20),
                hole_width_range=(8, 20),
                fill_value=0,
                p=0.3
            ),
            
            # Ensure correct size
            A.Resize(
                height=self.preprocess_config['target_size'][0],
                width=self.preprocess_config['target_size'][1]
            ),
        ]
        
        return A.Compose(transforms)
    
    def _build_val_transform(self) -> A.Compose:
        """
        Build validation/test transformation pipeline (no augmentation).
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            A.Resize(
                height=self.preprocess_config['target_size'][0],
                width=self.preprocess_config['target_size'][1],
                p=1.0
            ),
        ])
    
    def augment(self, image: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Apply augmentation to single image.
        
        Args:
            image: Input image (numpy array)
            is_training: Whether to apply training augmentations
            
        Returns:
            Augmented image
        """
        if is_training:
            augmented = self.train_transform(image=image)
        else:
            augmented = self.val_transform(image=image)
        
        return augmented['image']
    
    def visualize_augmentations(
        self, 
        image: np.ndarray, 
        n_examples: int = 9
    ) -> np.ndarray:
        """
        Create visualization of augmentation effects.
        
        Args:
            image: Input image
            n_examples: Number of augmented examples to generate
            
        Returns:
            Grid of augmented images
        """
        import matplotlib.pyplot as plt
        
        rows = int(np.sqrt(n_examples))
        cols = int(np.ceil(n_examples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx in range(n_examples):
            augmented = self.augment(image, is_training=True)
            axes[idx].imshow(augmented)
            axes[idx].axis('off')
            axes[idx].set_title(f'Augmented {idx + 1}')
        
        plt.tight_layout()
        return fig


class MixupAugmentor:
    """Implements Mixup augmentation."""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize Mixup augmentor.
        
        Args:
            alpha: Mixup interpolation strength
        """
        self.alpha = alpha
    
    def mixup(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple:
        """
        Apply mixup augmentation.
        
        Args:
            images: Batch of images
            labels: Batch of labels (one-hot encoded)
            
        Returns:
            Tuple of mixed images and labels
        """
        batch_size = images.shape[0]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        
        # Ensure lambda is applied correctly
        lam = np.maximum(lam, 1 - lam)
        lam = lam.reshape(batch_size, 1, 1, 1)
        
        # Shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix labels
        lam = lam.reshape(batch_size, 1)
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels


class CutMixAugmentor:
    """Implements CutMix augmentation."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix augmentor.
        
        Args:
            alpha: CutMix interpolation strength
        """
        self.alpha = alpha
    
    def cutmix(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images
            labels: Batch of labels (one-hot encoded)
            
        Returns:
            Tuple of mixed images and labels
        """
        batch_size, h, w, _ = images.shape
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Get random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random position
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Get bounding box
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        mixed_images = images.copy()
        mixed_images[:, y1:y2, x1:x2, :] = images[indices, y1:y2, x1:x2, :]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels


def test_augmentation():
    """Test augmentation pipeline."""
    print("Testing augmentation pipeline...")
    
    augmentor = DataAugmentor()
    
    print("\nAugmentation configuration:")
    print(f"  Rotation range: {augmentor.aug_config['rotation_range']}")
    print(f"  Horizontal flip: {augmentor.aug_config['horizontal_flip']}")
    print(f"  Zoom range: {augmentor.aug_config['zoom_range']}")
    
    print("\nAugmentation pipelines ready!")
    print(f"  Training transforms: {len(augmentor.train_transform.transforms)} operations")
    print(f"  Validation transforms: {len(augmentor.val_transform.transforms)} operations")


if __name__ == "__main__":
    test_augmentation()

