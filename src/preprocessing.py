"""
Image Preprocessing Module

Handles image loading, resizing, normalization, color standardization,
and optional leaf segmentation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import yaml
from PIL import Image
from skimage import color, filters, morphology
from tqdm import tqdm


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ImagePreprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocess_config = self.config['preprocessing']
        self.target_size = tuple(self.preprocess_config['target_size'])
        self.normalize = self.preprocess_config['normalize']
        self.mean = np.array(self.preprocess_config['mean'])
        self.std = np.array(self.preprocess_config['std'])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in RGB format
        """
        # Use PIL for better format support
        img = Image.open(image_path)
        img = img.convert('RGB')
        return np.array(img)
    
    def resize_image(
        self, 
        image: np.ndarray, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            target_size: Target (height, width), uses config if None
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        # Use high-quality interpolation
        resized = cv2.resize(
            image, 
            (target_size[1], target_size[0]),  # OpenCV uses (width, height)
            interpolation=cv2.INTER_LANCZOS4
        )
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        Args:
            image: Input image (0-255 range)
            
        Returns:
            Normalized image
        """
        # Convert to float and scale to [0, 1]
        img_normalized = image.astype(np.float32) / 255.0
        
        if self.normalize:
            # Apply mean and std normalization
            img_normalized = (img_normalized - self.mean) / self.std
        
        return img_normalized
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image back to [0, 255] range.
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        if self.normalize:
            img = image * self.std + self.mean
        else:
            img = image
        
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img
    
    def adjust_color_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust color balance for consistent appearance.
        
        Args:
            image: Input image
            
        Returns:
            Color-balanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return balanced
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Use bilateral filter to preserve edges while reducing noise
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised
    
    def segment_leaf(
        self, 
        image: np.ndarray, 
        return_mask: bool = False
    ) -> np.ndarray:
        """
        Segment leaf from background (useful for field images).
        
        Args:
            image: Input image
            return_mask: If True, return binary mask instead of segmented image
            
        Returns:
            Segmented image or binary mask
        """
        # Convert to HSV for better color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define green color range (adjust as needed)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find largest contour (assumed to be the leaf)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        if return_mask:
            return mask
        
        # Apply mask to image
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return segmented
    
    def preprocess(
        self, 
        image_path: str,
        apply_segmentation: bool = False,
        apply_color_balance: bool = False,
        apply_denoising: bool = False
    ) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            image_path: Path to input image
            apply_segmentation: Whether to apply leaf segmentation
            apply_color_balance: Whether to apply color balancing
            apply_denoising: Whether to apply noise reduction
            
        Returns:
            Preprocessed and normalized image
        """
        # Load image
        img = self.load_image(image_path)
        
        # Optional: Segment leaf
        if apply_segmentation:
            img = self.segment_leaf(img)
        
        # Optional: Denoise
        if apply_denoising:
            img = self.reduce_noise(img)
        
        # Optional: Color balance
        if apply_color_balance:
            img = self.adjust_color_balance(img)
        
        # Resize
        img = self.resize_image(img)
        
        # Normalize
        img = self.normalize_image(img)
        
        return img
    
    def preprocess_batch(
        self,
        image_paths: List[str],
        apply_segmentation: bool = False,
        apply_color_balance: bool = False,
        apply_denoising: bool = False,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Preprocess batch of images.
        
        Args:
            image_paths: List of image paths
            apply_segmentation: Whether to apply leaf segmentation
            apply_color_balance: Whether to apply color balancing
            apply_denoising: Whether to apply noise reduction
            show_progress: Whether to show progress bar
            
        Returns:
            Array of preprocessed images
        """
        images = []
        
        iterator = tqdm(image_paths) if show_progress else image_paths
        
        for path in iterator:
            try:
                img = self.preprocess(
                    path,
                    apply_segmentation=apply_segmentation,
                    apply_color_balance=apply_color_balance,
                    apply_denoising=apply_denoising
                )
                images.append(img)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        return np.array(images)


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("Testing preprocessing pipeline...")
    
    preprocessor = ImagePreprocessor()
    
    # Test with a sample image (you'll need to update this path)
    # This is just for demonstration
    print("\nPreprocessing configuration:")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Normalize: {preprocessor.normalize}")
    print(f"  Mean: {preprocessor.mean}")
    print(f"  Std: {preprocessor.std}")
    
    print("\nPreprocessing pipeline ready!")


if __name__ == "__main__":
    test_preprocessing()


