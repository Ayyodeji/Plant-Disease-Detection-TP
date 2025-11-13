"""
Feature Extraction Module

Implements manual feature extraction (color, texture, shape) and
deep learning feature extraction using pre-trained CNNs.
"""

import numpy as np
import cv2
import yaml
from typing import List, Dict, Tuple
from skimage import feature, measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import ndimage
import mahotas
from tqdm import tqdm


class ManualFeatureExtractor:
    """Extracts handcrafted features from images."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ManualFeatureExtractor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']['manual']
    
    def extract_color_histogram(
        self, 
        image: np.ndarray,
        bins: int = None
    ) -> np.ndarray:
        """
        Extract color histogram features.
        
        Args:
            image: Input image (RGB)
            bins: Number of bins per channel
            
        Returns:
            Flattened histogram features
        """
        if bins is None:
            bins = self.feature_config['color_histogram_bins']
        
        # Ensure image is in [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        histograms = []
        
        # Compute histogram for each channel
        for channel in range(3):
            hist = cv2.calcHist(
                [image], [channel], None, [bins], [0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        return np.concatenate(histograms)
    
    def extract_color_moments(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color moments (mean, std, skewness).
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Color moment features
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        features = []
        
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            
            # Mean
            mean = np.mean(channel_data)
            # Standard deviation
            std = np.std(channel_data)
            # Skewness
            skewness = np.mean(((channel_data - mean) / (std + 1e-7)) ** 3)
            
            features.extend([mean, std, skewness])
        
        return np.array(features)
    
    def extract_haralick_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Haralick texture features.
        
        Args:
            image: Input image
            
        Returns:
            Haralick texture features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = image
        
        # Compute Haralick features
        haralick = mahotas.features.haralick(gray, ignore_zeros=True)
        
        # Return mean across directions
        return haralick.mean(axis=0)
    
    def extract_lbp_features(
        self, 
        image: np.ndarray,
        num_points: int = 24,
        radius: int = 3
    ) -> np.ndarray:
        """
        Extract Local Binary Pattern (LBP) features.
        
        Args:
            image: Input image
            num_points: Number of points in LBP
            radius: Radius of LBP
            
        Returns:
            LBP histogram features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = image
        
        # Compute LBP
        lbp = local_binary_pattern(
            gray, num_points, radius, method='uniform'
        )
        
        # Compute histogram
        n_bins = num_points + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        
        return hist
    
    def extract_gabor_features(
        self, 
        image: np.ndarray,
        frequencies: List[float] = None,
        thetas: List[float] = None
    ) -> np.ndarray:
        """
        Extract Gabor filter features.
        
        Args:
            image: Input image
            frequencies: List of frequencies to use
            thetas: List of orientations to use
            
        Returns:
            Gabor features
        """
        if frequencies is None:
            frequencies = [0.1, 0.3, 0.5]
        if thetas is None:
            thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = image
        
        features = []
        
        for frequency in frequencies:
            for theta in thetas:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=5.0,
                    theta=theta,
                    lambd=1.0/frequency,
                    gamma=0.5,
                    psi=0
                )
                
                # Apply filter
                filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                
                # Compute statistics
                features.extend([
                    filtered.mean(),
                    filtered.std()
                ])
        
        return np.array(features)
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract shape features (Hu moments, area, perimeter).
        
        Args:
            image: Input image
            
        Returns:
            Shape features
        """
        # Convert to grayscale and threshold
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = image
        
        # Threshold to get binary image
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            # Return zeros if no contours found
            return np.zeros(9)  # 7 Hu moments + area + perimeter
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute Hu moments
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log scale for Hu moments
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # Compute area and perimeter
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Normalize by image size
        img_area = image.shape[0] * image.shape[1]
        area_ratio = area / img_area
        perimeter_ratio = perimeter / (2 * (image.shape[0] + image.shape[1]))
        
        features = np.concatenate([
            hu_moments,
            [area_ratio, perimeter_ratio]
        ])
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all manual features.
        
        Args:
            image: Input image
            
        Returns:
            Concatenated feature vector
        """
        features = []
        
        # Color features
        features.append(self.extract_color_histogram(image))
        features.append(self.extract_color_moments(image))
        
        # Texture features
        if 'haralick' in self.feature_config['texture_features']:
            features.append(self.extract_haralick_features(image))
        
        if 'lbp' in self.feature_config['texture_features']:
            features.append(self.extract_lbp_features(image))
        
        if 'gabor' in self.feature_config['texture_features']:
            features.append(self.extract_gabor_features(image))
        
        # Shape features
        if self.feature_config['shape_features']:
            features.append(self.extract_shape_features(image))
        
        return np.concatenate(features)
    
    def extract_batch_features(
        self,
        images: List[np.ndarray],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features from batch of images.
        
        Args:
            images: List of images
            show_progress: Whether to show progress bar
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = []
        
        iterator = tqdm(images, desc="Extracting features") if show_progress else images
        
        for img in iterator:
            try:
                feats = self.extract_all_features(img)
                features_list.append(feats)
            except Exception as e:
                print(f"Error extracting features: {e}")
                # Add zeros for failed extraction
                if features_list:
                    features_list.append(np.zeros_like(features_list[0]))
        
        return np.array(features_list)


class CNNFeatureExtractor:
    """Extracts features using pre-trained CNN models."""
    
    def __init__(self, model_name: str = 'mobilenet_v2', config_path: str = "config.yaml"):
        """
        Initialize CNN feature extractor.
        
        Args:
            model_name: Name of pre-trained model to use
            config_path: Path to configuration file
        """
        import torch
        import torchvision.models as models
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'mobilenet_v2':
            base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            # Remove classifier to get features
            self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 1280
        elif model_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove classifier to get features
            self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Remove classifier to get features
            self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Loaded {model_name} for feature extraction")
        print(f"Feature dimension: {feature_dim}")
        print(f"Using device: {self.device}")
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images using CNN.
        
        Args:
            images: Batch of preprocessed images (numpy array)
            
        Returns:
            Feature matrix
        """
        import torch
        
        # Convert numpy array to torch tensor
        # Assuming images are in (N, H, W, C) format, need (N, C, H, W) for PyTorch
        if images.shape[-1] == 3:  # If channels last
            images = np.transpose(images, (0, 3, 1, 2))
        
        images_tensor = torch.from_numpy(images).float().to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(images_tensor)
            # Flatten spatial dimensions if needed
            features = features.view(features.size(0), -1)
            features = features.cpu().numpy()
        
        return features

