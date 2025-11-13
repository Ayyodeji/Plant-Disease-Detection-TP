"""
Deployment Module

Converts models to deployment-ready formats (ONNX, TorchScript)
and creates inference pipelines for edge devices.
Converted from TensorFlow to PyTorch.
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import joblib
import onnx
import onnxruntime as ort


class ModelConverter:
    """Convert PyTorch models to deployment formats."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelConverter.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.deploy_config = self.config['deployment']
        self.models_dir = Path(self.config['output']['models_dir'])
        self.deploy_dir = self.models_dir / 'deployment'
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_to_torchscript(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        optimize: bool = True
    ) -> str:
        """
        Convert PyTorch model to TorchScript.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            input_shape: Input shape (batch, channels, height, width)
            optimize: Whether to optimize for mobile
            
        Returns:
            Path to saved TorchScript model
        """
        print(f"\nConverting {model_name} to TorchScript...")
        
        model.eval()
        
        # Create example input
        example_input = torch.randn(input_shape).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        if optimize:
            print("Optimizing for mobile...")
            # Optimize for mobile deployment
            from torch.utils.mobile_optimizer import optimize_for_mobile
            traced_model = optimize_for_mobile(traced_model)
        
        # Save
        suffix = '_optimized' if optimize else ''
        save_path = self.deploy_dir / f'{model_name}{suffix}.pt'
        
        traced_model.save(str(save_path))
        
        # Get model size
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"TorchScript model saved to: {save_path}")
        print(f"Model size: {size_mb:.2f} MB")
        
        return str(save_path)
    
    def convert_to_onnx(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset_version: int = 12
    ) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            input_shape: Input shape (batch, channels, height, width)
            opset_version: ONNX opset version
            
        Returns:
            Path to saved ONNX model
        """
        print(f"\nConverting {model_name} to ONNX...")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX
        output_path = str(self.deploy_dir / f'{model_name}.onnx')
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"ONNX model saved to: {output_path}")
        print(f"Model size: {size_mb:.2f} MB")
        
        return output_path
    
    def quantize_model(
        self,
        model: nn.Module,
        model_name: str,
        calibration_data: Optional[torch.Tensor] = None
    ) -> str:
        """
        Quantize PyTorch model for faster inference.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            calibration_data: Optional calibration data for quantization
            
        Returns:
            Path to saved quantized model
        """
        print(f"\nQuantizing {model_name}...")
        
        model.eval()
        model.cpu()  # Quantization works on CPU
        
        # Dynamic quantization (works without calibration data)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save quantized model
        save_path = self.deploy_dir / f'{model_name}_quantized.pth'
        
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'model_name': model_name,
        }, save_path)
        
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"Quantized model saved to: {save_path}")
        print(f"Model size: {size_mb:.2f} MB")
        
        return str(save_path)
    
    def test_torchscript_inference(
        self,
        torchscript_path: str,
        test_image: np.ndarray
    ) -> np.ndarray:
        """
        Test TorchScript model inference.
        
        Args:
            torchscript_path: Path to TorchScript model
            test_image: Test image
            
        Returns:
            Model predictions
        """
        # Load TorchScript model
        model = torch.jit.load(torchscript_path)
        model.eval()
        
        # Prepare input
        if len(test_image.shape) == 3:
            # Add batch dimension and convert to CHW format
            test_image = np.transpose(test_image, (2, 0, 1))
            test_image = np.expand_dims(test_image, axis=0)
        elif len(test_image.shape) == 4 and test_image.shape[-1] == 3:
            # Convert from NHWC to NCHW
            test_image = np.transpose(test_image, (0, 3, 1, 2))
        
        test_tensor = torch.from_numpy(test_image).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = model(test_tensor)
            output = torch.softmax(output, dim=1)
            output = output.cpu().numpy()
        
        return output
    
    def test_onnx_inference(
        self,
        onnx_path: str,
        test_image: np.ndarray
    ) -> np.ndarray:
        """
        Test ONNX model inference.
        
        Args:
            onnx_path: Path to ONNX model
            test_image: Test image
            
        Returns:
            Model predictions
        """
        # Create ONNX runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Prepare input
        if len(test_image.shape) == 3:
            # Add batch dimension and convert to CHW format
            test_image = np.transpose(test_image, (2, 0, 1))
            test_image = np.expand_dims(test_image, axis=0)
        elif len(test_image.shape) == 4 and test_image.shape[-1] == 3:
            # Convert from NHWC to NCHW
            test_image = np.transpose(test_image, (0, 3, 1, 2))
        
        test_image = test_image.astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: test_image})
        
        return outputs[0]


class InferencePipeline:
    """Production-ready inference pipeline for PyTorch models."""
    
    def __init__(
        self,
        model_path: str,
        class_mapping_path: str,
        config_path: str = "config.yaml",
        model_type: str = "pytorch"
    ):
        """
        Initialize InferencePipeline.
        
        Args:
            model_path: Path to model file
            class_mapping_path: Path to class mapping JSON
            config_path: Path to configuration file
            model_type: Type of model ('pytorch', 'torchscript', 'onnx', 'sklearn')
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model_type = model_type
        self.model_path = model_path
        self._load_model()
        
        # Initialize preprocessor
        from preprocessing import ImagePreprocessor
        self.preprocessor = ImagePreprocessor(config_path)
    
    def _load_model(self):
        """Load model based on type."""
        if self.model_type == 'pytorch':
            # Load PyTorch model
            from deep_learning_trainer import DeepLearningTrainer
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model_name = checkpoint.get('model_name', 'mobilenet_v2')
            
            trainer = DeepLearningTrainer(config_path='config.yaml')
            self.model = trainer.create_model(model_name, self.num_classes, freeze_base=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        
        elif self.model_type == 'torchscript':
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
        
        elif self.model_type == 'onnx':
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
        
        elif self.model_type == 'sklearn':
            self.model = joblib.load(self.model_path)
            # Also load scaler if it exists
            scaler_path = Path(self.model_path).parent / 'feature_scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Loaded {self.model_type} model from: {self.model_path}")
    
    def predict(
        self,
        image_path: str,
        top_k: int = 3,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
            return_probabilities: Whether to return probabilities
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        image = self.preprocessor.preprocess(image_path)
        
        # Add batch dimension and convert to CHW format if needed
        if len(image.shape) == 3:
            # Convert from HWC to CHW
            if image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))
            image_batch = np.expand_dims(image, axis=0)
        else:
            # Already has batch dimension, check format
            if image.shape[-1] == 3:
                image_batch = np.transpose(image, (0, 3, 1, 2))
            else:
                image_batch = image
        
        # Make prediction based on model type
        if self.model_type in ['pytorch', 'torchscript']:
            image_tensor = torch.from_numpy(image_batch).float().to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        elif self.model_type == 'onnx':
            outputs = self.session.run(
                None,
                {self.input_name: image_batch.astype(np.float32)}
            )[0][0]
            
            # Apply softmax
            exp_outputs = np.exp(outputs - np.max(outputs))
            probabilities = exp_outputs / exp_outputs.sum()
        
        elif self.model_type == 'sklearn':
            # For sklearn, we need features instead of raw image
            from feature_extraction import ManualFeatureExtractor
            extractor = ManualFeatureExtractor()
            
            # Load and preprocess image for feature extraction
            img = self.preprocessor.load_image(image_path)
            img = self.preprocessor.resize_image(img)
            
            features = extractor.extract_all_features(img)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            predictions.append({
                'class': self.idx_to_class[idx],
                'probability': float(probabilities[idx]),
                'confidence_percentage': float(probabilities[idx] * 100)
            })
        
        result = {
            'image_path': image_path,
            'top_prediction': predictions[0],
            'top_k_predictions': predictions,
            'all_probabilities': probabilities.tolist() if return_probabilities else None
        }
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Make predictions on batch of images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for path in batch_paths:
                try:
                    result = self.predict(path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    results.append({
                        'image_path': path,
                        'error': str(e)
                    })
        
        return results
    
    def save_deployment_package(self, output_dir: str) -> None:
        """
        Create deployment package with model and metadata.
        
        Args:
            output_dir: Output directory for deployment package
        """
        import shutil
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model
        shutil.copy(self.model_path, output_path / Path(self.model_path).name)
        
        # Copy class mapping
        shutil.copy(
            Path(self.config['data']['processed_data_dir']) / 'class_mapping.json',
            output_path / 'class_mapping.json'
        )
        
        # Copy config
        shutil.copy('config.yaml', output_path / 'config.yaml')
        
        # Create README
        readme_content = f"""
# Plant Disease Detection - Deployment Package

## Model Information
- Model Type: {self.model_type}
- Framework: PyTorch
- Number of Classes: {self.num_classes}
- Input Size: {self.config['preprocessing']['target_size']}

## Usage

```python
from deployment import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline(
    model_path='path/to/model',
    class_mapping_path='class_mapping.json',
    model_type='{self.model_type}'
)

# Make prediction
result = pipeline.predict('path/to/image.jpg')
print(result['top_prediction'])
```

## Model Types Supported
- `pytorch`: Standard PyTorch model (.pth)
- `torchscript`: TorchScript model for deployment (.pt)
- `onnx`: ONNX model for cross-platform deployment (.onnx)
- `sklearn`: Classical ML model (.pkl)

## Requirements
See requirements.txt for dependencies.

## Performance
For best performance:
- Use TorchScript models for production deployments
- Use ONNX models for edge devices or non-Python environments
- Use quantized models for mobile/embedded devices

## License
Please ensure compliance with dataset and model licensing terms.
"""
        
        with open(output_path / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"\nDeployment package saved to: {output_path}")
