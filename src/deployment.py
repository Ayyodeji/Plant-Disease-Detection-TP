"""
Deployment Module

Converts models to deployment-ready formats (ONNX, TFLite)
and creates inference pipelines for edge devices.
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
import joblib
import onnx
import onnxruntime as ort
from tensorflow import keras


class ModelConverter:
    """Convert models to deployment formats."""
    
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
    
    def convert_to_tflite(
        self,
        model: keras.Model,
        model_name: str,
        quantize: bool = True
    ) -> str:
        """
        Convert Keras model to TensorFlow Lite.
        
        Args:
            model: Keras model
            model_name: Name of the model
            quantize: Whether to apply quantization
            
        Returns:
            Path to saved TFLite model
        """
        print(f"\nConverting {model_name} to TensorFlow Lite...")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            print("Applying quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save
        suffix = '_quantized' if quantize else ''
        save_path = self.deploy_dir / f'{model_name}{suffix}.tflite'
        
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"TFLite model saved to: {save_path}")
        print(f"Model size: {size_mb:.2f} MB")
        
        return str(save_path)
    
    def convert_to_onnx(
        self,
        model_path: str,
        model_name: str,
        input_shape: Tuple[int, int, int, int] = (1, 224, 224, 3)
    ) -> str:
        """
        Convert Keras model to ONNX format.
        
        Args:
            model_path: Path to saved Keras model
            model_name: Name of the model
            input_shape: Input shape (batch, height, width, channels)
            
        Returns:
            Path to saved ONNX model
        """
        try:
            import tf2onnx
            
            print(f"\nConverting {model_name} to ONNX...")
            
            # Load model
            model = keras.models.load_model(model_path)
            
            # Convert to ONNX
            spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
            
            output_path = str(self.deploy_dir / f'{model_name}.onnx')
            
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                output_path=output_path
            )
            
            print(f"ONNX model saved to: {output_path}")
            
            return output_path
        
        except ImportError:
            print("tf2onnx not installed. Install with: pip install tf2onnx")
            return None
    
    def test_tflite_inference(
        self,
        tflite_path: str,
        test_image: np.ndarray
    ) -> np.ndarray:
        """
        Test TFLite model inference.
        
        Args:
            tflite_path: Path to TFLite model
            test_image: Test image
            
        Returns:
            Model predictions
        """
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_shape = input_details[0]['shape']
        if len(test_image.shape) == 3:
            test_image = np.expand_dims(test_image, axis=0)
        
        test_image = test_image.astype(input_details[0]['dtype'])
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data
    
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
            test_image = np.expand_dims(test_image, axis=0)
        
        test_image = test_image.astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: test_image})
        
        return outputs[0]


class InferencePipeline:
    """Production-ready inference pipeline."""
    
    def __init__(
        self,
        model_path: str,
        class_mapping_path: str,
        config_path: str = "config.yaml",
        model_type: str = "keras"
    ):
        """
        Initialize InferencePipeline.
        
        Args:
            model_path: Path to model file
            class_mapping_path: Path to class mapping JSON
            config_path: Path to configuration file
            model_type: Type of model ('keras', 'tflite', 'onnx', 'sklearn')
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Load model
        self.model_type = model_type
        self.model_path = model_path
        self._load_model()
        
        # Initialize preprocessor
        from preprocessing import ImagePreprocessor
        self.preprocessor = ImagePreprocessor(config_path)
    
    def _load_model(self):
        """Load model based on type."""
        if self.model_type == 'keras':
            self.model = keras.models.load_model(self.model_path)
        
        elif self.model_type == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        
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
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        # Make prediction based on model type
        if self.model_type == 'keras':
            probabilities = self.model.predict(image_batch, verbose=0)[0]
        
        elif self.model_type == 'tflite':
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                image_batch.astype(self.input_details[0]['dtype'])
            )
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]
        
        elif self.model_type == 'onnx':
            probabilities = self.session.run(
                None,
                {self.input_name: image_batch.astype(np.float32)}
            )[0][0]
        
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

## Requirements
See requirements.txt for dependencies.

## License
Please ensure compliance with dataset and model licensing terms.
"""
        
        with open(output_path / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"\nDeployment package saved to: {output_path}")


if __name__ == "__main__":
    print("Deployment module initialized")
    print("This module handles model conversion and inference")



