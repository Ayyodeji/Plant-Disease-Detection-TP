"""
Deep Learning Training Module

Trains deep learning models (MobileNet, ResNet, EfficientNet) for plant disease classification.
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0
)
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm


class DeepLearningTrainer:
    """Trains deep learning models for plant disease classification."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DeepLearningTrainer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dl_config = self.config['deep_learning']
        self.output_dir = Path(self.config['output']['models_dir']) / 'deep_learning'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        self.models = {}
        self.histories = {}
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config['data']['random_seed']
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def create_model(
        self,
        model_name: str,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        freeze_base: bool = False
    ) -> keras.Model:
        """
        Create a deep learning model.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            input_shape: Input image shape
            freeze_base: Whether to freeze base model weights
            
        Returns:
            Keras model
        """
        print(f"\nCreating {model_name} model...")
        
        # Load base model
        if model_name == 'mobilenet_v2':
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif model_name == 'resnet50':
            base_model = ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif model_name == 'efficientnet_b0':
            base_model = EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze base model if specified
        base_model.trainable = not freeze_base
        
        # Build model
        inputs = keras.Input(shape=input_shape)
        
        # Data augmentation layers (applied during training)
        x = inputs
        
        # Base model
        x = base_model(x, training=not freeze_base)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        print(f"Model created with {model.count_params():,} parameters")
        print(f"Base model frozen: {freeze_base}")
        
        return model
    
    def compile_model(
        self,
        model: keras.Model,
        learning_rate: Optional[float] = None
    ) -> keras.Model:
        """
        Compile model with optimizer and loss.
        
        Args:
            model: Keras model
            learning_rate: Learning rate (uses config if None)
            
        Returns:
            Compiled model
        """
        if learning_rate is None:
            learning_rate = self.dl_config['learning_rate']
        
        optimizer_name = self.dl_config['optimizer']
        
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            optimizer = optimizer_name
        
        model.compile(
            optimizer=optimizer,
            loss=self.dl_config['loss'],
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def get_callbacks(
        self,
        model_name: str,
        monitor: str = 'val_accuracy'
    ) -> List[keras.callbacks.Callback]:
        """
        Get training callbacks.
        
        Args:
            model_name: Name of the model
            monitor: Metric to monitor
            
        Returns:
            List of callbacks
        """
        checkpoint_path = self.output_dir / f'{model_name}_best.h5'
        
        callback_list = [
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.dl_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=self.dl_config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(
                self.output_dir / f'{model_name}_training_log.csv'
            )
        ]
        
        return callback_list
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        fine_tune: bool = True
    ) -> Dict:
        """
        Train a deep learning model.
        
        Args:
            model_name: Name of the model
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            num_classes: Number of classes
            fine_tune: Whether to perform fine-tuning after initial training
            
        Returns:
            Training history and results
        """
        print(f"\n{'='*70}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*70}")
        
        # Convert labels to categorical if needed
        if len(y_train.shape) == 1:
            y_train_cat = to_categorical(y_train, num_classes)
            y_val_cat = to_categorical(y_val, num_classes)
        else:
            y_train_cat = y_train
            y_val_cat = y_val
        
        # Phase 1: Train with frozen base
        print("\nPhase 1: Training with frozen base model...")
        model = self.create_model(model_name, num_classes, freeze_base=True)
        model = self.compile_model(model)
        
        history1 = model.fit(
            X_train, y_train_cat,
            batch_size=self.dl_config['batch_size'],
            epochs=min(10, self.dl_config['epochs'] // 3),
            validation_data=(X_val, y_val_cat),
            callbacks=self.get_callbacks(f'{model_name}_phase1'),
            verbose=1
        )
        
        # Phase 2: Fine-tuning (if enabled)
        if fine_tune:
            print("\nPhase 2: Fine-tuning entire model...")
            
            # Unfreeze base model
            for layer in model.layers:
                layer.trainable = True
            
            # Recompile with lower learning rate
            model = self.compile_model(
                model,
                learning_rate=self.dl_config['learning_rate'] / 10
            )
            
            history2 = model.fit(
                X_train, y_train_cat,
                batch_size=self.dl_config['batch_size'],
                epochs=self.dl_config['epochs'],
                initial_epoch=len(history1.history['loss']),
                validation_data=(X_val, y_val_cat),
                callbacks=self.get_callbacks(model_name),
                verbose=1
            )
            
            # Combine histories
            history = self._combine_histories(history1, history2)
        else:
            history = history1
        
        # Save final model
        model_path = self.output_dir / f'{model_name}_final.h5'
        model.save(str(model_path))
        print(f"\nFinal model saved to: {model_path}")
        
        # Store model and history
        self.models[model_name] = model
        self.histories[model_name] = history
        
        # Extract results
        results = {
            'model_name': model_name,
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'total_epochs': len(history.history['loss'])
        }
        
        # Save training history
        history_path = self.output_dir / f'{model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in history.history.items()},
                f,
                indent=2
            )
        
        return results
    
    def _combine_histories(self, hist1, hist2) -> keras.callbacks.History:
        """Combine two training histories."""
        combined = keras.callbacks.History()
        combined.history = {}
        
        for key in hist1.history.keys():
            combined.history[key] = hist1.history[key] + hist2.history[key]
        
        return combined
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int
    ) -> Dict:
        """
        Train all configured deep learning models.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with all training results
        """
        all_results = {}
        
        for model_name in self.dl_config['models']:
            results = self.train_model(
                model_name,
                X_train,
                y_train,
                X_val,
                y_val,
                num_classes
            )
            all_results[model_name] = results
            
            # Plot and save training curves
            self.plot_training_history(model_name)
        
        # Save all results
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"All results saved to: {results_path}")
        print(f"{'='*70}")
        
        return all_results
    
    def plot_training_history(self, model_name: str) -> None:
        """
        Plot and save training history.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.histories:
            return
        
        history = self.histories[model_name].history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Train')
        axes[0, 0].plot(history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Train')
        axes[0, 1].plot(history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history['precision'], label='Train')
        axes[1, 0].plot(history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history['recall'], label='Train')
        axes[1, 1].plot(history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        viz_dir = Path(self.config['output']['visualizations_dir'])
        viz_dir.mkdir(parents=True, exist_ok=True)
        fig_path = viz_dir / f'{model_name}_training_curves.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {fig_path}")
    
    def load_model(self, model_name: str, phase: str = 'final') -> keras.Model:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model
            phase: Which phase to load ('final' or 'best')
            
        Returns:
            Loaded model
        """
        if phase == 'final':
            model_path = self.output_dir / f'{model_name}_final.h5'
        else:
            model_path = self.output_dir / f'{model_name}_best.h5'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = keras.models.load_model(str(model_path))
        self.models[model_name] = model
        return model


if __name__ == "__main__":
    print("Deep Learning Trainer module initialized")
    print("This module should be imported and used in the main training pipeline")



