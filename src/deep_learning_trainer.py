"""
Deep Learning Training Module

Trains deep learning models (MobileNet, ResNet, EfficientNet) for plant disease classification.
Converted from TensorFlow to PyTorch.
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm


class PlantDiseaseDataset(Dataset):
    """Custom PyTorch Dataset for plant disease images."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize dataset.
        
        Args:
            images: Image array (N, H, W, C)
            labels: Label array
            transform: Optional transforms
        """
        # Convert images from (N, H, W, C) to (N, C, H, W) for PyTorch
        if images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
        
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


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
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        self.models = {}
        self.histories = {}
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config['data']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def create_model(
        self,
        model_name: str,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        freeze_base: bool = False
    ) -> nn.Module:
        """
        Create a deep learning model.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            input_shape: Input image shape (height, width, channels)
            freeze_base: Whether to freeze base model weights
            
        Returns:
            PyTorch model
        """
        print(f"\nCreating {model_name} model...")
        
        # Load base model
        if model_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            base_model = models.mobilenet_v2(weights=weights)
            feature_dim = 1280  # MobileNetV2 feature dimension
            # Use the features part only
            base_model = base_model.features
        
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            base_model = models.resnet50(weights=weights)
            feature_dim = 2048  # ResNet50 feature dimension
            # Remove the last fc layer and avgpool
            base_model = nn.Sequential(*list(base_model.children())[:-2])
        
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base_model = models.efficientnet_b0(weights=weights)
            feature_dim = 1280  # EfficientNetB0 feature dimension
            # Use the features part only
            base_model = base_model.features
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze base model if specified
        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Build complete model
        model = nn.Sequential(
            base_model,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
        print(f"Base model frozen: {freeze_base}")
        
        return model
    
    def get_optimizer(
        self,
        model: nn.Module,
        learning_rate: Optional[float] = None
    ) -> optim.Optimizer:
        """
        Get optimizer for model.
        
        Args:
            model: PyTorch model
            learning_rate: Learning rate (uses config if None)
            
        Returns:
            Optimizer
        """
        if learning_rate is None:
            learning_rate = self.dl_config['learning_rate']
        
        optimizer_name = self.dl_config['optimizer']
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        return optimizer
    
    def get_scheduler(
        self,
        optimizer: optim.Optimizer,
        patience: int = None
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Get learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            patience: Patience for ReduceLROnPlateau
            
        Returns:
            Scheduler
        """
        if patience is None:
            patience = self.dl_config['reduce_lr_patience']
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=patience,
            min_lr=1e-7
        )
        
        return scheduler
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions (logits)
            targets: True labels
            
        Returns:
            Dictionary of metrics
        """
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == targets).float()
        accuracy = correct.mean().item()
        
        # Calculate precision and recall (macro average)
        num_classes = predictions.shape[1]
        precision_sum = 0
        recall_sum = 0
        
        for c in range(num_classes):
            true_positives = ((pred_classes == c) & (targets == c)).sum().float()
            predicted_positives = (pred_classes == c).sum().float()
            actual_positives = (targets == c).sum().float()
            
            precision = true_positives / (predicted_positives + 1e-10)
            recall = true_positives / (actual_positives + 1e-10)
            
            precision_sum += precision.item()
            recall_sum += recall.item()
        
        return {
            'accuracy': accuracy,
            'precision': precision_sum / num_classes,
            'recall': recall_sum / num_classes
        }
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            all_predictions.append(outputs.detach())
            all_targets.append(labels.detach())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss
        
        return metrics
    
    def validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            model: PyTorch model
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track metrics
                running_loss += loss.item() * images.size(0)
                all_predictions.append(outputs)
                all_targets.append(labels)
        
        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss
        
        return metrics
    
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
        
        # Create datasets and data loaders
        train_dataset = PlantDiseaseDataset(X_train, y_train)
        val_dataset = PlantDiseaseDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.dl_config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.dl_config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Phase 1: Train with frozen base
        print("\nPhase 1: Training with frozen base model...")
        model = self.create_model(model_name, num_classes, freeze_base=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        
        history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        max_patience = self.dl_config['early_stopping_patience']
        
        phase1_epochs = min(10, self.dl_config['epochs'] // 3)
        
        for epoch in range(1, phase1_epochs + 1):
            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_metrics['accuracy'])
            
            # Record history
            history['loss'].append(train_metrics['loss'])
            history['accuracy'].append(train_metrics['accuracy'])
            history['precision'].append(train_metrics['precision'])
            history['recall'].append(train_metrics['recall'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                checkpoint_path = self.output_dir / f'{model_name}_phase1_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_accuracy,
                }, checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Phase 2: Fine-tuning (if enabled)
        if fine_tune:
            print("\nPhase 2: Fine-tuning entire model...")
            
            # Load best model from phase 1
            checkpoint = torch.load(self.output_dir / f'{model_name}_phase1_best.pth')
            
            # Create new model with unfrozen base
            model = self.create_model(model_name, num_classes, freeze_base=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # New optimizer with lower learning rate
            optimizer = self.get_optimizer(model, learning_rate=self.dl_config['learning_rate'] / 10)
            scheduler = self.get_scheduler(optimizer)
            
            best_val_accuracy = checkpoint['val_accuracy']
            patience_counter = 0
            
            initial_epoch = len(history['loss'])
            
            for epoch in range(1, self.dl_config['epochs'] - initial_epoch + 1):
                # Train
                train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, initial_epoch + epoch)
                
                # Validate
                val_metrics = self.validate_epoch(model, val_loader, criterion)
                
                # Update scheduler
                scheduler.step(val_metrics['accuracy'])
                
                # Record history
                history['loss'].append(train_metrics['loss'])
                history['accuracy'].append(train_metrics['accuracy'])
                history['precision'].append(train_metrics['precision'])
                history['recall'].append(train_metrics['recall'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_precision'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                
                print(f"\nEpoch {initial_epoch + epoch}:")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    checkpoint_path = self.output_dir / f'{model_name}_best.pth'
                    torch.save({
                        'epoch': initial_epoch + epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': best_val_accuracy,
                    }, checkpoint_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping triggered at epoch {initial_epoch + epoch}")
                    break
            
            # Load best model
            checkpoint = torch.load(self.output_dir / f'{model_name}_best.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save final model
        model_path = self.output_dir / f'{model_name}_final.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'model_name': model_name,
        }, model_path)
        print(f"\nFinal model saved to: {model_path}")
        
        # Store model and history
        self.models[model_name] = model
        self.histories[model_name] = history
        
        # Extract results
        results = {
            'model_name': model_name,
            'final_train_accuracy': float(history['accuracy'][-1]),
            'final_val_accuracy': float(history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(history['val_accuracy'])),
            'total_epochs': len(history['loss'])
        }
        
        # Save training history
        history_path = self.output_dir / f'{model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return results
    
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
            try:
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
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"ERROR: Failed to train {model_name}")
                print(f"Reason: {str(e)}")
                print(f"Skipping this model and continuing with others...")
                print(f"{'='*70}\n")
                all_results[model_name] = {'error': str(e), 'status': 'failed'}
                continue
        
        # Save all results
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"All results saved to: {results_path}")
        print(f"Successfully trained: {sum(1 for r in all_results.values() if 'error' not in r)} models")
        print(f"Failed to train: {sum(1 for r in all_results.values() if 'error' in r)} models")
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
        
        history = self.histories[model_name]
        
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
    
    def load_model(self, model_name: str, num_classes: int, phase: str = 'final') -> nn.Module:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model
            num_classes: Number of classes
            phase: Which phase to load ('final' or 'best')
            
        Returns:
            Loaded model
        """
        if phase == 'final':
            model_path = self.output_dir / f'{model_name}_final.pth'
        else:
            model_path = self.output_dir / f'{model_name}_best.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create model architecture
        model = self.create_model(model_name, num_classes, freeze_base=False)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.models[model_name] = model
        return model
