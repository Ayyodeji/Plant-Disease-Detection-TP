"""
Classical Machine Learning Training Module

Trains classical ML models (Random Forest, SVM, Gradient Boosting)
on manually extracted features.
"""

import numpy as np
import yaml
import joblib
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from tqdm import tqdm
import json


class ClassicalMLTrainer:
    """Trains classical machine learning models."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ClassicalMLTrainer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ml_config = self.config['classical_ml']
        self.output_dir = Path(self.config['output']['models_dir']) / 'classical_ml'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
    
    def _create_model(self, model_name: str):
        """
        Create model instance based on configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        if model_name == 'random_forest':
            config = self.ml_config['random_forest']
            return RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                n_jobs=config['n_jobs'],
                random_state=self.config['data']['random_seed'],
                verbose=1
            )
        
        elif model_name == 'svm':
            config = self.ml_config['svm']
            return SVC(
                kernel=config['kernel'],
                C=config['C'],
                gamma=config['gamma'],
                probability=True,
                random_state=self.config['data']['random_seed'],
                verbose=True
            )
        
        elif model_name == 'gradient_boosting':
            config = self.ml_config['gradient_boosting']
            return GradientBoostingClassifier(
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                max_depth=config['max_depth'],
                random_state=self.config['data']['random_seed'],
                verbose=1
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def prepare_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled features
        """
        print("Scaling features...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        scaler_path = self.output_dir / 'feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        use_cross_validation: bool = True
    ) -> Dict:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_cross_validation: Whether to perform cross-validation
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*70}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*70}")
        
        # Create model
        model = self._create_model(model_name)
        
        results = {'model_name': model_name}
        
        # Cross-validation
        if use_cross_validation:
            print("\nPerforming cross-validation...")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['evaluation']['cross_validation_folds'],
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            results['cv_scores'] = cv_scores.tolist()
            results['cv_mean'] = float(cv_scores.mean())
            results['cv_std'] = float(cv_scores.std())
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        print("\nTraining on full training set...")
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_score = model.score(X_train, y_train)
        results['train_accuracy'] = float(train_score)
        print(f"Training Accuracy: {train_score:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            results['val_accuracy'] = float(val_score)
            print(f"Validation Accuracy: {val_score:.4f}")
        
        # Save model
        model_path = self.output_dir / f'{model_name}.pkl'
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Store model
        self.models[model_name] = model
        
        return results
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        use_cross_validation: bool = True
    ) -> Dict:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            use_cross_validation: Whether to perform cross-validation
            
        Returns:
            Dictionary with all training results
        """
        all_results = {}
        
        for model_name in self.ml_config['models']:
            results = self.train_model(
                model_name,
                X_train,
                y_train,
                X_val,
                y_val,
                use_cross_validation
            )
            all_results[model_name] = results
        
        # Save results
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"All results saved to: {results_path}")
        print(f"{'='*70}")
        
        return all_results
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict
    ):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Best model and results
        """
        print(f"\nHyperparameter tuning for {model_name}...")
        
        base_model = self._create_model(model_name)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Save best model
        best_model_path = self.output_dir / f'{model_name}_tuned.pkl'
        joblib.dump(grid_search.best_estimator_, best_model_path)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def load_model(self, model_name: str):
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = self.output_dir / f'{model_name}.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        self.models[model_name] = model
        return model
    
    def predict(
        self,
        model_name: str,
        X: np.ndarray,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model
            X: Input features
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or probabilities
        """
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if return_proba:
            return model.predict_proba(X_scaled)
        else:
            return model.predict(X_scaled)


def get_default_param_grids():
    """Get default parameter grids for hyperparameter tuning."""
    return {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly']
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }

