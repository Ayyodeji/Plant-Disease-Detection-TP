"""
Model Evaluation Module

Comprehensive evaluation of models using various metrics,
confusion matrices, ROC curves, and per-class performance analysis.
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelEvaluator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config['evaluation']
        self.viz_dir = Path(self.config['output']['visualizations_dir'])
        self.results_dir = Path(self.config['output']['results_dir'])
        
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "model"
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            class_names: List of class names
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}")
        print(f"{'='*70}")
        
        results = {'model_name': model_name}
        
        # Basic metrics
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        results['precision_macro'] = float(precision_score(
            y_true, y_pred, average='macro', zero_division=0
        ))
        results['recall_macro'] = float(recall_score(
            y_true, y_pred, average='macro', zero_division=0
        ))
        results['f1_macro'] = float(f1_score(
            y_true, y_pred, average='macro', zero_division=0
        ))
        
        # Weighted metrics
        results['precision_weighted'] = float(precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        ))
        results['recall_weighted'] = float(recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        ))
        results['f1_weighted'] = float(f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        ))
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision_macro']:.4f} (macro)")
        print(f"  Recall:    {results['recall_macro']:.4f} (macro)")
        print(f"  F1-Score:  {results['f1_macro']:.4f} (macro)")
        
        # Per-class metrics
        if self.eval_config['per_class_metrics']:
            per_class = self._compute_per_class_metrics(
                y_true, y_pred, class_names
            )
            results['per_class_metrics'] = per_class
            self._save_per_class_report(per_class, model_name)
        
        # Confusion matrix
        if self.eval_config['confusion_matrix']:
            cm = confusion_matrix(y_true, y_pred)
            results['confusion_matrix'] = cm.tolist()
            self.plot_confusion_matrix(cm, class_names, model_name)
        
        # ROC curve and AUC
        if self.eval_config['roc_curve'] and y_pred_proba is not None:
            roc_metrics = self._compute_roc_metrics(
                y_true, y_pred_proba, class_names
            )
            results['roc_auc'] = roc_metrics
            self.plot_roc_curves(
                y_true, y_pred_proba, class_names, model_name
            )
        
        # Save results
        results_path = self.results_dir / f'{model_name}_evaluation.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
        return results
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """Compute per-class metrics."""
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        return report
    
    def _save_per_class_report(
        self,
        per_class_metrics: Dict,
        model_name: str
    ) -> None:
        """Save per-class metrics as CSV."""
        # Convert to DataFrame
        df = pd.DataFrame(per_class_metrics).T
        
        # Save
        csv_path = self.results_dir / f'{model_name}_per_class_metrics.csv'
        df.to_csv(csv_path)
        
        print(f"Per-class metrics saved to: {csv_path}")
    
    def _compute_roc_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """Compute ROC AUC metrics."""
        n_classes = y_pred_proba.shape[1]
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        roc_metrics = {}
        
        # Compute ROC AUC for each class
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = class_names[i] if class_names else f"Class_{i}"
                roc_metrics[class_name] = float(roc_auc)
            except:
                pass
        
        # Compute macro and micro average
        try:
            roc_metrics['macro'] = float(roc_auc_score(
                y_true_bin, y_pred_proba, average='macro'
            ))
            roc_metrics['micro'] = float(roc_auc_score(
                y_true_bin, y_pred_proba, average='micro'
            ))
        except:
            pass
        
        return roc_metrics
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        model_name: str = "model",
        normalize: bool = False
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            model_name: Name of the model
            normalize: Whether to normalize the confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Determine figure size based on number of classes
        n_classes = cm.shape[0]
        figsize = max(10, n_classes * 0.5)
        
        plt.figure(figsize=(figsize, figsize))
        
        # Use smaller font if many classes
        if n_classes > 20:
            sns.heatmap(
                cm, annot=False, fmt='.2f' if normalize else 'd',
                cmap='Blues', cbar=True
            )
        else:
            sns.heatmap(
                cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                cbar=True
            )
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save
        suffix = '_normalized' if normalize else ''
        save_path = self.viz_dir / f'{model_name}_confusion_matrix{suffix}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        model_name: str = "model",
        plot_micro_macro: bool = True
    ) -> None:
        """
        Plot ROC curves.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            class_names: List of class names
            model_name: Name of the model
            plot_micro_macro: Whether to plot micro/macro averages
        """
        n_classes = y_pred_proba.shape[1]
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Plot individual class ROC curves (limit to first 10 for readability)
        n_plot = min(10, n_classes)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_plot))
        
        for i, color in zip(range(n_plot), colors):
            class_name = class_names[i] if class_names else f"Class {i}"
            plt.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
            )
        
        # Plot micro-average ROC curve
        if plot_micro_macro and n_classes > 2:
            fpr_micro, tpr_micro, _ = roc_curve(
                y_true_bin.ravel(), y_pred_proba.ravel()
            )
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            
            plt.plot(
                fpr_micro, tpr_micro,
                label=f'Micro-average (AUC = {roc_auc_micro:.2f})',
                color='deeppink', linestyle=':', linewidth=3
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {model_name}', fontsize=14)
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save
        save_path = self.viz_dir / f'{model_name}_roc_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to: {save_path}")
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        metrics: List[str] = None
    ) -> None:
        """
        Compare multiple models.
        
        Args:
            results_dict: Dictionary of model results
            metrics: List of metrics to compare
        """
        if metrics is None:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in results:
                    row[metric] = results[metric]
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        csv_path = self.results_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nModel comparison saved to: {csv_path}")
        
        # Print comparison
        print("\nModel Comparison:")
        print(df.to_string(index=False))
        
        # Plot comparison
        self._plot_model_comparison(df, metrics)
    
    def _plot_model_comparison(
        self,
        df: pd.DataFrame,
        metrics: List[str]
    ) -> None:
        """Plot model comparison bar chart."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[idx]
                df.plot(
                    x='Model', y=metric, kind='bar',
                    ax=ax, legend=False, color='steelblue'
                )
                ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
                ax.set_ylabel('Score', fontsize=10)
                ax.set_xlabel('')
                ax.set_ylim([0, 1])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', fontsize=8)
        
        plt.tight_layout()
        
        save_path = self.viz_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison chart saved to: {save_path}")
    
    def _prepare_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

