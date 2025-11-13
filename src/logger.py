"""
Logging Configuration Module

Provides structured logging for the entire plant disease detection system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml


class SystemLogger:
    """Centralized logging system for all modules."""
    
    def __init__(self, config_path: str = "config.yaml", log_name: str = None):
        """
        Initialize system logger.
        
        Args:
            config_path: Path to configuration file
            log_name: Name for this log session (auto-generated if None)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create logs directory
        self.logs_dir = Path(self.config['output']['logs_dir'])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"training_{timestamp}"
        
        self.log_file = self.logs_dir / f"{log_name}.log"
        
        # Set up logging
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logging handlers and formatters."""
        # Create logger
        self.logger = logging.getLogger('PlantDiseaseDetection')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler (detailed)
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (simpler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info("="*80)
        self.logger.info("PLANT DISEASE DETECTION SYSTEM - LOGGING INITIALIZED")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*80)
    
    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger
    
    def log_config(self, config: dict):
        """Log configuration settings."""
        self.logger.info("\nConfiguration:")
        self.logger.info("-" * 80)
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"  {subkey}: {subvalue}")
            else:
                self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 80)
    
    def log_step(self, step_name: str, step_number: int = None):
        """Log pipeline step start."""
        if step_number:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STEP {step_number}: {step_name.upper()}")
            self.logger.info(f"{'='*80}")
        else:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"{step_name.upper()}")
            self.logger.info(f"{'='*80}")
    
    def log_results(self, results: dict, title: str = "Results"):
        """Log structured results."""
        self.logger.info(f"\n{title}:")
        self.logger.info("-" * 80)
        for key, value in results.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"  {subkey}: {subvalue}")
            elif isinstance(value, float):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 80)
    
    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Log training/evaluation metrics."""
        prefix_str = f"{prefix} - " if prefix else ""
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                self.logger.info(f"{prefix_str}{metric_name}: {metric_value:.4f}")
            else:
                self.logger.info(f"{prefix_str}{metric_name}: {metric_value}")
    
    def log_model_summary(self, model_name: str, params: dict):
        """Log model summary information."""
        self.logger.info(f"\nModel: {model_name}")
        self.logger.info("-" * 40)
        for key, value in params.items():
            if isinstance(value, int) and value > 1000:
                self.logger.info(f"{key}: {value:,}")
            else:
                self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 40)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        self.logger.error(f"\n{'!'*80}")
        if context:
            self.logger.error(f"ERROR in {context}")
        self.logger.error(f"Exception: {type(error).__name__}")
        self.logger.error(f"Message: {str(error)}")
        self.logger.error(f"{'!'*80}")
    
    def log_dataset_info(self, dataset_info: dict):
        """Log dataset information."""
        self.logger.info("\nDataset Information:")
        self.logger.info("-" * 80)
        self.logger.info(f"Total Images: {dataset_info.get('total_images', 'N/A')}")
        self.logger.info(f"Number of Classes: {dataset_info.get('num_classes', 'N/A')}")
        self.logger.info(f"Train Images: {dataset_info.get('train_images', 'N/A')}")
        self.logger.info(f"Validation Images: {dataset_info.get('val_images', 'N/A')}")
        self.logger.info(f"Test Images: {dataset_info.get('test_images', 'N/A')}")
        
        if 'class_distribution' in dataset_info:
            self.logger.info("\nClass Distribution:")
            for class_name, count in sorted(dataset_info['class_distribution'].items()):
                self.logger.info(f"  {class_name}: {count} images")
        self.logger.info("-" * 80)
    
    def close(self):
        """Close logging handlers."""
        self.logger.info("\n" + "="*80)
        self.logger.info("LOGGING SESSION COMPLETE")
        self.logger.info(f"Full log saved to: {self.log_file}")
        self.logger.info("="*80)
        
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_global_logger = None


def get_logger(config_path: str = "config.yaml", log_name: str = None):
    """
    Get or create global logger instance.
    
    Args:
        config_path: Path to configuration file
        log_name: Name for log session
        
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = SystemLogger(config_path, log_name)
    
    return _global_logger.get_logger()


def log_step(step_name: str, step_number: int = None):
    """Quick access to log step."""
    if _global_logger:
        _global_logger.log_step(step_name, step_number)


def log_results(results: dict, title: str = "Results"):
    """Quick access to log results."""
    if _global_logger:
        _global_logger.log_results(results, title)


def log_metrics(metrics: dict, prefix: str = ""):
    """Quick access to log metrics."""
    if _global_logger:
        _global_logger.log_metrics(metrics, prefix)
