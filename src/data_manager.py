"""
Data Management Module

Handles structured data organization, versioning, and metadata tracking.
"""

import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np


class DataManager:
    """Manages data versions, metadata, and organization."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataManager.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = Path(self.config['data']['processed_data_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = self.processed_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Create versions directory
        self.versions_dir = self.processed_dir / 'versions'
        self.versions_dir.mkdir(exist_ok=True)
    
    def save_dataset_metadata(self, **kwargs) -> str:
        """
        Save dataset metadata with timestamp.
        
        Args:
            **kwargs: Metadata fields to save
            
        Returns:
            Path to saved metadata file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metadata = {
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'config': self.config,
            **kwargs
        }
        
        # Save main metadata
        metadata_file = self.metadata_dir / f'dataset_metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save as latest
        latest_file = self.metadata_dir / 'latest_metadata.json'
        with open(latest_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_file}")
        return str(metadata_file)
    
    def save_training_session(
        self,
        session_name: str,
        models: Dict,
        results: Dict,
        config: Dict
    ) -> str:
        """
        Save complete training session data.
        
        Args:
            session_name: Name of the training session
            models: Dictionary of model names and paths
            results: Training results
            config: Configuration used
            
        Returns:
            Path to session directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.versions_dir / f"{session_name}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session metadata
        session_metadata = {
            'session_name': session_name,
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'config': config,
            'models': models,
            'results': results
        }
        
        with open(session_dir / 'session_metadata.json', 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        # Save results separately
        with open(session_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save config
        with open(session_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        print(f"Training session saved to: {session_dir}")
        return str(session_dir)
    
    def get_dataset_stats(self) -> Dict:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'files': {}
        }
        
        # Check for data files
        for file_pattern in ['X_train.npy', 'X_val.npy', 'X_test.npy', 
                             'y_train.npy', 'y_val.npy', 'y_test.npy',
                             'features_train.npy', 'features_val.npy', 'features_test.npy']:
            file_path = self.processed_dir / file_pattern
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                data = np.load(file_path)
                stats['files'][file_pattern] = {
                    'exists': True,
                    'size_mb': round(size_mb, 2),
                    'shape': list(data.shape),
                    'dtype': str(data.dtype)
                }
            else:
                stats['files'][file_pattern] = {'exists': False}
        
        # Load class mapping if exists
        class_mapping_file = self.processed_dir / 'class_mapping.json'
        if class_mapping_file.exists():
            with open(class_mapping_file, 'r') as f:
                class_mapping = json.load(f)
            stats['num_classes'] = len(class_mapping)
            stats['classes'] = sorted(class_mapping.keys())
        
        # Load split info if exists
        split_info_file = self.processed_dir / 'split_info.json'
        if split_info_file.exists():
            with open(split_info_file, 'r') as f:
                split_info = json.load(f)
            
            train_counts = {k: len(v) for k, v in split_info['train'].items()}
            val_counts = {k: len(v) for k, v in split_info['validation'].items()}
            test_counts = {k: len(v) for k, v in split_info['test'].items()}
            
            stats['split_distribution'] = {
                'train': train_counts,
                'validation': val_counts,
                'test': test_counts,
                'train_total': sum(train_counts.values()),
                'val_total': sum(val_counts.values()),
                'test_total': sum(test_counts.values())
            }
        
        # Save stats
        stats_file = self.metadata_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def create_experiment_report(
        self,
        experiment_name: str,
        results: Dict,
        metadata: Dict
    ) -> str:
        """
        Create comprehensive experiment report.
        
        Args:
            experiment_name: Name of the experiment
            results: Experimental results
            metadata: Additional metadata
            
        Returns:
            Path to report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.config['output']['reports_dir'])
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'results': results,
            'metadata': metadata,
            'config': self.config
        }
        
        # Save JSON report
        json_report = report_dir / f"{experiment_name}_{timestamp}.json"
        with open(json_report, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown report
        md_report = report_dir / f"{experiment_name}_{timestamp}.md"
        self._create_markdown_report(report, md_report)
        
        print(f"Experiment report saved to:")
        print(f"  JSON: {json_report}")
        print(f"  Markdown: {md_report}")
        
        return str(json_report)
    
    def _create_markdown_report(self, report: Dict, output_path: Path):
        """Create markdown version of experiment report."""
        md_content = f"""# {report['experiment_name']}

**Date**: {report['date']}  
**Timestamp**: {report['timestamp']}

## Configuration

```yaml
"""
        # Add key config sections
        for key in ['data', 'preprocessing', 'deep_learning', 'classical_ml']:
            if key in report['config']:
                md_content += f"{key}:\n"
                for k, v in report['config'][key].items():
                    md_content += f"  {k}: {v}\n"
        
        md_content += "```\n\n## Results\n\n"
        
        # Add results
        if 'results' in report:
            for model_name, model_results in report['results'].items():
                if isinstance(model_results, dict) and 'accuracy' in model_results:
                    md_content += f"\n### {model_name}\n\n"
                    for metric, value in model_results.items():
                        if isinstance(value, float):
                            md_content += f"- **{metric}**: {value:.4f}\n"
                        elif not isinstance(value, dict):
                            md_content += f"- **{metric}**: {value}\n"
        
        md_content += "\n## Metadata\n\n"
        
        if 'metadata' in report:
            for key, value in report['metadata'].items():
                if not isinstance(value, (dict, list)):
                    md_content += f"- **{key}**: {value}\n"
        
        md_content += f"\n---\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def cleanup_old_versions(self, keep_last: int = 5):
        """
        Clean up old version directories, keeping only recent ones.
        
        Args:
            keep_last: Number of recent versions to keep
        """
        version_dirs = sorted(self.versions_dir.glob("*_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(version_dirs) > keep_last:
            print(f"\nCleaning up old versions (keeping last {keep_last})...")
            for old_dir in version_dirs[keep_last:]:
                shutil.rmtree(old_dir)
                print(f"Removed: {old_dir.name}")
