"""
Comprehensive evaluation utilities for QuickDraw models.

Provides:
- Checkpoint loading
- Evaluation metrics (Top-1, Top-5, per-class accuracy)
- Confusion matrix generation
- Latency benchmarking
- Qualitative analysis (most confused pairs, sample predictions)
- Calibration data generation
"""

from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

try:
    from .logging_config import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class EvaluationResults:
    """Container for all evaluation results."""
    # Core metrics
    top1_accuracy: float
    top5_accuracy: float
    avg_loss: float
    
    # Per-class metrics
    per_class_accuracy: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    
    # Confusion analysis
    confusion_matrix: np.ndarray
    most_confused_pairs: List[Tuple[str, str, int]]  # (true_class, pred_class, count)
    
    # Model info
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    
    # Latency (if measured)
    latency_cpu_ms: Optional[Dict[str, float]] = None  # mean, p50, p95
    latency_gpu_ms: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON compatibility
        if isinstance(result['confusion_matrix'], np.ndarray):
            result['confusion_matrix'] = result['confusion_matrix'].tolist()
        return result
    
    def save_json(self, path: str | Path) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Evaluation results saved to {path}")


class ModelEvaluator:
    """Comprehensive model evaluation toolkit."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f"class_{i}" for i in range(self._get_num_classes())]
        
    def _get_num_classes(self) -> int:
        """Infer number of classes from model."""
        # Look for the final classification layer
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                num_classes = module.out_features
        return num_classes
    
    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """Load model checkpoint and return metadata."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the model state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Extract metadata
        metadata = {}
        if isinstance(checkpoint, dict):
            metadata = {k: v for k, v in checkpoint.items() 
                       if k != 'model_state_dict' and not k.endswith('_state_dict')}
        
        logger.info(f"Checkpoint loaded successfully. Metadata keys: {list(metadata.keys())}")
        return metadata
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        compute_confusion: bool = True,
        save_predictions: bool = False
    ) -> EvaluationResults:
        """Comprehensive evaluation on a dataset."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_logits = []
        total_loss = 0.0
        num_samples = 0
        
        # For sample predictions
        sample_predictions = []
        
        logger.info(f"Evaluating on {len(dataloader)} batches...")
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = criterion(logits, targets)
                
                # Accumulate
                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)
                
                # Store predictions
                predictions = logits.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                
                # Save some sample predictions for qualitative analysis
                if save_predictions and len(sample_predictions) < 20:
                    for i in range(min(5, len(predictions))):
                        sample_predictions.append({
                            'true_class': self.class_names[targets[i].item()],
                            'pred_class': self.class_names[predictions[i].item()],
                            'confidence': torch.softmax(logits[i], dim=0).max().item(),
                            'correct': targets[i].item() == predictions[i].item()
                        })
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_logits = np.array(all_logits)
        
        avg_loss = total_loss / num_samples
        
        logger.info(f"Evaluation complete: {num_samples} samples processed")
        
        # Compute metrics
        results = self._compute_metrics(
            all_targets, all_predictions, all_logits, avg_loss, compute_confusion
        )
        
        # Store sample predictions
        if save_predictions:
            results.sample_predictions = sample_predictions
        
        return results
    
    def _compute_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        logits: np.ndarray,
        avg_loss: float,
        compute_confusion: bool = True
    ) -> EvaluationResults:
        """Compute all evaluation metrics."""
        
        # Top-1 accuracy
        top1_accuracy = (predictions == targets).mean() * 100
        
        # Top-5 accuracy
        top5_predictions = np.argsort(logits, axis=1)[:, -5:]
        top5_accuracy = np.mean([target in top5_pred for target, top5_pred 
                                in zip(targets, top5_predictions)]) * 100
        
        # Per-class metrics using sklearn
        from sklearn.metrics import classification_report
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        per_class_accuracy = {}
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        
        for class_name in self.class_names:
            if class_name in report:
                per_class_precision[class_name] = report[class_name]['precision']
                per_class_recall[class_name] = report[class_name]['recall']
                per_class_f1[class_name] = report[class_name]['f1-score']
                
                # Calculate per-class accuracy manually
                class_idx = self.class_names.index(class_name)
                class_mask = targets == class_idx
                if class_mask.sum() > 0:
                    per_class_accuracy[class_name] = (
                        predictions[class_mask] == targets[class_mask]
                    ).mean() * 100
                else:
                    per_class_accuracy[class_name] = 0.0
        
        # Confusion matrix and most confused pairs
        conf_matrix = None
        most_confused_pairs = []
        
        if compute_confusion:
            conf_matrix = confusion_matrix(targets, predictions)
            most_confused_pairs = self._find_most_confused_pairs(conf_matrix)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        
        return EvaluationResults(
            top1_accuracy=top1_accuracy,
            top5_accuracy=top5_accuracy,
            avg_loss=avg_loss,
            per_class_accuracy=per_class_accuracy,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            confusion_matrix=conf_matrix,
            most_confused_pairs=most_confused_pairs,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb
        )
    
    def _find_most_confused_pairs(
        self, 
        conf_matrix: np.ndarray, 
        top_k: int = 10
    ) -> List[Tuple[str, str, int]]:
        """Find the most confused class pairs from confusion matrix."""
        confused_pairs = []
        
        # Look at off-diagonal elements (misclassifications)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                if i != j and conf_matrix[i, j] > 0:  # Misclassification
                    confused_pairs.append((
                        self.class_names[i],  # true class
                        self.class_names[j],  # predicted class
                        int(conf_matrix[i, j])  # count
                    ))
        
        # Sort by count (descending) and return top_k
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        return confused_pairs[:top_k]
    
    def benchmark_latency(
        self,
        input_shape: Tuple[int, ...] = (1, 1, 224, 224),
        batch_sizes: List[int] = [1, 8],
        warmup_iters: int = 20,
        benchmark_iters: int = 100,
        devices: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark model latency on different devices and batch sizes."""
        if devices is None:
            devices = ['cpu']
            if torch.cuda.is_available():
                devices.append('cuda')
        
        results = {}
        
        for device in devices:
            device_results = {}
            
            # Move model to device
            model_device = self.model.to(device)
            model_device.eval()
            
            for batch_size in batch_sizes:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape[1:]).to(device)
                
                # Warmup
                logger.info(f"Warming up {device} with batch_size={batch_size}...")
                with torch.no_grad():
                    for _ in range(warmup_iters):
                        _ = model_device(dummy_input)
                
                # Synchronize for accurate timing
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                logger.info(f"Benchmarking {device} with batch_size={batch_size}...")
                times = []
                
                with torch.no_grad():
                    for _ in range(benchmark_iters):
                        start_time = time.perf_counter()
                        _ = model_device(dummy_input)
                        
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate statistics (per image)
                times = np.array(times) / batch_size
                device_results[f'batch_{batch_size}'] = {
                    'mean_ms': float(np.mean(times)),
                    'p50_ms': float(np.percentile(times, 50)),
                    'p95_ms': float(np.percentile(times, 95)),
                    'std_ms': float(np.std(times))
                }
                
                logger.info(f"{device} batch_{batch_size}: "
                           f"mean={np.mean(times):.2f}ms, "
                           f"p50={np.percentile(times, 50):.2f}ms, "
                           f"p95={np.percentile(times, 95):.2f}ms")
            
            results[device] = device_results
        
        # Move model back to original device
        self.model.to(self.device)
        
        return results
    
    def create_calibration_dataset(
        self,
        dataloader: DataLoader,
        num_samples: int = 2048,
        save_path: Optional[str | Path] = None
    ) -> torch.Tensor:
        """Create calibration dataset for quantization."""
        logger.info(f"Creating calibration dataset with {num_samples} samples...")
        
        calibration_data = []
        samples_collected = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for images, _ in dataloader:
                batch_size = images.size(0)
                samples_needed = min(batch_size, num_samples - samples_collected)
                
                calibration_data.append(images[:samples_needed])
                samples_collected += samples_needed
                
                if samples_collected >= num_samples:
                    break
        
        calibration_tensor = torch.cat(calibration_data, dim=0)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(calibration_tensor, save_path)
            logger.info(f"Calibration dataset saved to {save_path}")
        
        logger.info(f"Calibration dataset created: {calibration_tensor.shape}")
        return calibration_tensor


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=figsize)
    
    if normalize:
        # Normalize by true class (rows)
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Handle division by zero
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
        matrix_to_plot = conf_matrix_norm
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
        matrix_to_plot = conf_matrix
    
    sns.heatmap(
        matrix_to_plot,
        annot=len(class_names) <= 20,  # Only annotate if not too many classes
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def save_most_confused_pairs(
    confused_pairs: List[Tuple[str, str, int]],
    save_path: str | Path
) -> None:
    """Save most confused pairs to text file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Most Confused Class Pairs\n")
        f.write("=" * 50 + "\n\n")
        f.write("Format: True Class -> Predicted Class (Count)\n\n")
        
        for i, (true_class, pred_class, count) in enumerate(confused_pairs, 1):
            f.write(f"{i:2d}. {true_class} -> {pred_class} ({count})\n")
    
    logger.info(f"Most confused pairs saved to {save_path}")


