"""
Virtual splits: Deterministic, flexible train/val/calib splitting without physical files.

This approach provides:
- Flexibility: Any class combination, any sample count
- Speed: No file I/O, direct index computation
- Reproducibility: Same seed = same splits always
- Memory efficiency: No need to store split files
"""

from __future__ import annotations
import random
from typing import List, Tuple, Dict, Optional
import hashlib


class VirtualSplitter:
    """
    Deterministic virtual splitting that generates consistent train/val/calib indices
    for any class combination without storing physical split files.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def _get_class_seed(self, class_name: str, split_type: str) -> int:
        """Generate deterministic seed for a specific class and split type."""
        # Create deterministic seed from class name, split type, and base seed
        hash_input = f"{class_name}_{split_type}_{self.seed}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        return hash_value % (2**31)  # Keep within int32 range
    
    def create_virtual_split(
        self,
        class_indices: Dict[str, List[int]],
        train_samples_per_class: int,
        val_samples_per_class: int,
        calib_samples_per_class: int
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Create deterministic train/val/calib split for given classes.
        
        Args:
            class_indices: Dict mapping class names to their available indices
            train_samples_per_class: Training samples per class
            val_samples_per_class: Validation samples per class  
            calib_samples_per_class: Calibration samples per class
            
        Returns:
            (train_indices, val_indices, calib_indices)
        """
        
        train_indices = []
        val_indices = []
        calib_indices = []
        
        for class_name, indices in class_indices.items():
            # Use class-specific seed for deterministic but independent sampling
            class_seed = self._get_class_seed(class_name, "split")
            rng = random.Random(class_seed)
            
            # Shuffle indices deterministically
            class_indices_copy = indices.copy()
            rng.shuffle(class_indices_copy)
            
            # Allocate samples: train -> val -> calib
            total_needed = train_samples_per_class + val_samples_per_class + calib_samples_per_class
            available = min(len(class_indices_copy), total_needed)
            
            # Calculate actual allocation
            train_end = min(train_samples_per_class, available)
            val_end = min(train_end + val_samples_per_class, available)
            calib_end = min(val_end + calib_samples_per_class, available)
            
            # Split indices
            train_indices.extend(class_indices_copy[:train_end])
            val_indices.extend(class_indices_copy[train_end:val_end])
            calib_indices.extend(class_indices_copy[val_end:calib_end])
        
        return train_indices, val_indices, calib_indices
    
    def auto_compute_calib_samples(self, num_classes: int, target_total: int = 2048) -> int:
        """
        Auto-compute calibration samples per class following README guidance.
        
        Args:
            num_classes: Number of classes
            target_total: Target total calibration samples
            
        Returns:
            Calibration samples per class
        """
        calib_per_class = max(8, min(200, target_total // num_classes))
        
        # Apply README scaling rules
        if num_classes >= 300:  # All classes case
            calib_per_class = max(6, min(12, target_total // num_classes))
        elif num_classes >= 40:  # Medium class count (40-299)
            calib_per_class = max(40, min(80, target_total // num_classes))
        elif num_classes <= 10:  # Small class count
            calib_per_class = max(100, min(200, target_total // num_classes))
        
        return calib_per_class


def create_flexible_dataloaders(
    data_dir: str = "data/quickdraw_parquet",
    classes: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_samples_per_class: int = 1000,
    val_samples_per_class: int = 200,
    calib_samples_per_class: Optional[int] = None,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    invert_colors: bool = True,
    seed: int = 42,
    include_calibration: bool = False
) -> Tuple[...]:
    """
    Create flexible dataloaders using virtual splits.
    
    This approach:
    1. Loads only the required classes
    2. Computes splits deterministically on-the-fly
    3. No physical split files needed
    4. Same seed = same splits always
    5. Works with any class combination
    
    Returns:
        If include_calibration=False: (train_loader, val_loader, metadata)
        If include_calibration=True: (train_loader, val_loader, calib_loader, metadata)
    """
    
    from data import QuickDrawDataset, get_all_class_names
    from logging_config import get_logger, log_and_print
    from torch.utils.data import DataLoader
    import torch
    import json
    from pathlib import Path
    
    # Load available classes
    metadata_file = Path(data_dir) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        data_metadata = json.load(f)
    available_classes = data_metadata['classes']
    
    # Auto-select classes if needed
    if classes is None and num_classes is not None:
        random.seed(seed)
        classes = sorted(random.sample(available_classes, min(num_classes, len(available_classes))))
    
    # Auto-compute calibration samples
    splitter = VirtualSplitter(seed)
    if calib_samples_per_class is None:
        calib_samples_per_class = splitter.auto_compute_calib_samples(len(classes))
    
    logger = get_logger(__name__)
    log_and_print(f"Virtual split config:", logger_instance=logger)
    log_and_print(f"  Classes: {len(classes)} ({classes[:3]}{'...' if len(classes) > 3 else ''})", logger_instance=logger)
    log_and_print(f"  Train: {train_samples_per_class}/class, Val: {val_samples_per_class}/class", logger_instance=logger)
    if include_calibration:
        log_and_print(f"  Calib: {calib_samples_per_class}/class ({calib_samples_per_class * len(classes)} total)", logger_instance=logger)
    
    # Calculate max samples needed per class
    max_samples = train_samples_per_class + val_samples_per_class
    if include_calibration:
        max_samples += calib_samples_per_class
    
    # Create dataset to get class indices
    dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=max_samples,
        image_size=image_size,
        augment=False,  # We'll handle augmentation per split
        invert_colors=invert_colors,
        seed=seed
    )
    
    # Group indices by class
    class_indices = {class_name: [] for class_name in dataset.selected_classes}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_name = dataset.id_to_class[label]
        class_indices[class_name].append(idx)
    
    # Create virtual split
    if include_calibration:
        train_indices, val_indices, calib_indices = splitter.create_virtual_split(
            class_indices, train_samples_per_class, val_samples_per_class, calib_samples_per_class
        )
    else:
        train_indices, val_indices, _ = splitter.create_virtual_split(
            class_indices, train_samples_per_class, val_samples_per_class, 0
        )
    
    # Create datasets with appropriate augmentation
    train_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=max_samples,
        image_size=image_size,
        augment=True,  # Augmentation for training
        invert_colors=invert_colors,
        seed=seed
    )
    
    val_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=max_samples,
        image_size=image_size,
        augment=False,  # No augmentation for validation
        invert_colors=invert_colors,
        seed=seed
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create metadata
    metadata = {
        'class_to_id': dataset.class_to_id,
        'id_to_class': dataset.id_to_class,
        'selected_classes': dataset.selected_classes,
        'num_classes': dataset.num_classes,
        'image_size': image_size,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices)
    }
    
    if include_calibration:
        calib_dataset = QuickDrawDataset(
            data_dir=data_dir,
            classes=classes,
            max_samples_per_class=max_samples,
            image_size=image_size,
            augment=False,  # No augmentation for calibration
            invert_colors=invert_colors,
            seed=seed
        )
        
        calib_subset = torch.utils.data.Subset(calib_dataset, calib_indices)
        calib_loader = DataLoader(
            calib_subset,
            batch_size=batch_size,
            shuffle=False,  # Deterministic order for calibration
            num_workers=num_workers,
            pin_memory=True
        )
        
        metadata['calib_samples'] = len(calib_indices)
        
        log_and_print(f"Virtual dataloaders created:", logger_instance=logger)
        log_and_print(f"  Train: {len(train_loader)} batches ({len(train_indices)} samples)", logger_instance=logger)
        log_and_print(f"  Val: {len(val_loader)} batches ({len(val_indices)} samples)", logger_instance=logger)
        log_and_print(f"  Calib: {len(calib_loader)} batches ({len(calib_indices)} samples)", logger_instance=logger)
        
        return train_loader, val_loader, calib_loader, metadata
    else:
        log_and_print(f"Virtual dataloaders created:", logger_instance=logger)
        log_and_print(f"  Train: {len(train_loader)} batches ({len(train_indices)} samples)", logger_instance=logger)
        log_and_print(f"  Val: {len(val_loader)} batches ({len(val_indices)} samples)", logger_instance=logger)
        
        return train_loader, val_loader, metadata
