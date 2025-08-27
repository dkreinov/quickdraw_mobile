"""
Memory-efficient lazy loading dataset that doesn't load all data into memory.

Instead of loading all parquet data into memory, this approach:
1. Scans parquet files to get metadata (class counts, file locations)
2. Creates virtual index mapping (class_name, sample_idx) → (file_path, row_idx)
3. Loads individual samples on-demand during __getitem__
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import io

from logging_config import get_logger, log_and_print


class LazyQuickDrawDataset(Dataset):
    """
    Memory-efficient QuickDraw dataset that loads samples on-demand.
    
    Benefits:
    - Constant memory usage regardless of dataset size
    - Fast initialization (just scans metadata)
    - Works with datasets too large for memory
    - Same interface as regular QuickDrawDataset
    """
    
    def __init__(
        self,
        data_dir: str = "data/quickdraw_parquet",
        classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        image_size: int = 224,
        augment: bool = True,
        invert_colors: bool = True,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.invert_colors = invert_colors
        self.seed = seed
        
        logger = get_logger(__name__)
        
        # Load metadata
        metadata_file = self.data_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Determine classes to use
        available_classes = self.metadata['classes']
        if classes is None:
            self.selected_classes = available_classes
        else:
            missing_classes = [c for c in classes if c not in available_classes]
            if missing_classes:
                raise ValueError(f"Unknown class names: {missing_classes}")
            self.selected_classes = sorted(classes)
        
        # Create class mappings
        self.class_to_id = {name: idx for idx, name in enumerate(self.selected_classes)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        
        log_and_print(f"Lazy loading {len(self.selected_classes)} classes", logger_instance=logger)
        
        # Create virtual index mapping (doesn't load actual data)
        self._create_virtual_index(max_samples_per_class, seed, logger)
        
        # Setup transforms
        from data import QuickDrawDataset
        temp_dataset = QuickDrawDataset.__new__(QuickDrawDataset)
        self.transforms = temp_dataset._create_transforms(image_size, augment)
        
        log_and_print(f"Lazy dataset ready: {len(self.virtual_indices)} samples", logger_instance=logger)
    
    def _create_virtual_index(self, max_samples_per_class: Optional[int], seed: int, logger):
        """Create virtual index mapping without loading actual data."""
        
        # Check for per-class format
        per_class_dir = self.data_dir / "per_class"
        per_class_metadata_file = per_class_dir / "metadata.json"
        
        if per_class_dir.exists() and per_class_metadata_file.exists():
            self._create_per_class_virtual_index(per_class_dir, per_class_metadata_file, max_samples_per_class, seed, logger)
        else:
            # Fallback to monolithic
            parquet_file = self.data_dir / "quickdraw_data.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"No data found: {parquet_file}")
            self._create_monolithic_virtual_index(parquet_file, max_samples_per_class, seed, logger)
    
    def _create_per_class_virtual_index(self, per_class_dir: Path, metadata_file: Path, max_samples_per_class: Optional[int], seed: int, logger):
        """Create virtual index for per-class format."""
        
        with open(metadata_file, 'r') as f:
            per_class_metadata = json.load(f)
        
        self.virtual_indices = []  # List of (file_path, row_idx, class_name)
        
        for class_name in self.selected_classes:
            if class_name not in per_class_metadata['class_files']:
                log_and_print(f"  Warning: No file for class '{class_name}'", logger_instance=logger)
                continue
            
            class_info = per_class_metadata['class_files'][class_name]
            class_file = per_class_dir / class_info['filename']
            
            if not class_file.exists():
                log_and_print(f"  Warning: Missing file: {class_file}", logger_instance=logger)
                continue
            
            # Get sample count without loading data (read just metadata)
            parquet_file = pd.read_parquet(class_file, columns=[])  # Empty columns = just get row count
            total_samples = len(parquet_file)
            
            # Determine how many samples to use
            if max_samples_per_class is not None:
                num_samples = min(max_samples_per_class, total_samples)
            else:
                num_samples = total_samples
            
            # Create virtual indices for this class
            if num_samples < total_samples:
                # Sample specific rows
                random.seed(seed + hash(class_name) % 1000)  # Class-specific seed
                selected_rows = sorted(random.sample(range(total_samples), num_samples))
            else:
                # Use all rows
                selected_rows = list(range(total_samples))
            
            # Add to virtual index
            for row_idx in selected_rows:
                self.virtual_indices.append((class_file, row_idx, class_name))
            
            log_and_print(f"    {class_name}: {num_samples}/{total_samples} samples", logger_instance=logger)
        
        # Shuffle virtual indices for better mixing
        random.seed(seed)
        random.shuffle(self.virtual_indices)
    
    def _create_monolithic_virtual_index(self, parquet_file: Path, max_samples_per_class: Optional[int], seed: int, logger):
        """Create virtual index for monolithic format."""
        
        # Read just the class_name column to get class distribution
        df_classes = pd.read_parquet(parquet_file, columns=['class_name'])
        
        self.virtual_indices = []
        
        for class_name in self.selected_classes:
            # Find all rows for this class
            class_mask = df_classes['class_name'] == class_name
            class_row_indices = df_classes.index[class_mask].tolist()
            
            # Sample if needed
            if max_samples_per_class is not None and len(class_row_indices) > max_samples_per_class:
                random.seed(seed + hash(class_name) % 1000)
                class_row_indices = sorted(random.sample(class_row_indices, max_samples_per_class))
            
            # Add to virtual index
            for row_idx in class_row_indices:
                self.virtual_indices.append((parquet_file, row_idx, class_name))
            
            log_and_print(f"    {class_name}: {len(class_row_indices)} samples", logger_instance=logger)
        
        # Shuffle for better mixing
        random.seed(seed)
        random.shuffle(self.virtual_indices)
    
    def __len__(self) -> int:
        return len(self.virtual_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load sample on-demand."""
        file_path, row_idx, class_name = self.virtual_indices[idx]
        
        # Load just this one row from parquet
        df_row = pd.read_parquet(file_path, columns=['image_bytes', 'class_name'])
        row = df_row.iloc[row_idx]
        
        # Load image
        image_bytes = row['image_bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Get label
        label = self.class_to_id[class_name]
        
        # Apply transforms
        image = self.transforms(image)
        
        # Invert colors if needed
        if self.invert_colors:
            image = 1.0 - image
        
        return image, label
    
    @property
    def num_classes(self) -> int:
        return len(self.selected_classes)


def create_lazy_dataloaders(
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
    Create dataloaders using lazy loading (memory-efficient).
    
    This approach:
    1. Scans data files to create virtual indices
    2. Loads samples on-demand during training
    3. Uses constant memory regardless of dataset size
    4. Same interface as regular dataloaders
    """
    
    from virtual_splits import VirtualSplitter
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
    log_and_print(f"Lazy virtual split config:", logger_instance=logger)
    log_and_print(f"  Classes: {len(classes)} ({classes[:3]}{'...' if len(classes) > 3 else ''})", logger_instance=logger)
    log_and_print(f"  Train: {train_samples_per_class}/class, Val: {val_samples_per_class}/class", logger_instance=logger)
    if include_calibration:
        log_and_print(f"  Calib: {calib_samples_per_class}/class ({calib_samples_per_class * len(classes)} total)", logger_instance=logger)
    
    # Calculate max samples needed per class
    max_samples = train_samples_per_class + val_samples_per_class
    if include_calibration:
        max_samples += calib_samples_per_class
    
    # Create lazy dataset (doesn't load all data into memory)
    dataset = LazyQuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=max_samples,
        image_size=image_size,
        augment=False,  # We'll handle augmentation per split
        invert_colors=invert_colors,
        seed=seed
    )
    
    # Group indices by class (virtual - no actual data loading)
    class_indices = {class_name: [] for class_name in dataset.selected_classes}
    for idx, (_, _, class_name) in enumerate(dataset.virtual_indices):
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
    train_dataset = LazyQuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=max_samples,
        image_size=image_size,
        augment=True,  # Augmentation for training
        invert_colors=invert_colors,
        seed=seed
    )
    
    val_dataset = LazyQuickDrawDataset(
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
        calib_dataset = LazyQuickDrawDataset(
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
        
        log_and_print(f"Lazy dataloaders created:", logger_instance=logger)
        log_and_print(f"  Train: {len(train_loader)} batches ({len(train_indices)} samples)", logger_instance=logger)
        log_and_print(f"  Val: {len(val_loader)} batches ({len(val_indices)} samples)", logger_instance=logger)
        log_and_print(f"  Calib: {len(calib_loader)} batches ({len(calib_indices)} samples)", logger_instance=logger)
        
        return train_loader, val_loader, calib_loader, metadata
    else:
        log_and_print(f"Lazy dataloaders created:", logger_instance=logger)
        log_and_print(f"  Train: {len(train_loader)} batches ({len(train_indices)} samples)", logger_instance=logger)
        log_and_print(f"  Val: {len(val_loader)} batches ({len(val_indices)} samples)", logger_instance=logger)
        
        return train_loader, val_loader, metadata

