
"""Step 2: QuickDraw (bitmap) dataset loader.

Loads 28x28 grayscale bitmaps from pre-converted Parquet files.
Keeps them as single-channel (grayscale) for mobile efficiency.
Provides class filtering and stratified splits for training.

Note: Use scripts/download_quickdraw.py to convert HuggingFace data to Parquet format first.
"""
from typing import List, Optional, Dict, Tuple
import random
import json
import io
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

# Try to import logging config, fallback to basic logging if not available
try:
    from .logging_config import get_logger, log_and_print
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)
    def log_and_print(msg, logger_instance=None, level="INFO"): print(msg)


class QuickDrawDataset(Dataset):
    """
    QuickDraw bitmap dataset (28x28 grayscale) loaded from Parquet files.
    
    Key features:
    - Loads 28x28 grayscale bitmaps from efficient Parquet format
    - Keeps single channel (no RGB conversion) for mobile efficiency  
    - Supports class filtering for faster iteration
    - Applies transforms for training/validation
    
    Prerequisites:
    - Run scripts/download_quickdraw.py to convert HuggingFace data to Parquet first
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
        """
        Args:
            data_dir: Directory containing Parquet files and metadata.json
            classes: List of class names to include. If None, uses all available classes
            max_samples_per_class: Limit samples per class (for faster experimentation)  
            image_size: Target image size (224 for ViT, 256 for MobileViT)
            augment: Whether to apply data augmentation
            invert_colors: If True, invert colors to black-on-white for training (default: True)
            seed: Random seed for reproducible sampling
        """
        self.data_dir = Path(data_dir)
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please run: python scripts/download_quickdraw.py --num-classes 10 --samples-per-class 1000"
            )
        
        # Load metadata
        metadata_file = self.data_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        logger = get_logger(__name__)
        log_and_print(f"Loading QuickDraw dataset from: {data_dir}", logger_instance=logger)
        log_and_print(f"Available classes: {len(self.metadata['classes'])}", logger_instance=logger)
        
        # Determine which classes to use
        available_classes = self.metadata['classes']
        if classes is None:
            self.selected_classes = available_classes
        else:
            # Validate that requested classes exist
            missing_classes = [c for c in classes if c not in available_classes]
            if missing_classes:
                raise ValueError(f"Unknown class names: {missing_classes}. Available: {available_classes}")
            self.selected_classes = classes
            
        log_and_print(f"Using {len(self.selected_classes)} classes: {self.selected_classes[:10]}{'...' if len(self.selected_classes) > 10 else ''}", logger_instance=logger)
        
        # Create mapping from class names to contiguous IDs (0, 1, 2, ...)
        self.selected_classes = sorted(self.selected_classes)  # Keep consistent order
        self.class_to_id = {name: idx for idx, name in enumerate(self.selected_classes)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        
        # Load the data
        parquet_file = self.data_dir / "quickdraw_data.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet data file not found: {parquet_file}")
            
        log_and_print("Loading Parquet data...", logger_instance=logger)
        self.df = pd.read_parquet(parquet_file)
        
        # Filter for selected classes and apply sampling limits
        self.sample_indices = self._filter_and_sample_data(max_samples_per_class, seed)
        
        log_and_print(f"Total samples: {len(self.sample_indices)}", logger_instance=logger)
        
        # Setup transforms
        self.image_size = image_size
        self.invert_colors = invert_colors
        self.transforms = self._create_transforms(image_size, augment)
        
    def _filter_and_sample_data(self, max_per_class: Optional[int], seed: int) -> List[int]:
        """Filter for selected classes and apply sampling limits."""
        
        # Filter DataFrame for selected classes
        filtered_df = self.df[self.df['class_name'].isin(self.selected_classes)].copy()
        
        # Group by class and apply sampling limits
        if max_per_class is not None:
            random.seed(seed)
            sampled_indices = []
            
            for class_name in self.selected_classes:
                class_rows = filtered_df[filtered_df['class_name'] == class_name]
                
                if len(class_rows) > max_per_class:
                    # Sample randomly
                    sampled_rows = class_rows.sample(n=max_per_class, random_state=seed)
                else:
                    sampled_rows = class_rows
                
                sampled_indices.extend(sampled_rows.index.tolist())
                logger = get_logger(__name__)
                logger.info(f"  {class_name}: {len(sampled_rows)} samples")
                print(f"  {class_name}: {len(sampled_rows)} samples")
            
            return sampled_indices
        else:
            # Use all samples for selected classes
            for class_name in self.selected_classes:
                class_count = len(filtered_df[filtered_df['class_name'] == class_name])
                logger = get_logger(__name__)
                logger.info(f"  {class_name}: {class_count} samples")
                print(f"  {class_name}: {class_count} samples")
            
            return filtered_df.index.tolist()
        
    def _create_transforms(self, image_size: int, augment: bool):
        """Create transform pipeline for images."""
        
        # Base transforms: resize and convert to tensor
        base_transforms = [
            # Use NEAREST interpolation to keep doodles crisp
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # Converts to [0,1] and adds channel dim
        ]
        
        # Add augmentation for training
        if augment:
            aug_transforms = [
                # Mild geometric augmentations - don't distort the doodles too much
                transforms.RandomAffine(
                    degrees=10,           # Small rotation
                    translate=(0.1, 0.1), # Small translation  
                    scale=(0.9, 1.1),     # Small scale changes
                    fill=0                # Fill with black (background)
                ),
                transforms.RandomHorizontalFlip(p=0.1),  # Very low probability - most doodles aren't symmetric
                
                # Contrast/brightness adjustments
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ], p=0.3),
            ]
            base_transforms = base_transforms[:-1] + aug_transforms + [base_transforms[-1]]  # Insert before ToTensor
            
        return transforms.Compose(base_transforms)
        
    def __len__(self) -> int:
        return len(self.sample_indices)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Tensor of shape (1, H, W) - single channel grayscale
            label: Integer class ID (0 to num_classes-1)
        """
        # Get the actual DataFrame index
        df_idx = self.sample_indices[idx]
        row = self.df.iloc[df_idx]
        
        # Load image from bytes
        image_bytes = row['image_bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Ensure grayscale
        
        # Get class name and convert to our contiguous class ID
        class_name = row['class_name']
        label = self.class_to_id[class_name]
        
        # Apply transforms (resize, augment, convert to tensor)
        image = self.transforms(image)
        
        # Invert colors if requested (white-on-black → black-on-white for training)
        if self.invert_colors:
            image = 1.0 - image
        
        return image, label
    
    @property
    def num_classes(self) -> int:
        return len(self.selected_classes)


def create_stratified_split(
    dataset: QuickDrawDataset,
    train_samples_per_class: int = 4000,
    val_samples_per_class: int = 500,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Create stratified train/val split ensuring equal samples per class.
    
    Returns:
        train_indices: List of indices for training
        val_indices: List of indices for validation
    """
    
    # Group dataset indices by class
    class_groups = {class_id: [] for class_id in range(dataset.num_classes)}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_groups[label].append(idx)
    
    # Sample train/val indices from each class
    random.seed(seed)
    train_indices = []
    val_indices = []
    
    for class_id, indices in class_groups.items():
        random.shuffle(indices)
        
        # Take first N for training, next M for validation
        train_end = min(train_samples_per_class, len(indices))
        val_end = min(train_end + val_samples_per_class, len(indices))
        
        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        
        class_name = dataset.id_to_class[class_id]
        logger = get_logger(__name__)
        logger.info(f"  {class_name}: {train_end} train, {val_end - train_end} val samples")
        print(f"  {class_name}: {train_end} train, {val_end - train_end} val samples")
    
    return train_indices, val_indices


def create_dataloaders(
    data_dir: str = "data/quickdraw_parquet",
    classes: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_samples_per_class: int = 4000,
    val_samples_per_class: int = 500,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    invert_colors: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation dataloaders with metadata.
    
    Args:
        data_dir: Directory containing Parquet files and metadata.json
        classes: Specific class names to use
        num_classes: If classes=None, randomly sample this many classes
        train_samples_per_class: Training samples per class
        val_samples_per_class: Validation samples per class
        image_size: Target image size
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        invert_colors: If True, invert colors to black-on-white for training (default: True)
        seed: Random seed
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader  
        metadata: Dict with class mappings and info
    """
    
    # Load available classes from metadata
    metadata_file = Path(data_dir) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Please run: python scripts/download_quickdraw.py --num-classes 10 --samples-per-class 1000"
        )
    
    with open(metadata_file, 'r') as f:
        data_metadata = json.load(f)
    
    available_classes = data_metadata['classes']
    
    # Auto-select classes if num_classes is specified
    if classes is None and num_classes is not None:
        random.seed(seed)
        classes = sorted(random.sample(available_classes, min(num_classes, len(available_classes))))
        logger = get_logger(__name__)
        log_and_print(f"Auto-selected {num_classes} classes: {classes}", logger_instance=logger)
    
    # Create training dataset (with augmentation)
    train_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=train_samples_per_class + val_samples_per_class,
        image_size=image_size,
        augment=True,
        invert_colors=invert_colors,
        seed=seed
    )
    
    # Create validation dataset (no augmentation)
    val_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=train_dataset.selected_classes,  # Use same classes as train
        max_samples_per_class=train_samples_per_class + val_samples_per_class,
        image_size=image_size,
        augment=False,  # No augmentation for validation
        invert_colors=invert_colors,
        seed=seed
    )
    
    # Create stratified split
    logger = get_logger(__name__)
    log_and_print("\nCreating stratified train/val split...", logger_instance=logger)
    train_indices, val_indices = create_stratified_split(
        train_dataset, train_samples_per_class, val_samples_per_class, seed
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
        pin_memory=True,
        drop_last=True  # For consistent batch sizes
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
        'class_to_id': train_dataset.class_to_id,
        'id_to_class': train_dataset.id_to_class,
        'selected_classes': train_dataset.selected_classes,
        'num_classes': train_dataset.num_classes,
        'image_size': image_size,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices)
    }
    
    log_and_print(f"\nDataloaders created:", logger_instance=logger)
    log_and_print(f"  Training: {len(train_loader)} batches ({len(train_indices)} samples)", logger_instance=logger)
    log_and_print(f"  Validation: {len(val_loader)} batches ({len(val_indices)} samples)", logger_instance=logger)
    
    return train_loader, val_loader, metadata


def get_all_class_names(data_dir: str = "data/quickdraw_parquet") -> List[str]:
    """Get list of all available QuickDraw class names from downloaded data."""
    metadata_file = Path(data_dir) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Please run: python scripts/download_quickdraw.py to download data first"
        )
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata['classes']
