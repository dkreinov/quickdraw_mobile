
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

# PyArrow imports for faster parquet operations
try:
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Concurrent loading imports
try:
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False

# Try to import logging config, fallback to basic logging if not available
try:
    from .logging_config import get_logger, log_and_print
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)
    def log_and_print(msg, logger_instance=None, level="INFO"): print(msg)

# Memory management imports
import psutil
import os


# ============================================================================
# Adaptive Memory Loading System
# ============================================================================

def get_available_memory() -> int:
    """Get available system memory in bytes."""
    return psutil.virtual_memory().available


def auto_compute_calib_samples(num_classes: int) -> int:
    """
    Auto-compute calibration samples per class based on README guidance.
    
    Target: ~2048 total calibration samples, distributed based on class count:
    - ≤10 classes: 100-200 samples/class
    - 11-39 classes: 40-80 samples/class  
    - 40-299 classes: 40-80 samples/class
    - ≥300 classes: 6-12 samples/class
    """
    target_total_calib = 2048
    calib_per_class = max(8, min(200, target_total_calib // num_classes))
    
    if num_classes >= 300:
        calib_per_class = max(6, min(12, target_total_calib // num_classes))
    elif num_classes >= 40:
        calib_per_class = max(40, min(80, target_total_calib // num_classes))
    elif num_classes <= 10:
        calib_per_class = max(100, min(200, target_total_calib // num_classes))
    
    return calib_per_class


def estimate_dataset_memory(
    num_classes: Optional[int] = None,
    train_samples_per_class: int = 1000,
    val_samples_per_class: int = 200,
    calib_samples_per_class: Optional[int] = None,
    image_size_bytes: int = 28 * 28 * 1  # 28x28 grayscale
) -> int:
    """
    Estimate memory requirements for a dataset configuration.
    
    Args:
        num_classes: Number of classes to load
        train_samples_per_class: Training samples per class
        val_samples_per_class: Validation samples per class  
        calib_samples_per_class: Calibration samples per class (auto-computed if None)
        image_size_bytes: Size of each image in bytes
        
    Returns:
        Estimated memory usage in bytes
    """
    if num_classes is None:
        raise ValueError("Must provide num_classes for memory estimation")
    
    if calib_samples_per_class is None:
        calib_samples_per_class = auto_compute_calib_samples(num_classes)
    
    total_samples_per_class = train_samples_per_class + val_samples_per_class + calib_samples_per_class
    total_samples = num_classes * total_samples_per_class
    
    # Estimate memory: images + labels + overhead
    image_memory = total_samples * image_size_bytes
    label_memory = total_samples * 8  # 8 bytes per label (int64)
    overhead = int(image_memory * 0.3)  # 30% overhead for PyTorch tensors, indices, etc.
    
    return image_memory + label_memory + overhead


def should_use_lazy_loading(
    num_classes: Optional[int] = None,
    train_samples_per_class: int = 1000,
    val_samples_per_class: int = 200,
    calib_samples_per_class: Optional[int] = None,
    memory_threshold: float = 0.6
) -> bool:
    """
    Decide whether to use lazy loading based on memory requirements.
    
    Args:
        num_classes: Number of classes to load
        train_samples_per_class: Training samples per class
        val_samples_per_class: Validation samples per class
        calib_samples_per_class: Calibration samples per class (auto-computed if None)
        memory_threshold: Use lazy loading if estimated memory > threshold * available memory
        
    Returns:
        True if should use lazy loading, False for in-memory loading
    """
    estimated_memory = estimate_dataset_memory(
        num_classes=num_classes,
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        calib_samples_per_class=calib_samples_per_class
    )
    
    available_memory = get_available_memory()
    threshold_memory = available_memory * memory_threshold
    
    return estimated_memory > threshold_memory


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
        
        # Load the data - auto-detect format
        self._load_data(max_samples_per_class, seed, logger)
        
        log_and_print(f"Total samples: {len(self.sample_indices)}", logger_instance=logger)
        
        # Setup transforms
        self.image_size = image_size
        self.invert_colors = invert_colors
        self.transforms = self._create_transforms(image_size, augment)
        
    def _load_data(self, max_samples_per_class: Optional[int], seed: int, logger):
        """Load data using auto-detected format (per-class preferred, monolithic fallback)."""
        
        # Check for per-class format first (now preferred for all scenarios)
        per_class_dir = self.data_dir / "per_class"
        per_class_metadata = per_class_dir / "metadata.json"
        
        if per_class_dir.exists() and per_class_metadata.exists():
            log_and_print("Using per-class Parquet format (recommended)...", logger_instance=logger)
            self._load_per_class_data(per_class_dir, max_samples_per_class, seed, logger)
        else:
            # Fall back to monolithic format
            log_and_print("Using monolithic Parquet format (fallback)...", logger_instance=logger)
            parquet_file = self.data_dir / "quickdraw_data.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(
                    f"No data found. Expected either:\n"
                    f"  - Per-class (recommended): {per_class_dir}/\n"
                    f"  - Monolithic (fallback): {parquet_file}\n"
                    f"Please run download script or split existing data.\n"
                    f"For best performance, use: python scripts/split_parquet_by_class.py --input-dir {self.data_dir}"
                )
            
            log_and_print("Using monolithic Parquet format (fallback - consider splitting for better performance)...", logger_instance=logger)
            self._load_monolithic_data(parquet_file, max_samples_per_class, seed, logger)
    
    def _load_per_class_data(self, per_class_dir: Path, max_samples_per_class: Optional[int], seed: int, logger):
        """Load data from per-class Parquet files with optional concurrent loading."""
        
        # Load per-class metadata to get file mappings
        with open(per_class_dir / "metadata.json", 'r') as f:
            per_class_metadata = json.load(f)
        
        # Determine whether to use concurrent loading
        use_concurrent = (CONCURRENT_AVAILABLE and 
                         len(self.selected_classes) > 3 and  # Worth it for 4+ classes
                         multiprocessing.cpu_count() > 1)   # Multiple cores available
        
        if use_concurrent:
            log_and_print(f"  Loading {len(self.selected_classes)} classes concurrently from per-class files...", logger_instance=logger)
            all_samples = self._load_per_class_concurrent(per_class_dir, per_class_metadata, max_samples_per_class, seed, logger)
        else:
            log_and_print(f"  Loading {len(self.selected_classes)} classes sequentially from per-class files...", logger_instance=logger)
            all_samples = self._load_per_class_sequential(per_class_dir, per_class_metadata, max_samples_per_class, seed, logger)
        
        if not all_samples:
            raise RuntimeError(f"No data loaded for selected classes: {self.selected_classes}")
        
        # Combine all class data
        self.df = pd.concat(all_samples, ignore_index=True)
        
        # Create sample indices (just 0 to len-1 since we loaded exactly what we need)
        self.sample_indices = list(range(len(self.df)))
        
        # Shuffle if we have multiple classes
        if len(self.selected_classes) > 1:
            random.seed(seed)
            random.shuffle(self.sample_indices)
        
        concurrent_msg = " (concurrent)" if use_concurrent else ""
        log_and_print(f"  Per-class loading complete{concurrent_msg}: {len(self.sample_indices)} samples", logger_instance=logger)
    
    def _load_per_class_sequential(self, per_class_dir: Path, per_class_metadata: dict, max_samples_per_class: Optional[int], seed: int, logger) -> List[pd.DataFrame]:
        """Load per-class files sequentially (original method)."""
        all_samples = []
        
        for class_name in self.selected_classes:
            if class_name not in per_class_metadata['class_files']:
                log_and_print(f"  Warning: No file found for class '{class_name}'", logger_instance=logger)
                continue
            
            # Load this class's parquet file
            class_info = per_class_metadata['class_files'][class_name]
            class_file = per_class_dir / class_info['filename']
            
            if not class_file.exists():
                log_and_print(f"  Warning: File missing for class '{class_name}': {class_file}", logger_instance=logger)
                continue
            
            class_df = pd.read_parquet(class_file)
            
            # Apply sampling if needed
            if max_samples_per_class is not None and len(class_df) > max_samples_per_class:
                class_df = class_df.sample(n=max_samples_per_class, random_state=seed)
            
            all_samples.append(class_df)
            log_and_print(f"    {class_name}: {len(class_df)} samples", logger_instance=logger)
        
        return all_samples
    
    def _load_per_class_concurrent(self, per_class_dir: Path, per_class_metadata: dict, max_samples_per_class: Optional[int], seed: int, logger) -> List[pd.DataFrame]:
        """Load per-class files concurrently for better performance."""
        
        def load_single_class(class_name: str) -> tuple:
            """Load a single class file - designed for concurrent execution."""
            try:
                if class_name not in per_class_metadata['class_files']:
                    return class_name, None, f"No file found for class '{class_name}'"
                
                class_info = per_class_metadata['class_files'][class_name]
                class_file = per_class_dir / class_info['filename']
                
                if not class_file.exists():
                    return class_name, None, f"File missing: {class_file}"
                
                class_df = pd.read_parquet(class_file)
                
                # Apply sampling if needed
                if max_samples_per_class is not None and len(class_df) > max_samples_per_class:
                    class_df = class_df.sample(n=max_samples_per_class, random_state=seed)
                
                return class_name, class_df, None
                
            except Exception as e:
                return class_name, None, f"Error loading {class_name}: {e}"
        
        # Use ThreadPoolExecutor for I/O-bound parquet reading
        # Use min(cpu_count, num_classes) to avoid over-threading
        max_workers = min(multiprocessing.cpu_count(), len(self.selected_classes), 8)  # Cap at 8 threads
        
        all_samples = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all loading tasks
            future_to_class = {
                executor.submit(load_single_class, class_name): class_name 
                for class_name in self.selected_classes
            }
            
            # Collect results as they complete
            for future in future_to_class:
                class_name, class_df, error = future.result()
                
                if error:
                    log_and_print(f"  Warning: {error}", logger_instance=logger)
                elif class_df is not None:
                    all_samples.append(class_df)
                    log_and_print(f"    {class_name}: {len(class_df)} samples", logger_instance=logger)
        
        return all_samples
    
    def _load_monolithic_data(self, parquet_file: Path, max_samples_per_class: Optional[int], seed: int, logger):
        """Load data from monolithic Parquet file with PyArrow optimization."""
        
        # Try PyArrow optimization first, fall back to pandas if not available
        if PYARROW_AVAILABLE and len(self.selected_classes) < len(self.metadata['classes']) * 0.5:
            # Use PyArrow for subset loading (< 50% of classes)
            log_and_print("Loading Parquet data with PyArrow optimization...", logger_instance=logger)
            self._load_monolithic_pyarrow(parquet_file, max_samples_per_class, seed, logger)
        else:
            # Use pandas for full dataset or when PyArrow not available
            log_and_print("Loading Parquet data with pandas...", logger_instance=logger)
            self._load_monolithic_pandas(parquet_file, max_samples_per_class, seed, logger)
    
    def _load_monolithic_pyarrow(self, parquet_file: Path, max_samples_per_class: Optional[int], seed: int, logger):
        """Load data using PyArrow for efficient subset filtering."""
        
        log_and_print("  Step 1: Filtering data with PyArrow...", logger_instance=logger)
        
        try:
            # Use PyArrow's built-in filtering - much more efficient
            # Create filter expression for selected classes
            filters = [('class_name', 'in', self.selected_classes)]
            
            # Read table with class filter applied at the file level
            table = pq.read_table(str(parquet_file), filters=filters)
            
            # Convert to pandas for further processing
            filtered_df = table.to_pandas()
            
            log_and_print(f"  PyArrow filtered to {len(filtered_df)} samples for selected classes", logger_instance=logger)
            
            # Apply per-class sampling if needed
            if max_samples_per_class is not None:
                log_and_print(f"  Step 2: Applying per-class sampling...", logger_instance=logger)
                sampled_dfs = []
                random.seed(seed)
                
                for class_name in self.selected_classes:
                    class_df = filtered_df[filtered_df['class_name'] == class_name]
                    
                    if len(class_df) > max_samples_per_class:
                        class_df = class_df.sample(n=max_samples_per_class, random_state=seed)
                    
                    sampled_dfs.append(class_df)
                    log_and_print(f"    {class_name}: {len(class_df)} samples", logger_instance=logger)
                
                self.df = pd.concat(sampled_dfs, ignore_index=True)
            else:
                self.df = filtered_df
                
                # Log samples per class
                for class_name in self.selected_classes:
                    class_count = len(filtered_df[filtered_df['class_name'] == class_name])
                    log_and_print(f"    {class_name}: {class_count} samples", logger_instance=logger)
            
            # Shuffle the final dataset
            if len(self.selected_classes) > 1:
                self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            # Create sample indices
            self.sample_indices = list(range(len(self.df)))
            
            log_and_print(f"PyArrow loading complete: {len(self.sample_indices)} samples", logger_instance=logger)
            
        except Exception as e:
            # Fall back to pandas if PyArrow filtering fails
            log_and_print(f"  PyArrow filtering failed ({e}), falling back to pandas...", logger_instance=logger)
            self._load_monolithic_pandas(parquet_file, max_samples_per_class, seed, logger)
    
    def _load_monolithic_pandas(self, parquet_file: Path, max_samples_per_class: Optional[int], seed: int, logger):
        """Load data using pandas (fallback method)."""
        
        log_and_print("  Step 1: Loading metadata...", logger_instance=logger)
        metadata_df = pd.read_parquet(parquet_file, columns=['class_name', 'label'])
        
        # Filter and sample using metadata only
        self.sample_indices = self._filter_and_sample_metadata(metadata_df, max_samples_per_class, seed)
        
        # Then load only the rows we need
        log_and_print(f"  Step 2: Loading {len(self.sample_indices)} selected samples...", logger_instance=logger)
        # Load full data and then select the indices we need
        full_df = pd.read_parquet(parquet_file)
        self.df = full_df.iloc[self.sample_indices].copy().reset_index(drop=True)
        del full_df  # Free memory
        
        # Reset sample indices to be contiguous (0, 1, 2, ...) since we have a new DataFrame
        self.sample_indices = list(range(len(self.df)))
        
        log_and_print(f"Pandas loading complete: {len(self.sample_indices)} samples", logger_instance=logger)
        
    def _filter_and_sample_metadata(self, metadata_df: pd.DataFrame, max_per_class: Optional[int], seed: int) -> List[int]:
        """Filter for selected classes and apply sampling limits using metadata only."""
        
        # Filter DataFrame for selected classes
        filtered_df = metadata_df[metadata_df['class_name'].isin(self.selected_classes)].copy()
        
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
                log_and_print(f"    {class_name}: {len(sampled_rows)} samples", logger_instance=logger)
            
            return sampled_indices
        else:
            # Use all samples for selected classes
            for class_name in self.selected_classes:
                class_count = len(filtered_df[filtered_df['class_name'] == class_name])
                logger = get_logger(__name__)
                log_and_print(f"    {class_name}: {class_count} samples", logger_instance=logger)
            
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


def load_precomputed_split(
    classes: List[str],
    train_samples_per_class: int = 1000,
    val_samples_per_class: int = 200,
    calib_samples_per_class: int = 200,
    seed: int = 42,
    splits_dir: str = "data/splits",
    include_calibration: bool = False
) -> Optional[Tuple[List[int], ...]]:
    """
    Load pre-computed train/val/calibration split if available.
    
    Args:
        include_calibration: If True, return (train, val, calib), else (train, val)
    
    Returns:
        (train_indices, val_indices) or (train_indices, val_indices, calib_indices) if found, None otherwise
    """
    
    # Try to find split with calibration first
    classes_str = f"{len(classes)}c"
    samples_str = f"{train_samples_per_class}+{val_samples_per_class}+{calib_samples_per_class}"
    filename = f"split_{classes_str}_{samples_str}_seed{seed}.json"
    split_file = Path(splits_dir) / filename
    
    # Fallback to old format without calibration
    if not split_file.exists():
        samples_str = f"{train_samples_per_class}+{val_samples_per_class}"
        filename = f"split_{classes_str}_{samples_str}_seed{seed}.json"
        split_file = Path(splits_dir) / filename
    
    if not split_file.exists():
        return None
    
    try:
        with open(split_file, 'r') as f:
            split_config = json.load(f)
        
        # Verify the split matches our requirements
        metadata = split_config['metadata']
        # Handle both class count (int) and class list comparisons
        classes_match = (
            (isinstance(classes, int) and metadata['classes'] == classes) or
            (isinstance(classes, list) and metadata['classes'] == len(classes))
        )
        if (classes_match and 
            metadata['train_samples'] == train_samples_per_class and
            metadata['val_samples'] == val_samples_per_class and
            metadata['seed'] == seed):
            
            splits = split_config['splits']
            train_indices = splits['train_indices']
            val_indices = splits['val_indices']
            
            if include_calibration and 'calib_indices' in splits:
                calib_indices = splits['calib_indices']
                return train_indices, val_indices, calib_indices
            else:
                return train_indices, val_indices
        else:
            logger = get_logger(__name__)
            logger.warning(f"Pre-computed split found but metadata mismatch: {split_file}")
            return None
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to load pre-computed split {split_file}: {e}")
        return None

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


class QuickDrawLazyDataset(Dataset):
    """
    Lazy-loading version of QuickDrawDataset for memory efficiency.
    
    Loads images on-demand from Parquet files instead of keeping everything in memory.
    Provides identical interface and results to QuickDrawDataset but with lower memory usage.
    """
    
    def __init__(
        self,
        data_dir: str = "data/quickdraw_parquet",
        classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.seed = seed
        
        logger = get_logger(__name__)
        
        # Load metadata to get class information
        metadata_file = self.data_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Select classes
        if classes is not None:
            self.selected_classes = sorted(classes)
        else:
            self.selected_classes = sorted(self.metadata['classes'])
        
        # Validate classes
        available_classes = set(self.metadata['classes'])
        for cls in self.selected_classes:
            if cls not in available_classes:
                raise ValueError(f"Class '{cls}' not found in dataset. Available: {sorted(available_classes)}")
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        # Load sample indices for each class (but not the actual data)
        self.class_indices = {}
        self.sample_metadata = []  # List of (class_name, local_index) tuples
        
        # Check if we have per-class files (preferred) or monolithic file
        per_class_dir = self.data_dir / "per_class"
        if per_class_dir.exists():
            self._load_per_class_indices(per_class_dir, max_samples_per_class, logger)
        else:
            self._load_monolithic_indices(max_samples_per_class, logger)
        
        log_and_print(f"Lazy dataset initialized: {len(self.sample_metadata)} samples from {len(self.selected_classes)} classes", logger_instance=logger)
    
    def _load_per_class_indices(self, per_class_dir: Path, max_samples_per_class: Optional[int], logger):
        """Load sample indices from per-class Parquet files."""
        
        # Load per-class metadata
        with open(per_class_dir / "metadata.json", 'r') as f:
            per_class_metadata = json.load(f)
        
        for class_name in self.selected_classes:
            if class_name not in per_class_metadata['files']:
                raise ValueError(f"No per-class file found for class '{class_name}'")
            
            # Load just the metadata (not the actual images)
            class_file = per_class_dir / per_class_metadata['files'][class_name]
            if not class_file.exists():
                raise FileNotFoundError(f"Per-class file not found: {class_file}")
            
            # Read parquet metadata to get row count
            if PYARROW_AVAILABLE:
                parquet_file = pq.ParquetFile(class_file)
                total_samples = parquet_file.metadata.num_rows
            else:
                # Fallback: read just to get length (less efficient)
                df_meta = pd.read_parquet(class_file, columns=['word'])
                total_samples = len(df_meta)
            
            # Determine how many samples to use
            if max_samples_per_class is not None:
                num_samples = min(max_samples_per_class, total_samples)
            else:
                num_samples = total_samples
            
            # Create indices for this class
            indices = list(range(num_samples))
            if num_samples < total_samples:
                # Sample deterministically
                random.seed(self.seed + hash(class_name))
                indices = sorted(random.sample(range(total_samples), num_samples))
            
            self.class_indices[class_name] = {
                'file_path': class_file,
                'indices': indices,
                'total_available': total_samples
            }
            
            # Add to global sample metadata
            for local_idx in indices:
                self.sample_metadata.append((class_name, local_idx))
        
        # Shuffle the global sample order
        random.seed(self.seed)
        random.shuffle(self.sample_metadata)
    
    def _load_monolithic_indices(self, max_samples_per_class: Optional[int], logger):
        """Load sample indices from monolithic Parquet file."""
        
        parquet_file = self.data_dir / "quickdraw_data.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Monolithic parquet file not found: {parquet_file}")
        
        log_and_print("Using monolithic Parquet format for lazy loading (consider splitting for better performance)...", logger_instance=logger)
        
        # Load just the class labels to build indices
        if PYARROW_AVAILABLE:
            table = pq.read_table(parquet_file, columns=['word'])
            df_labels = table.to_pandas()
        else:
            df_labels = pd.read_parquet(parquet_file, columns=['word'])
        
        # Group by class and sample
        for class_name in self.selected_classes:
            class_mask = df_labels['word'] == class_name
            class_indices = df_labels.index[class_mask].tolist()
            
            if max_samples_per_class is not None and len(class_indices) > max_samples_per_class:
                random.seed(self.seed + hash(class_name))
                class_indices = sorted(random.sample(class_indices, max_samples_per_class))
            
            self.class_indices[class_name] = {
                'file_path': parquet_file,
                'indices': class_indices,
                'total_available': len(df_labels[class_mask])
            }
            
            # Add to global sample metadata
            for global_idx in class_indices:
                self.sample_metadata.append((class_name, global_idx))
        
        # Shuffle the global sample order
        random.seed(self.seed)
        random.shuffle(self.sample_metadata)
    
    def __len__(self) -> int:
        return len(self.sample_metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load a single sample on-demand."""
        class_name, local_idx = self.sample_metadata[idx]
        
        # Load the specific sample from the appropriate file
        file_path = self.class_indices[class_name]['file_path']
        
        if str(file_path).endswith('per_class'):
            # Per-class file: local_idx is the row index within the class file
            if PYARROW_AVAILABLE:
                table = pq.read_table(file_path, filters=[('__index_level_0__', '=', local_idx)])
                if len(table) == 0:
                    # Fallback: read by row number
                    table = pq.read_table(file_path)
                    row_data = table.slice(local_idx, 1).to_pandas().iloc[0]
                else:
                    row_data = table.to_pandas().iloc[0]
            else:
                df = pd.read_parquet(file_path)
                row_data = df.iloc[local_idx]
        else:
            # Monolithic file: local_idx is the global row index
            if PYARROW_AVAILABLE:
                table = pq.read_table(file_path)
                row_data = table.slice(local_idx, 1).to_pandas().iloc[0]
            else:
                df = pd.read_parquet(file_path)
                row_data = df.iloc[local_idx]
        
        # Convert image data
        image_bytes = row_data['image']
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Ensure grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)
        
        # Get class label
        label = self.class_to_idx[class_name]
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the number of samples per class."""
        distribution = {}
        for class_name in self.selected_classes:
            distribution[class_name] = len(self.class_indices[class_name]['indices'])
        return distribution


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
    seed: int = 42,
    splits_dir: str = "data/splits"
) -> Tuple[DataLoader, DataLoader, Dict]:
    import time
    import json
    from pathlib import Path
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
    dataset_start = time.time()
    print(f"⏱️  Creating train dataset...")
    train_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=train_samples_per_class + val_samples_per_class,
        image_size=image_size,
        augment=True,
        invert_colors=invert_colors,
        seed=seed
    )
    train_dataset_time = time.time() - dataset_start
    print(f"⏱️  Train dataset creation took: {train_dataset_time:.2f}s")
    
    # Create validation dataset (no augmentation)
    val_dataset_start = time.time()
    print(f"⏱️  Creating val dataset...")
    val_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=train_dataset.selected_classes,  # Use same classes as train
        max_samples_per_class=train_samples_per_class + val_samples_per_class,
        image_size=image_size,
        augment=False,  # No augmentation for validation
        invert_colors=invert_colors,
        seed=seed
    )
    val_dataset_time = time.time() - val_dataset_start
    print(f"⏱️  Val dataset creation took: {val_dataset_time:.2f}s")
    
    # Try to load pre-computed split first
    logger = get_logger(__name__)
    log_and_print("\nLoading train/val split...", logger_instance=logger)
    timing_start = time.time()
    
    precomputed_split = load_precomputed_split(
        train_dataset.selected_classes, train_samples_per_class, val_samples_per_class, 
        seed=seed, splits_dir=splits_dir
    )
    
    if precomputed_split is not None:
        train_indices, val_indices = precomputed_split
        log_and_print("Using pre-computed split (fast loading)", logger_instance=logger)
        split_time = time.time() - timing_start
        print(f"⏱️  Split loading took: {split_time:.2f}s")
    else:
        log_and_print("Pre-computed split not found, computing on-the-fly...", logger_instance=logger)
        log_and_print("Run 'python scripts/precompute_splits.py' to speed up future runs", logger_instance=logger)
        train_indices, val_indices = create_stratified_split(
            train_dataset, train_samples_per_class, val_samples_per_class, seed
        )
        split_time = time.time() - timing_start
        print(f"⏱️  Split computation took: {split_time:.2f}s")
        
        # Debug: verify indices are in range
        max_idx = max(max(train_indices), max(val_indices))
        dataset_size = len(train_dataset)
        print(f"🔍 Split verification: max_index={max_idx}, dataset_size={dataset_size}")
        if max_idx >= dataset_size:
            print(f"❌ ERROR: Split indices out of range! max_idx={max_idx} >= dataset_size={dataset_size}")
        else:
            print(f"✅ Split indices OK: max_idx={max_idx} < dataset_size={dataset_size}")
            # Save the correct split for future use
            split_data = {
                'metadata': {'classes': len(train_dataset.selected_classes), 'train_samples': train_samples_per_class, 'val_samples': val_samples_per_class, 'seed': seed},
                'splits': {'train_indices': train_indices, 'val_indices': val_indices},
                'class_info': {'selected_classes': train_dataset.selected_classes}
            }
            Path(splits_dir).mkdir(exist_ok=True)
            split_file_path = Path(splits_dir) / f"split_{len(train_dataset.selected_classes)}c_{train_samples_per_class}+{val_samples_per_class}_seed{seed}.json"
            with open(split_file_path, 'w') as f:
                json.dump(split_data, f)
            print(f"💾 Saved correct split to: {split_file_path}")
    
    # Create subset datasets
    subset_start = time.time()
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    subset_time = time.time() - subset_start
    print(f"⏱️  Subset creation took: {subset_time:.2f}s")
    
    # Create dataloaders
    dataloader_start = time.time()
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
    
    dataloader_time = time.time() - dataloader_start
    print(f"⏱️  DataLoader creation took: {dataloader_time:.2f}s")
    
    total_time = time.time() - timing_start
    print(f"⏱️  TOTAL data loading time: {total_time:.2f}s")
    
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


def create_calibration_dataloader(
    data_dir: str = "data/quickdraw_parquet",
    classes: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    calib_samples_per_class: Optional[int] = None,
    train_samples_per_class: int = 1000,
    val_samples_per_class: int = 200,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    invert_colors: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, Dict]:
    """
    Create calibration dataloader for quantization using pre-computed splits.
    
    Args:
        calib_samples_per_class: Calibration samples per class
        Other args: Same as create_dataloaders
        
    Returns:
        calib_loader: Calibration dataloader
        metadata: Dict with class mappings and info
    """
    
    # Load available classes from metadata
    metadata_file = Path(data_dir) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Please run: python scripts/download_quickdraw_direct.py"
        )
    
    with open(metadata_file, 'r') as f:
        data_metadata = json.load(f)
    
    available_classes = data_metadata['classes']
    
    # Auto-select classes if num_classes is specified
    if classes is None and num_classes is not None:
        random.seed(seed)
        classes = sorted(random.sample(available_classes, min(num_classes, len(available_classes))))
        logger = get_logger(__name__)
        log_and_print(f"Auto-selected {num_classes} classes for calibration: {classes}", logger_instance=logger)
    
    # Auto-compute calibration samples based on README guidance
    if calib_samples_per_class is None:
        actual_num_classes = len(classes) if classes else num_classes
        # Target 2048 total calibration samples, distributed across classes
        target_total_calib = 2048
        calib_samples_per_class = max(8, min(200, target_total_calib // actual_num_classes))
        
        # Apply README scaling rules
        if actual_num_classes >= 300:  # All classes case
            calib_samples_per_class = max(6, min(12, target_total_calib // actual_num_classes))
        elif actual_num_classes >= 40:  # Medium class count (40-299)
            calib_samples_per_class = max(40, min(80, target_total_calib // actual_num_classes))
        elif actual_num_classes <= 10:  # Small class count
            calib_samples_per_class = max(100, min(200, target_total_calib // actual_num_classes))
        
        logger = get_logger(__name__)
        log_and_print(f"Auto-computed calibration: {calib_samples_per_class} samples/class ({calib_samples_per_class * actual_num_classes} total)", logger_instance=logger)
    
    # Try to load pre-computed split with calibration
    logger = get_logger(__name__)
    log_and_print("\nLoading calibration split...", logger_instance=logger)
    
    precomputed_split = load_precomputed_split(
        classes, train_samples_per_class, val_samples_per_class, calib_samples_per_class, 
        seed, include_calibration=True
    )
    
    if precomputed_split is not None and len(precomputed_split) == 3:
        train_indices, val_indices, calib_indices = precomputed_split
        log_and_print("Using pre-computed calibration split (fast loading)", logger_instance=logger)
        
        # Create calibration dataset
        calib_dataset = QuickDrawDataset(
            data_dir=data_dir,
            classes=classes,
            max_samples_per_class=train_samples_per_class + val_samples_per_class + calib_samples_per_class,
            image_size=image_size,
            augment=False,  # No augmentation for calibration
            invert_colors=invert_colors,
            seed=seed
        )
        
        # Create subset dataset for calibration
        calib_subset = torch.utils.data.Subset(calib_dataset, calib_indices)
        
    else:
        log_and_print("Pre-computed calibration split not found, using validation data for calibration", logger_instance=logger)
        log_and_print("Run 'python scripts/precompute_splits.py --main' to generate proper calibration splits", logger_instance=logger)
        
        # Fallback: use validation data as calibration
        _, val_loader, cal_metadata = create_dataloaders(
            data_dir=data_dir,
            classes=classes,
            num_classes=num_classes,
            train_samples_per_class=train_samples_per_class,
            val_samples_per_class=val_samples_per_class,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            invert_colors=invert_colors,
            seed=seed
        )
        return val_loader, cal_metadata
    
    # Create calibration dataloader
    calib_loader = DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order for calibration
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create metadata
    metadata = {
        'class_to_id': calib_dataset.class_to_id,
        'id_to_class': calib_dataset.id_to_class,
        'selected_classes': calib_dataset.selected_classes,
        'num_classes': calib_dataset.num_classes,
        'image_size': image_size,
        'calib_samples': len(calib_indices)
    }
    
    log_and_print(f"\nCalibration dataloader created:", logger_instance=logger)
    log_and_print(f"  Calibration: {len(calib_loader)} batches ({len(calib_indices)} samples)", logger_instance=logger)
    
    return calib_loader, metadata


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


def create_adaptive_dataloaders(
    data_dir: str = "data/quickdraw_parquet",
    classes: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_samples_per_class: int = 1000,
    val_samples_per_class: int = 200,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    invert_colors: bool = True,
    seed: int = 42,
    memory_threshold: float = 0.6,
    force_loading_method: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders with adaptive memory management.
    
    Automatically chooses between in-memory and lazy loading based on:
    - Estimated dataset memory requirements
    - Available system memory
    - User-specified threshold
    
    Args:
        data_dir: Directory containing QuickDraw Parquet files
        classes: List of class names to include (None = auto-select)
        num_classes: Number of classes to auto-select (ignored if classes provided)
        train_samples_per_class: Training samples per class
        val_samples_per_class: Validation samples per class
        image_size: Target image size for transforms
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        invert_colors: Whether to invert colors (white on black -> black on white)
        seed: Random seed for reproducibility
        memory_threshold: Use lazy loading if estimated memory > threshold * available memory
        force_loading_method: Force 'in_memory' or 'lazy' (None = auto-decide)
        
    Returns:
        Tuple of (train_loader, val_loader, metadata)
        
    Note:
        Both loading methods provide identical results for the same parameters.
        The choice only affects memory usage and loading performance.
    """
    logger = get_logger(__name__)
    
    # Determine classes to use
    if classes is None and num_classes is not None:
        available_classes = get_all_class_names(data_dir)
        random.seed(seed)
        classes = sorted(random.sample(available_classes, min(num_classes, len(available_classes))))
        log_and_print(f"Auto-selected {num_classes} classes: {classes[:3]}{'...' if len(classes) > 3 else ''}", logger_instance=logger)
    elif classes is not None:
        classes = sorted(classes)
    else:
        raise ValueError("Must provide either 'classes' or 'num_classes'")
    
    actual_num_classes = len(classes)
    
    # Decide loading method
    if force_loading_method:
        use_lazy = (force_loading_method.lower() == 'lazy')
        method_reason = f"forced to {force_loading_method}"
    else:
        use_lazy = should_use_lazy_loading(
            num_classes=actual_num_classes,
            train_samples_per_class=train_samples_per_class,
            val_samples_per_class=val_samples_per_class,
            memory_threshold=memory_threshold
        )
        
        estimated_mb = estimate_dataset_memory(
            num_classes=actual_num_classes,
            train_samples_per_class=train_samples_per_class,
            val_samples_per_class=val_samples_per_class
        ) / (1024 * 1024)
        
        available_gb = get_available_memory() / (1024 * 1024 * 1024)
        method_reason = f"estimated {estimated_mb:.1f}MB vs {available_gb:.1f}GB available"
    
    loading_method = "lazy" if use_lazy else "in-memory"
    log_and_print(f"Using {loading_method} loading ({method_reason})", logger_instance=logger)
    
    # Create datasets
    total_samples_per_class = train_samples_per_class + val_samples_per_class
    
    if use_lazy:
        # Create transforms for lazy dataset
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
        
        if invert_colors:
            transform_list.append(transforms.Lambda(lambda x: 1.0 - x))
        
        transform = transforms.Compose(transform_list)
        
        # Use lazy loading dataset
        dataset = QuickDrawLazyDataset(
            data_dir=data_dir,
            classes=classes,
            max_samples_per_class=total_samples_per_class,
            transform=transform,
            seed=seed
        )
    else:
        # Use in-memory dataset (uses built-in transform parameters)
        dataset = QuickDrawDataset(
            data_dir=data_dir,
            classes=classes,
            max_samples_per_class=total_samples_per_class,
            image_size=image_size,
            augment=False,  # No augmentation for validation
            invert_colors=invert_colors,
            seed=seed
        )
    
    # Create stratified train/val split
    train_indices, val_indices = create_stratified_split(
        dataset, train_samples_per_class, val_samples_per_class, seed
    )
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create metadata
    metadata = {
        'class_to_id': dataset.class_to_idx,
        'id_to_class': {v: k for k, v in dataset.class_to_idx.items()},
        'selected_classes': classes,
        'num_classes': actual_num_classes,
        'image_size': image_size,
        'loading_method': loading_method,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'estimated_memory_mb': estimate_dataset_memory(
            num_classes=actual_num_classes,
            train_samples_per_class=train_samples_per_class,
            val_samples_per_class=val_samples_per_class
        ) / (1024 * 1024)
    }
    
    log_and_print(f"\nAdaptive dataloaders created:", logger_instance=logger)
    log_and_print(f"  Method: {loading_method}", logger_instance=logger)
    log_and_print(f"  Classes: {actual_num_classes}", logger_instance=logger)
    log_and_print(f"  Train: {len(train_loader)} batches ({len(train_indices)} samples)", logger_instance=logger)
    log_and_print(f"  Val: {len(val_loader)} batches ({len(val_indices)} samples)", logger_instance=logger)
    
    return train_loader, val_loader, metadata
