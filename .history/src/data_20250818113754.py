
"""Step 2: QuickDraw (bitmap) dataset loader.

Loads 28x28 grayscale bitmaps from HuggingFace google/quickdraw dataset.
Keeps them as single-channel (grayscale) for mobile efficiency.
Provides class filtering and stratified splits for training.
"""
from typing import List, Optional, Dict, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset


class QuickDrawDataset(Dataset):
    """
    QuickDraw bitmap dataset (28x28 grayscale) from HuggingFace.
    
    Key features:
    - Loads 28x28 grayscale bitmaps and upsamples to target size
    - Keeps single channel (no RGB conversion) for mobile efficiency  
    - Supports class filtering for faster iteration
    - Applies transforms for training/validation
    """
    
    def __init__(
        self,
        classes: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        image_size: int = 224,
        augment: bool = True,
        seed: int = 42
    ):
        """
        Args:
            classes: List of class names to include. If None, uses all 345 classes
            max_samples_per_class: Limit samples per class (for faster experimentation)  
            image_size: Target image size (224 for ViT, 256 for MobileViT)
            augment: Whether to apply data augmentation
            seed: Random seed for reproducible sampling
        """
        print("Loading QuickDraw dataset...")
        
        # Load the dataset from HuggingFace
        self.hf_dataset = load_dataset("google/quickdraw", "preprocessed_bitmaps", split="train")
        
        # Get all available class names (345 total)
        all_class_names = self.hf_dataset.features["label"].names
        print(f"Available classes: {len(all_class_names)}")
        
        # Determine which classes to use
        if classes is None:
            self.selected_classes = all_class_names
        else:
            # Validate that requested classes exist
            missing_classes = [c for c in classes if c not in all_class_names]
            if missing_classes:
                raise ValueError(f"Unknown class names: {missing_classes}")
            self.selected_classes = classes
            
        print(f"Using {len(self.selected_classes)} classes: {self.selected_classes[:10]}{'...' if len(self.selected_classes) > 10 else ''}")
        
        # Create mapping from class names to contiguous IDs (0, 1, 2, ...)
        self.selected_classes = sorted(self.selected_classes)  # Keep consistent order
        self.class_to_id = {name: idx for idx, name in enumerate(self.selected_classes)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        
        # Sample dataset indices for selected classes
        self.sample_indices = self._sample_dataset_indices(
            all_class_names, max_samples_per_class, seed
        )
        
        print(f"Total samples: {len(self.sample_indices)}")
        
        # Setup transforms
        self.image_size = image_size
        self.transforms = self._create_transforms(image_size, augment)
        
        # Keep reference to original class names for lookup
        self.all_class_names = all_class_names
        
    def _sample_dataset_indices(self, all_class_names: List[str], max_per_class: Optional[int], seed: int) -> List[int]:
        """Sample indices from the dataset, optionally limiting samples per class."""
        
        # Group indices by class
        class_indices = {name: [] for name in self.selected_classes}
        
        print("Scanning dataset for selected classes...")
        for idx, example in enumerate(self.hf_dataset):
            class_name = all_class_names[example["label"]]
            if class_name in class_indices:
                class_indices[class_name].append(idx)
                
        # Sample from each class if limit is specified
        random.seed(seed)
        sampled_indices = []
        
        for class_name, indices in class_indices.items():
            if max_per_class and len(indices) > max_per_class:
                indices = random.sample(indices, max_per_class)
            sampled_indices.extend(indices)
            print(f"  {class_name}: {len(indices)} samples")
            
        return sampled_indices
        
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
        # Get the actual dataset index
        dataset_idx = self.sample_indices[idx]
        
        # Load the example from HuggingFace dataset
        example = self.hf_dataset[dataset_idx]
        
        # Get image (PIL Image in 'L' mode - grayscale) and label
        image = example["image"]  # PIL Image, mode='L', size=(28, 28)
        original_label = example["label"]
        
        # Convert original label to our contiguous class ID
        class_name = self.all_class_names[original_label]
        label = self.class_to_id[class_name]
        
        # Apply transforms (resize, augment, convert to tensor)
        image = self.transforms(image)
        
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
        print(f"  {class_name}: {train_end} train, {val_end - train_end} val samples")
    
    return train_indices, val_indices


def create_dataloaders(
    classes: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    train_samples_per_class: int = 4000,
    val_samples_per_class: int = 500,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation dataloaders with metadata.
    
    Args:
        classes: Specific class names to use
        num_classes: If classes=None, randomly sample this many classes
        train_samples_per_class: Training samples per class
        val_samples_per_class: Validation samples per class
        image_size: Target image size
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        seed: Random seed
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader  
        metadata: Dict with class mappings and info
    """
    
    # Auto-select classes if num_classes is specified
    if classes is None and num_classes is not None:
        # Get all available classes and sample subset
        tmp_dataset = load_dataset("google/quickdraw", "preprocessed_bitmaps", split="train")
        all_class_names = tmp_dataset.features["label"].names
        
        random.seed(seed)
        classes = sorted(random.sample(all_class_names, num_classes))
        print(f"Auto-selected {num_classes} classes: {classes}")
    
    # Create training dataset (with augmentation)
    train_dataset = QuickDrawDataset(
        classes=classes,
        max_samples_per_class=train_samples_per_class + val_samples_per_class,
        image_size=image_size,
        augment=True,
        seed=seed
    )
    
    # Create validation dataset (no augmentation)
    val_dataset = QuickDrawDataset(
        classes=train_dataset.selected_classes,  # Use same classes as train
        max_samples_per_class=train_samples_per_class + val_samples_per_class,
        image_size=image_size,
        augment=False,  # No augmentation for validation
        seed=seed
    )
    
    # Create stratified split
    print("\nCreating stratified train/val split...")
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
    
    print(f"\nDataloaders created:")
    print(f"  Training: {len(train_loader)} batches ({len(train_indices)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_indices)} samples)")
    
    return train_loader, val_loader, metadata


def get_all_class_names() -> List[str]:
    """Get list of all 345 QuickDraw class names."""
    dataset = load_dataset("google/quickdraw", "preprocessed_bitmaps", split="train")
    return dataset.features["label"].names
