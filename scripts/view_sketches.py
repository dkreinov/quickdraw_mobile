#!/usr/bin/env python3
"""
QuickDraw Sketch Viewer

A simple script to visualize sketches from the QuickDraw dataset.
Displays sketches in a grid format with class labels and provides
interactive features for exploration.

Usage Examples:
    # Interactive mode with menu
    python scripts/view_sketches.py --interactive
    
    # View specific classes
    python scripts/view_sketches.py --classes cat dog apple --num-samples 16
    
    # View random classes
    python scripts/view_sketches.py --random-classes 5 --num-samples 20
    
    # View single class
    python scripts/view_sketches.py --class-name cat --num-samples 25
    
    # Save to file (useful for headless environments)
    python scripts/view_sketches.py --classes cat dog --save sketches.png
    
    # Invert colors for traditional black-on-white appearance
    python scripts/view_sketches.py --classes cat dog --invert-colors
    
    # List all available classes
    python scripts/view_sketches.py --list-classes
"""

import argparse
import sys
import random
from pathlib import Path
from typing import List, Optional, Tuple
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import io

# Add the src directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data import QuickDrawDataset, get_all_class_names
except ImportError as e:
    print(f"❌ Error importing data module: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def plot_sketches_grid(
    images: List[np.ndarray], 
    labels: List[str], 
    class_names: List[str],
    title: str = "QuickDraw Sketches",
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None,
    output_dir: str = "scripts/output",
    invert_colors: bool = False
):
    """
    Plot a grid of sketches with their class labels.
    
    Args:
        images: List of image arrays (28x28 grayscale)
        labels: List of class labels (integers)
        class_names: List mapping label indices to class names
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        output_dir: Directory to save plots to
        invert_colors: If True, show black strokes on white background (default: white on black)
    """
    
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return
    
    # Calculate grid dimensions
    cols = min(8, n_images)  # Max 8 columns
    rows = math.ceil(n_images / cols)
    
    # Set figure size
    if figsize is None:
        figsize = (2 * cols, 2.5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Handle single row case
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(n_images):
        ax = axes[i]
        
        # Optionally invert colors (white strokes on black -> black strokes on white)
        image_to_show = 255 - images[i] if invert_colors else images[i]
        
        # Display image
        ax.imshow(image_to_show, cmap='gray', interpolation='nearest')
        ax.set_title(f"{class_names[labels[i]]}", fontsize=10, pad=5)
        ax.axis('off')
        
        # Add a subtle border
        rect = patches.Rectangle((0, 0), 27, 27, linewidth=1, 
                               edgecolor='lightgray', facecolor='none')
        ax.add_patch(rect)
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # If save_path doesn't include directory, save to output_dir
        save_file = Path(save_path)
        if save_file.parent == Path('.'):
            full_save_path = output_path / save_file.name
        else:
            full_save_path = save_file
            
        plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Plot saved to: {full_save_path}")
    
    plt.show()


def load_sample_sketches(
    dataset: QuickDrawDataset,
    num_samples: int = 16,
    classes: Optional[List[str]] = None,
    random_seed: int = 42
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load sample sketches from the dataset.
    
    Args:
        dataset: QuickDrawDataset instance
        num_samples: Number of samples to load
        classes: Optional list of specific classes to sample from
        random_seed: Random seed for sampling
        
    Returns:
        images: List of image arrays
        labels: List of label integers
        class_names: List of class names
    """
    
    random.seed(random_seed)
    
    # Get available indices
    if classes is None:
        # Sample from all available data
        available_indices = list(range(len(dataset)))
    else:
        # Filter for specific classes
        available_indices = []
        class_ids = [dataset.class_to_id[cls] for cls in classes if cls in dataset.class_to_id]
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in class_ids:
                available_indices.append(idx)
    
    if len(available_indices) == 0:
        print("❌ No samples found for the specified classes")
        return [], [], []
    
    # Sample random indices
    sample_size = min(num_samples, len(available_indices))
    sampled_indices = random.sample(available_indices, sample_size)
    
    # Load images and labels
    images = []
    labels = []
    
    for idx in sampled_indices:
        image_tensor, label = dataset[idx]
        
        # Convert tensor to numpy array (remove channel dimension)
        image_np = image_tensor.squeeze(0).numpy()
        
        images.append(image_np)
        labels.append(label)
    
    return images, labels, dataset.selected_classes


def interactive_sketch_viewer(data_dir: str = "data/quickdraw_parquet"):
    """
    Interactive sketch viewer with menu options.
    """
    
    print("🎨 QuickDraw Sketch Viewer")
    print("=" * 50)
    
    try:
        # Load available classes
        available_classes = get_all_class_names(data_dir)
        print(f"📊 Found {len(available_classes)} available classes")
        
        while True:
            print("\nChoose an option:")
            print("1. View random sketches from all classes")
            print("2. View sketches from specific classes")
            print("3. View sketches from a single class")
            print("4. List all available classes")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                # Random sketches from all classes
                num_samples = int(input("Number of sketches to display (default 16): ") or "16")
                
                print(f"\n🎲 Loading {num_samples} random sketches...")
                dataset = QuickDrawDataset(data_dir=data_dir, augment=False, invert_colors=False)
                
                images, labels, class_names = load_sample_sketches(
                    dataset, num_samples=num_samples
                )
                
                if images:
                    plot_sketches_grid(
                        images, labels, class_names,
                        title=f"Random QuickDraw Sketches ({len(images)} samples)"
                    )
                
            elif choice == '2':
                # Specific classes
                print(f"\nAvailable classes: {', '.join(available_classes[:10])}{'...' if len(available_classes) > 10 else ''}")
                class_input = input("Enter class names (comma-separated): ").strip()
                
                if not class_input:
                    continue
                
                classes = [cls.strip() for cls in class_input.split(',')]
                num_samples = int(input("Number of sketches to display (default 16): ") or "16")
                
                # Validate classes
                invalid_classes = [cls for cls in classes if cls not in available_classes]
                if invalid_classes:
                    print(f"⚠️  Unknown classes: {invalid_classes}")
                    continue
                
                print(f"\n🖼️  Loading sketches from classes: {classes}")
                dataset = QuickDrawDataset(data_dir=data_dir, classes=classes, augment=False, invert_colors=False)
                
                images, labels, class_names = load_sample_sketches(
                    dataset, num_samples=num_samples, classes=classes
                )
                
                if images:
                    plot_sketches_grid(
                        images, labels, class_names,
                        title=f"QuickDraw Sketches: {', '.join(classes)} ({len(images)} samples)"
                    )
                
            elif choice == '3':
                # Single class
                print(f"\nAvailable classes: {', '.join(available_classes[:10])}{'...' if len(available_classes) > 10 else ''}")
                class_name = input("Enter class name: ").strip()
                
                if class_name not in available_classes:
                    print(f"⚠️  Unknown class: {class_name}")
                    continue
                
                num_samples = int(input("Number of sketches to display (default 25): ") or "25")
                
                print(f"\n🖼️  Loading {num_samples} sketches of '{class_name}'...")
                dataset = QuickDrawDataset(data_dir=data_dir, classes=[class_name], augment=False, invert_colors=False)
                
                images, labels, class_names = load_sample_sketches(
                    dataset, num_samples=num_samples, classes=[class_name]
                )
                
                if images:
                    plot_sketches_grid(
                        images, labels, class_names,
                        title=f"QuickDraw Sketches: {class_name} ({len(images)} samples)"
                    )
                
            elif choice == '4':
                # List all classes
                print(f"\n📋 All available classes ({len(available_classes)}):")
                for i, cls in enumerate(available_classes, 1):
                    print(f"  {i:3d}. {cls}")
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="View QuickDraw sketches")
    
    parser.add_argument("--data-dir", default="data/quickdraw_parquet",
                       help="Data directory (default: data/quickdraw_parquet)")
    
    # Viewing options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--classes", nargs="+", help="Specific class names to view")
    group.add_argument("--class-name", help="Single class name to view")
    group.add_argument("--random-classes", type=int, help="Number of random classes to select")
    group.add_argument("--interactive", action="store_true", help="Interactive mode with menu")
    
    parser.add_argument("--num-samples", type=int, default=16,
                       help="Number of sketches to display (default: 16)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--save", help="Save plot to file")
    parser.add_argument("--output-dir", default="scripts/output", 
                       help="Output directory for saved plots (default: scripts/output)")
    parser.add_argument("--invert-colors", action="store_true",
                       help="Invert colors (black strokes on white background instead of white on black)")
    parser.add_argument("--figsize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                       help="Figure size in inches")
    parser.add_argument("--list-classes", action="store_true",
                       help="List all available classes and exit")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Please run the download script first:")
        print("  python scripts/download_quickdraw_direct.py --num-classes 10 --samples-per-class 1000")
        return 1
    
    try:
        # List classes option
        if args.list_classes:
            available_classes = get_all_class_names(args.data_dir)
            print(f"📋 Available QuickDraw classes ({len(available_classes)}):")
            for i, cls in enumerate(available_classes, 1):
                print(f"  {i:3d}. {cls}")
            return 0
        
        # Interactive mode
        if args.interactive:
            interactive_sketch_viewer(args.data_dir)
            return 0
        
        # Determine classes to use
        available_classes = get_all_class_names(args.data_dir)
        
        if args.classes:
            # Validate classes
            invalid_classes = [cls for cls in args.classes if cls not in available_classes]
            if invalid_classes:
                print(f"❌ Unknown classes: {invalid_classes}")
                print("Use --list-classes to see all available classes")
                return 1
            selected_classes = args.classes
        elif args.class_name:
            if args.class_name not in available_classes:
                print(f"❌ Unknown class: {args.class_name}")
                print("Use --list-classes to see all available classes")
                return 1
            selected_classes = [args.class_name]
        elif args.random_classes:
            random.seed(args.seed)
            selected_classes = random.sample(available_classes, 
                                           min(args.random_classes, len(available_classes)))
            print(f"🎲 Randomly selected classes: {selected_classes}")
        else:
            # Default: show interactive mode
            interactive_sketch_viewer(args.data_dir)
            return 0
        
        # Load dataset
        print(f"📊 Loading dataset with classes: {selected_classes}")
        dataset = QuickDrawDataset(
            data_dir=args.data_dir,
            classes=selected_classes,
            augment=False,  # No augmentation for viewing
            invert_colors=False,  # Show original format for visualization
            seed=args.seed
        )
        
        # Load sample sketches
        print(f"🖼️  Loading {args.num_samples} sample sketches...")
        images, labels, class_names = load_sample_sketches(
            dataset, 
            num_samples=args.num_samples,
            classes=selected_classes,
            random_seed=args.seed
        )
        
        if not images:
            print("❌ No images to display")
            return 1
        
        # Create plot
        title = f"QuickDraw Sketches: {', '.join(selected_classes)} ({len(images)} samples)"
        plot_sketches_grid(
            images, labels, class_names,
            title=title,
            figsize=tuple(args.figsize) if args.figsize else None,
            save_path=args.save,
            output_dir=args.output_dir,
            invert_colors=args.invert_colors
        )
        
        print(f"✅ Displayed {len(images)} sketches")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
