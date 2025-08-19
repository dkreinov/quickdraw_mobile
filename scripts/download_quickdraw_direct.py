#!/usr/bin/env python3
"""
Download QuickDraw dataset directly from Google's sources and convert to Parquet.

This script bypasses HuggingFace's deprecated loading script by downloading
the raw data directly from Google's QuickDraw dataset repository.

Usage:
    python scripts/download_quickdraw_direct.py --classes cat dog apple --samples-per-class 5000
    python scripts/download_quickdraw_direct.py --num-classes 10 --samples-per-class 3000
    python scripts/download_quickdraw_direct.py --all-classes --samples-per-class 1000
"""

import argparse
import json
import os
import urllib.request
import numpy as np
from pathlib import Path
from typing import List, Optional
import random
import io
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add src to path for logging
sys.path.append(str(Path(__file__).parent.parent / "src"))
try:
    from logging_config import get_logger, log_and_print
except ImportError:
    # Fallback if logging config is not available
    def log_and_print(msg, **kwargs):
        print(msg)
    def get_logger(name):
        return None


# Google's QuickDraw class list (345 classes)
QUICKDRAW_CLASSES = [
    "aircraft carrier", "airplane", "alarm clock", "ambulance", "angel", "animal migration",
    "ant", "anvil", "apple", "arm", "asparagus", "axe", "backpack", "banana", "bandage",
    "barn", "baseball", "baseball bat", "basket", "basketball", "bat", "bathtub", "beach",
    "bear", "beard", "bed", "bee", "belt", "bench", "bicycle", "binoculars", "bird",
    "birthday cake", "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie",
    "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket", "bulldozer",
    "bus", "bush", "butterfly", "cactus", "cake", "calculator", "calendar", "camel",
    "camera", "camouflage", "campfire", "candle", "cannon", "canoe", "car", "carrot",
    "castle", "cat", "ceiling fan", "cell phone", "cello", "chair", "chandelier",
    "church", "circle", "clarinet", "clock", "cloud", "coffee cup", "compass", "computer",
    "cookie", "cooler", "couch", "cow", "crab", "crayon", "crocodile", "crown", "cruise ship",
    "cup", "diamond", "dishwasher", "diving board", "dog", "dolphin", "donut", "door",
    "dragon", "dresser", "drill", "drums", "duck", "dumbbell", "ear", "elbow", "elephant",
    "envelope", "eraser", "eye", "eyeglasses", "face", "fan", "feather", "fence", "finger",
    "fire hydrant", "fireplace", "firetruck", "fish", "flamingo", "flashlight", "flip flops",
    "floor lamp", "flower", "flying saucer", "foot", "fork", "frog", "frying pan",
    "garden", "garden hose", "giraffe", "goatee", "golf club", "grapes", "grass",
    "guitar", "hamburger", "hammer", "hand", "harp", "hat", "headphones", "hedgehog",
    "helicopter", "helmet", "hexagon", "hockey puck", "hockey stick", "horse", "hospital",
    "hot air balloon", "hot dog", "hot tub", "hourglass", "house", "house plant", "hurricane",
    "ice cream", "jacket", "jail", "kangaroo", "key", "keyboard", "knee", "knife",
    "ladder", "lantern", "laptop", "leaf", "leg", "light bulb", "lighter", "lighthouse",
    "lightning", "line", "lion", "lipstick", "lobster", "lollipop", "mailbox", "map",
    "marker", "matches", "megaphone", "mermaid", "microphone", "microwave", "monkey",
    "moon", "mosquito", "motorbike", "mountain", "mouse", "moustache", "mouth", "mug",
    "mushroom", "nail", "necklace", "nose", "ocean", "octagon", "octopus", "onion",
    "oven", "owl", "paintbrush", "paint can", "palm tree", "panda", "pants", "paper clip",
    "parachute", "parrot", "passport", "peanut", "pear", "peas", "pencil", "penguin",
    "piano", "pickup truck", "picture frame", "pig", "pillow", "pineapple", "pizza",
    "pliers", "police car", "pond", "pool", "popsicle", "postcard", "potato", "power outlet",
    "purse", "rabbit", "raccoon", "radio", "rain", "rainbow", "rake", "remote control",
    "rhinoceros", "rifle", "river", "roller coaster", "rollerskates", "sailboat", "sandwich",
    "saw", "saxophone", "school bus", "scissors", "scorpion", "screwdriver", "sea turtle",
    "see saw", "shark", "sheep", "shoe", "shorts", "shovel", "sink", "skateboard", "skull",
    "skyscraper", "sleeping bag", "smiley face", "snail", "snake", "snorkel", "snowflake",
    "snowman", "soccer ball", "sock", "speedboat", "spider", "spoon", "spreadsheet",
    "square", "squiggle", "squirrel", "stairs", "star", "steak", "stereo", "stethoscope",
    "stitches", "stop sign", "stove", "strawberry", "streetlight", "string bean", "submarine",
    "suitcase", "sun", "swan", "sweater", "swing set", "sword", "syringe", "table",
    "teapot", "teddy-bear", "telephone", "television", "tennis racquet", "tent", "The Eiffel Tower",
    "The Great Wall of China", "The Mona Lisa", "tiger", "toaster", "toe", "toilet",
    "tooth", "toothbrush", "toothpaste", "tornado", "tractor", "traffic light", "train",
    "tree", "triangle", "trombone", "truck", "trumpet", "umbrella", "underwear", "van",
    "vase", "violin", "washing machine", "watermelon", "waterslide", "whale", "wheel",
    "windmill", "wine bottle", "wine glass", "wristwatch", "yoga", "zebra", "zigzag"
]


def download_class_data(class_name: str, max_samples: int = None) -> List[dict]:
    """Download bitmap data for a specific class from Google's QuickDraw dataset."""
    
    # Convert class name to filename format (replace spaces with %20)
    filename = class_name.replace(' ', '%20')
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{filename}.npy"
    
    print(f"  Downloading {class_name}...")
    
    try:
        # Download the .npy file
        with urllib.request.urlopen(url) as response:
            data = response.read()
        
        # Load numpy array from bytes
        array_data = np.load(io.BytesIO(data))
        
        print(f"    Found {len(array_data)} samples")
        
        # Limit samples if specified
        if max_samples and len(array_data) > max_samples:
            # Randomly sample
            indices = np.random.choice(len(array_data), max_samples, replace=False)
            array_data = array_data[indices]
            print(f"    Sampled {max_samples} samples")
        
        # Convert to list of dictionaries with image data
        samples = []
        for i, bitmap in enumerate(array_data):
            # Reshape from 784 (28*28) to 28x28
            image_array = bitmap.reshape(28, 28)
            
            # Convert to PIL Image
            image = Image.fromarray(image_array.astype(np.uint8), mode='L')
            
            # Convert to bytes for storage
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            
            samples.append({
                'image_bytes': img_bytes.getvalue(),
                'class_name': class_name,
                'original_idx': i
            })
        
        return samples
        
    except Exception as e:
        print(f"    Error downloading {class_name}: {e}")
        return []


def download_and_convert(
    classes: List[str],
    samples_per_class: int,
    output_dir: str = "data/quickdraw_parquet",
    seed: int = 42
):
    """Download QuickDraw data directly from Google and convert to Parquet format."""
    
    np.random.seed(seed)
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(__name__)
    log_and_print(f"Downloading QuickDraw data directly from Google...", logger_instance=logger)
    log_and_print(f"   Classes: {len(classes)} classes", logger_instance=logger)
    log_and_print(f"   Samples per class: {samples_per_class}", logger_instance=logger)
    log_and_print(f"   Output directory: {output_path}", logger_instance=logger)
    
    # Create class ID mapping
    class_to_id = {name: idx for idx, name in enumerate(sorted(classes))}
    id_to_class = {idx: name for name, idx in class_to_id.items()}
    
    # Download data for each class
    all_samples = []
    successful_classes = []
    
    for class_name in tqdm(classes, desc="Downloading classes"):
        samples = download_class_data(class_name, samples_per_class)
        
        if samples:
            # Add label information
            for sample in samples:
                sample['label'] = class_to_id[class_name]
            
            all_samples.extend(samples)
            successful_classes.append(class_name)
        else:
            print(f"  Failed to download {class_name}")
    
    if not all_samples:
        raise RuntimeError("Failed to download any data!")
    
    log_and_print(f"\nSuccessfully downloaded {len(successful_classes)}/{len(classes)} classes", logger_instance=logger)
    log_and_print(f"   Total samples: {len(all_samples)}", logger_instance=logger)
    
    # Shuffle all samples
    random.shuffle(all_samples)
    
    log_and_print(f"\nConverting {len(all_samples)} samples to Parquet format...", logger_instance=logger)
    
    # Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # Save to Parquet
    parquet_file = output_path / "quickdraw_data.parquet"
    df.to_parquet(parquet_file, index=False, compression='snappy')
    
    # Update class mappings for successful classes only
    successful_class_to_id = {name: idx for idx, name in enumerate(sorted(successful_classes))}
    successful_id_to_class = {idx: name for name, idx in successful_class_to_id.items()}
    
    # Save metadata
    metadata = {
        'classes': sorted(successful_classes),
        'class_to_id': successful_class_to_id,
        'id_to_class': successful_id_to_class,
        'num_classes': len(successful_classes),
        'samples_per_class': samples_per_class,
        'total_samples': len(all_samples),
        'seed': seed,
        'parquet_file': str(parquet_file),
        'created_with': 'download_quickdraw_direct.py',
        'source': 'Google QuickDraw Dataset (direct download)'
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_and_print(f"\nConversion complete!", logger_instance=logger)
    log_and_print(f"   Data file: {parquet_file}", logger_instance=logger)
    log_and_print(f"   Metadata: {metadata_file}", logger_instance=logger)
    log_and_print(f"   Total size: {parquet_file.stat().st_size / (1024*1024):.1f} MB", logger_instance=logger)
    log_and_print(f"   Successfully downloaded: {len(successful_classes)} classes", logger_instance=logger)
    
    if len(successful_classes) < len(classes):
        failed_classes = set(classes) - set(successful_classes)
        print(f"   Failed to download: {sorted(failed_classes)}")
    
    return str(output_path)


def verify_download(output_dir: str):
    """Verify that the downloaded dataset is valid and complete."""
    
    try:
        # Load metadata
        metadata_file = Path(output_dir) / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load parquet file
        parquet_file = Path(output_dir) / "quickdraw_data.parquet"
        df = pd.read_parquet(parquet_file)
        
        # Basic checks
        assert len(df) > 0, "Dataset is empty"
        assert 'image_bytes' in df.columns, "Missing image_bytes column"
        assert 'class_name' in df.columns, "Missing class_name column"
        assert 'label' in df.columns, "Missing label column"
        
        # Check metadata consistency
        assert metadata['total_samples'] == len(df), "Sample count mismatch"
        assert metadata['num_classes'] == len(metadata['classes']), "Class count mismatch"
        
        # Test loading a few images
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            image_bytes = row['image_bytes']
            
            # Try to load image
            image = Image.open(io.BytesIO(image_bytes))
            assert image.mode == 'L', f"Expected grayscale image, got {image.mode}"
            assert image.size == (28, 28), f"Expected 28x28 image, got {image.size}"
            
            # Check label is valid
            label = row['label']
            class_name = row['class_name']
            assert 0 <= label < metadata['num_classes'], f"Invalid label: {label}"
            assert class_name in metadata['classes'], f"Invalid class name: {class_name}"
        
        print(f" Verification passed:")
        print(f"   📊 {len(df):,} samples across {metadata['num_classes']} classes")
        print(f"   🖼️  Images: 28x28 grayscale PNG format")
        print(f"   📝 Metadata: consistent labels and class mappings")
        print(f"    File size: {(parquet_file.stat().st_size / (1024*1024)):.1f} MB")
        
    except Exception as e:
        print(f"  Verification warning: {e}")
        print(f"   Dataset may still be usable, but please check manually")


def main():
    parser = argparse.ArgumentParser(description="Download QuickDraw dataset directly from Google")
    
    # Class selection options (mutually exclusive)
    class_group = parser.add_mutually_exclusive_group(required=True)
    class_group.add_argument("--classes", nargs="+", help="Specific class names to download")
    class_group.add_argument("--num-classes", type=int, help="Number of classes to randomly select")
    class_group.add_argument("--all-classes", action="store_true", help="Download all 345 classes")
    
    # Other options
    parser.add_argument("--samples-per-class", type=int, default=1000, 
                       help="Number of samples per class (default: 1000)")
    parser.add_argument("--output-dir", default="data/quickdraw_parquet",
                       help="Output directory (default: data/quickdraw_parquet)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--list-classes", action="store_true", 
                       help="List all available class names and exit")
    
    args = parser.parse_args()
    
    # Handle list classes option
    if args.list_classes:
        print(" Available QuickDraw classes:")
        for i, class_name in enumerate(QUICKDRAW_CLASSES, 1):
            print(f"  {i:3d}. {class_name}")
        print(f"\nTotal: {len(QUICKDRAW_CLASSES)} classes")
        return
    
    # Determine which classes to download
    if args.classes:
        # Validate class names
        missing_classes = [c for c in args.classes if c not in QUICKDRAW_CLASSES]
        if missing_classes:
            print(f" Unknown class names: {missing_classes}")
            print(f"Use --list-classes to see all available classes")
            return 1
        selected_classes = args.classes
    elif args.num_classes:
        np.random.seed(args.seed)
        selected_classes = sorted(np.random.choice(QUICKDRAW_CLASSES, args.num_classes, replace=False))
        print(f" Randomly selected {args.num_classes} classes: {selected_classes}")
    elif args.all_classes:
        selected_classes = QUICKDRAW_CLASSES
        print(f" Downloading all {len(selected_classes)} classes")
    
    # Download and convert
    try:
        output_dir = download_and_convert(
            classes=selected_classes,
            samples_per_class=args.samples_per_class,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        # Verify the download integrity
        print(f"\n Verifying download integrity...")
        verify_download(output_dir)
        
        print(f"\n Success! QuickDraw dataset ready for training.")
        print(f"   Use this in your training script:")
        print(f"   from data import QuickDrawDataset")
        print(f"   dataset = QuickDrawDataset(data_dir='{output_dir}')")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
