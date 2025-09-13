#!/usr/bin/env python3
"""
Generate labels.txt file for Android app from QuickDraw class names.

This script creates the labels.txt file needed by the Android app to map
class indices to human-readable names.
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from data import get_all_class_names
    from logging_config import get_logger, log_and_print
except ImportError:
    # Fallback if src modules not available
    def get_logger(name):
        import logging
        return logging.getLogger(name)
    def log_and_print(msg, logger_instance=None, level="INFO"):
        print(msg)

# Google's QuickDraw class list (345 classes) - fallback if data not available
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
    "suitcase", "sun", "swan", "sweater", "swing set", "sword", "syringe", "table", "teapot",
    "teddy-bear", "telephone", "television", "tennis racquet", "tent", "The Eiffel Tower",
    "The Great Wall of China", "The Mona Lisa", "tiger", "toaster", "toe", "toilet", "tooth",
    "toothbrush", "toothpaste", "tornado", "tractor", "traffic light", "train", "tree",
    "triangle", "trombone", "truck", "trumpet", "t-shirt", "umbrella", "underwear", "van",
    "vase", "violin", "washing machine", "watermelon", "waterslide", "whale", "wheel",
    "windmill", "wine bottle", "wine glass", "wristwatch", "yoga", "zebra", "zigzag"
]

def get_class_names_from_data(data_dir: str) -> list:
    """Try to get class names from downloaded data."""
    try:
        return get_all_class_names(data_dir)
    except Exception:
        return None

def create_labels_file(output_path: str, num_classes: int = 344, data_dir: str = None):
    """Create labels.txt file for Android app."""
    logger = get_logger(__name__)
    
    # Try to get class names from actual data first
    class_names = None
    if data_dir:
        class_names = get_class_names_from_data(data_dir)
    
    # Fallback to hardcoded list
    if class_names is None:
        log_and_print(f"Using fallback class list (data not available)", logger_instance=logger)
        class_names = QUICKDRAW_CLASSES
    else:
        log_and_print(f"Using class names from data: {data_dir}", logger_instance=logger)
    
    # Sort to ensure consistent ordering (same as training)
    class_names = sorted(class_names)
    
    # Take first num_classes
    if len(class_names) < num_classes:
        raise ValueError(f"Only {len(class_names)} classes available, but {num_classes} requested")
    
    selected_classes = class_names[:num_classes]
    
    # Write labels file
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        for class_name in selected_classes:
            f.write(f"{class_name}\n")
    
    log_and_print(f"Created labels file: {output_path}", logger_instance=logger)
    log_and_print(f"  Classes: {num_classes}", logger_instance=logger)
    log_and_print(f"  First 5: {selected_classes[:5]}", logger_instance=logger)
    log_and_print(f"  Last 5: {selected_classes[-5:]}", logger_instance=logger)
    
    return selected_classes

def main():
    parser = argparse.ArgumentParser(description='Create labels.txt for Android app')
    parser.add_argument('--output', type=str, default='labels.txt',
                       help='Output path for labels file')
    parser.add_argument('--classes', type=int, default=344,
                       help='Number of classes to include')
    parser.add_argument('--data-dir', type=str, default='../data/quickdraw_parquet',
                       help='Path to QuickDraw data directory')
    
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    log_and_print(f"Creating labels file for Android app", logger_instance=logger)
    
    try:
        selected_classes = create_labels_file(
            args.output, 
            args.classes, 
            args.data_dir
        )
        
        log_and_print(f"\n✓ Labels file ready for Android app!", logger_instance=logger)
        log_and_print(f"  Copy {args.output} to Android app's assets/ folder", logger_instance=logger)
        
        return 0
        
    except Exception as e:
        log_and_print(f"❌ Failed to create labels file: {e}", logger_instance=logger)
        return 1

if __name__ == "__main__":
    sys.exit(main())

