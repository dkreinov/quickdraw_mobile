
"""Step 2 placeholder: QuickDraw (bitmap) dataset loader.

Plan:
- Use Hugging Face `datasets` (google/quickdraw or quickdraw_bitmap via TFDS wrapper).
- Upsample 28x28 grayscale to 224x224 and repeat channels to 3.
- Provide subset selection for classes to keep iteration fast.
- Return PyTorch tensors.
"""
from typing import List, Optional, Tuple
import torch

class TODOQuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, classes: Optional[List[str]] = None, split: str = "train"):
        raise NotImplementedError("Implemented in Step 2")
    def __len__(self) -> int:
        return 0
    def __getitem__(self, idx: int):
        raise NotImplementedError("Implemented in Step 2")
