#!/usr/bin/env python3
"""
Setup W&B authentication and create sweep
"""

import os
import wandb

# Set API key
os.environ["WANDB_API_KEY"] = "68e59e19727c439501ba94fced5c65bab307109d"

# Login to W&B
try:
    wandb.login()
    print("✅ W&B login successful!")
    
    # Check current user
    api = wandb.Api()
    print(f"Logged in as: {api.default_entity}")
    
except Exception as e:
    print(f"❌ W&B login failed: {e}")
