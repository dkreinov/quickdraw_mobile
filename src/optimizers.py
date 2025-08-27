"""Custom optimizers for large batch training.

This module implements optimizers specifically designed for multi-GPU large batch training:
- LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
- Enhanced AdamW with better large batch handling
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, Optional


class LAMB(Optimizer):
    """
    LAMB optimizer for large batch training.
    
    Paper: "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
    https://arxiv.org/abs/1904.00962
    
    LAMB addresses the challenge of large batch training by:
    1. Layer-wise adaptive learning rates
    2. Proper normalization for different layer scales
    3. Better gradient handling for large batches
    
    Args:
        params: model parameters
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added to denominator for numerical stability (default: 1e-6)
        weight_decay: weight decay coefficient (default: 0.01)
        always_adapt: always apply layer-wise adaptation (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        always_adapt: bool = True
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            always_adapt=always_adapt
        )
        super(LAMB, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply bias correction
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Compute update
                update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group['eps'])
                
                # Add weight decay
                if group['weight_decay'] > 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Layer-wise adaptation
                weight_norm = p.data.norm()
                update_norm = update.norm()
                
                if weight_norm > 0 and update_norm > 0:
                    # LAMB layer-wise adaptation
                    trust_ratio = weight_norm / update_norm
                    trust_ratio = min(trust_ratio, 10.0)  # Clip for stability
                else:
                    trust_ratio = 1.0
                
                # Apply update with layer-wise learning rate
                if group['always_adapt'] or weight_norm > 0:
                    p.data.add_(update, alpha=-group['lr'] * trust_ratio)
                else:
                    p.data.add_(update, alpha=-group['lr'])
        
        return loss


class AdamWLargeBatch(Optimizer):
    """
    Enhanced AdamW for large batch training with improved stability.
    
    Modifications for large batch training:
    - Better epsilon handling
    - Gradient clipping integration
    - Improved bias correction for large batch scenarios
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        correct_bias: bool = True,
        max_grad_norm: Optional[float] = None
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            max_grad_norm=max_grad_norm
        )
        super(AdamWLargeBatch, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        # Apply gradient clipping if specified
        for group in self.param_groups:
            if group['max_grad_norm'] is not None:
                params_with_grad = [p for p in group['params'] if p.grad is not None]
                if params_with_grad:
                    torch.nn.utils.clip_grad_norm_(params_with_grad, group['max_grad_norm'])
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamWLargeBatch does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                
                # Bias correction for large batch training
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply weight decay (L2 regularization)
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


def create_optimizer_for_large_batch(
    model,
    optimizer_name: str = "lamb",
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-6,
    **kwargs
) -> Optimizer:
    """
    Create an optimizer suitable for large batch training.
    
    Args:
        model: PyTorch model
        optimizer_name: "lamb", "adamw_large", or "adamw"
        lr: learning rate
        weight_decay: weight decay coefficient
        betas: beta parameters for momentum
        eps: epsilon for numerical stability
        **kwargs: additional optimizer-specific arguments
    
    Returns:
        Configured optimizer
    """
    
    # Separate parameters for weight decay (best practice)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to bias and norm layers
        if param.ndim <= 1 or 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_name.lower() == "lamb":
        return LAMB(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "adamw_large":
        return AdamWLargeBatch(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_recommended_lr_for_batch_size(
    base_lr: float,
    base_batch_size: int,
    target_batch_size: int,
    scaling_method: str = "sqrt"
) -> float:
    """
    Get recommended learning rate for a given batch size.
    
    Args:
        base_lr: baseline learning rate
        base_batch_size: baseline batch size
        target_batch_size: target batch size
        scaling_method: "linear", "sqrt", or "custom"
    
    Returns:
        Scaled learning rate
    """
    scale_factor = target_batch_size / base_batch_size
    
    if scaling_method == "linear":
        return base_lr * scale_factor
    elif scaling_method == "sqrt":
        return base_lr * math.sqrt(scale_factor)
    elif scaling_method == "custom":
        # Custom scaling that works well for vision models
        if scale_factor <= 4:
            return base_lr * scale_factor
        else:
            return base_lr * (4 + math.sqrt(scale_factor - 4))
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")


# Example usage and testing
if __name__ == "__main__":
    # Create a dummy model for testing
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Test LAMB optimizer
    optimizer = create_optimizer_for_large_batch(
        model,
        optimizer_name="lamb",
        lr=0.001,
        weight_decay=0.01
    )
    
    print(f"Created LAMB optimizer: {type(optimizer).__name__}")
    print(f"Parameter groups: {len(optimizer.param_groups)}")
    
    # Test learning rate scaling
    base_lr = 0.0003
    base_batch = 64
    target_batch = 1024
    
    for method in ["linear", "sqrt", "custom"]:
        scaled_lr = get_recommended_lr_for_batch_size(
            base_lr, base_batch, target_batch, method
        )
        print(f"LR scaling ({method}): {base_lr} -> {scaled_lr:.6f} (batch {base_batch} -> {target_batch})")
