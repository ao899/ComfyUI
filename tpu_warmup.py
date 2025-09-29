"""
Enhanced TPU Warmup utilities for ComfyUI
Supports both automatic and manual warmup modes with advanced XLA integration
"""

import logging
import time
import torch
import json
from typing import Dict, Any, Optional

def is_tpu_warmup_needed():
    """Check if TPU warmup should be performed"""
    from comfy.cli_args import args
    import comfy.model_management as model_management
    
    if args.no_tpu_warmup:
        return False
    if not model_management.is_tpu():
        return False
        
    # Check if manual warmup mode is enabled
    if getattr(args, 'tpu_manual_warmup', False):
        logging.info("TPU manual warmup mode enabled - skipping automatic warmup")
        return False
        
    return True

def is_manual_warmup_mode():
    """Check if manual warmup mode is enabled"""
    from comfy.cli_args import args
    return getattr(args, 'tpu_manual_warmup', False)

def run_tpu_warmup_steps(warmup_steps: int = 2) -> bool:
    """
    Run enhanced TPU warmup with XLA integration
    
    Args:
        warmup_steps: Number of warmup iterations to run
        
    Returns:
        bool: True if warmup completed successfully, False otherwise
    """
    try:
        import comfy.model_management as model_management
        from comfy.cli_args import args
        
        if not is_tpu_warmup_needed():
            return True
            
        logging.info(f"Starting enhanced TPU warmup with {warmup_steps} steps...")
        warmup_start_time = time.perf_counter()
        
        device = model_management.get_torch_device()
        
        # Try to use advanced TPU features if available
        try:
            from comfy.tpu_xla import tpu_manager, create_execution_signature
            use_advanced_tpu = True
            logging.info("Using advanced TPU/XLA warmup features")
        except ImportError:
            use_advanced_tpu = False
            logging.info("Using basic TPU warmup")
        
        # Run warmup steps with XLA optimization
        for step in range(warmup_steps):
            step_start = time.perf_counter()
            
            try:
                with torch.no_grad():
                    if use_advanced_tpu:
                        # Advanced XLA warmup with step boundaries
                        with tpu_manager.step_context(f"warmup_step_{step}"):
                            with tpu_manager.autocast_context():
                                _run_warmup_operations(device)
                    else:
                        # Basic TPU warmup
                        _run_warmup_operations(device)
                
                # Force XLA compilation if on TPU
                if model_management.is_tpu():
                    try:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()  # Trigger compilation
                    except ImportError:
                        pass
                
                step_time = time.perf_counter() - step_start
                logging.info(f"TPU warmup step {step + 1}/{warmup_steps} completed in {step_time:.2f} seconds")
                    
            except Exception as e:
                logging.warning(f"TPU warmup step {step + 1} encountered error: {e}")
                continue
        
        # Final XLA sync
        if model_management.is_tpu():
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
                xm.wait_device_ops()
            except Exception as e:
                logging.warning(f"TPU warmup XLA sync failed: {e}")
        
        total_warmup_time = time.perf_counter() - warmup_start_time
        logging.info(f"Enhanced TPU warmup completed in {total_warmup_time:.2f} seconds")
        
        # Mark warmup as completed in advanced mode
        if use_advanced_tpu:
            try:
                # Create a basic signature for the warmup
                cache_key = create_execution_signature(
                    model_id="warmup",
                    input_shape=(1, 4, 64, 64),
                    batch_size=1,
                    cfg_scale=7.0,
                    steps=warmup_steps
                )
                tpu_manager.mark_compiled(cache_key)
            except Exception as e:
                logging.warning(f"Failed to mark warmup cache: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"Enhanced TPU warmup failed: {e}")
        return False

def _run_warmup_operations(device: torch.device):
    """Run basic warmup tensor operations"""
    # Text encoder simulation - small embedding operations
    x = torch.randn(1, 77, 512, device=device, dtype=torch.bfloat16)
    y = torch.randn(512, 768, device=device, dtype=torch.bfloat16)
    text_features = torch.matmul(x, y)
    
    # UNet simulation - basic convolution and attention-like operations  
    latent = torch.randn(1, 4, 64, 64, device=device, dtype=torch.bfloat16)
    conv_weight = torch.randn(320, 4, 3, 3, device=device, dtype=torch.bfloat16)
    conv_out = torch.conv2d(latent, conv_weight, padding=1)
    
    # Attention-like operations
    batch_size, channels, height, width = conv_out.shape
    attention_input = conv_out.view(batch_size, channels, height * width).transpose(1, 2)
    attention_weight = torch.randn(channels, channels, device=device, dtype=torch.bfloat16)
    attention_out = torch.matmul(attention_input, attention_weight)
    
    # VAE simulation - decoder-like operations
    vae_input = torch.randn(1, 4, 64, 64, device=device, dtype=torch.bfloat16)
    vae_weight = torch.randn(512, 4, 3, 3, device=device, dtype=torch.bfloat16)
    vae_out = torch.conv2d(vae_input, vae_weight, padding=1)
    
    # Force computation
    _ = vae_out.sum()

def should_skip_warmup_files(filename: str) -> bool:
    """
    Check if warmup-generated files should be ignored
    """
    return filename.startswith("tpu_warmup_ignore")

def log_manual_warmup_usage():
    """Log usage instructions for manual warmup"""
    from comfy.cli_args import args
    
    if not getattr(args, 'tpu_manual_warmup', False):
        return
        
    logging.info("TPU Manual Warmup Mode Active")
    logging.info("Usage:")
    logging.info("1. Use 'TPU Manual Warmup' node to warm up with target resolution/settings")
    logging.info("2. Set warmup parameters to match your intended workflow")
    logging.info("3. Execute warmup (1 step recommended)")
    logging.info("4. Run your full workflow with same settings for optimal performance")
    logging.info("5. Warmup cache will be reused for matching signatures")