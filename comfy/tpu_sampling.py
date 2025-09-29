"""
TPU/XLA-optimized sampling utilities for ComfyUI
Provides step boundary management, compilation caching, and mixed precision
"""

import logging
import torch
import time
from typing import Optional, Dict, Any, Tuple, Callable, Union
from contextlib import contextmanager

def is_tpu_sampling_available() -> bool:
    """Check if TPU sampling features are available"""
    try:
        from . import model_management
        return model_management.is_tpu() and model_management.is_tpu_xla_available()
    except ImportError:
        return False

class TPUSamplingContext:
    """Context manager for TPU-optimized sampling with step boundaries"""
    
    def __init__(self, 
                 model_id: str,
                 input_shape: Tuple[int, ...],
                 batch_size: int = 1,
                 cfg_scale: float = 7.0,
                 scheduler: str = "default",
                 steps: int = 20,
                 enable_compilation_cache: bool = True):
        
        self.model_id = model_id
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.cfg_scale = cfg_scale
        self.scheduler = scheduler
        self.steps = steps
        self.enable_compilation_cache = enable_compilation_cache
        
        self._cache_key = None
        self._is_cached = False
        self._step_counter = 0
        
        # Import TPU manager if available
        try:
            from .tpu_xla import tpu_manager, create_execution_signature
            self.tpu_manager = tpu_manager
            self.create_signature = create_execution_signature
            self.available = True
        except ImportError:
            self.available = False
    
    def __enter__(self):
        if not self.available:
            return self
            
        # Create compilation cache key
        self._cache_key = self.create_signature(
            model_id=self.model_id,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            cfg_scale=self.cfg_scale,
            scheduler=self.scheduler,
            steps=self.steps
        )
        
        # Check if already cached
        self._is_cached = self.tpu_manager.is_cached(self._cache_key)
        
        if self._is_cached:
            logging.info(f"TPU compilation cache HIT for {self.model_id}")
        else:
            logging.info(f"TPU compilation cache MISS for {self.model_id} - will compile")
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.available:
            return
            
        # Mark as compiled if execution completed successfully
        if exc_type is None and self._cache_key:
            self.tpu_manager.mark_compiled(self._cache_key)
            logging.info(f"TPU compilation completed for {self.model_id}")
    
    @contextmanager
    def step_boundary(self, step_name: str = "diffusion_step"):
        """Create step boundary for XLA optimization"""
        if not self.available:
            yield
            return
            
        step_id = f"{step_name}_{self._step_counter}"
        self._step_counter += 1
        
        try:
            with self.tpu_manager.step_context(step_id):
                yield
        except Exception as e:
            logging.warning(f"TPU step boundary failed: {e}")
            yield
    
    @contextmanager 
    def autocast_context(self, dtype: Optional[torch.dtype] = None):
        """TPU-optimized autocast context"""
        if not self.available:
            # Fallback for non-TPU
            with torch.autocast(device_type="cpu", dtype=dtype or torch.float32):
                yield
            return
            
        effective_dtype = dtype or torch.bfloat16
        with self.tpu_manager.autocast_context(effective_dtype):
            yield
    
    def is_cache_hit(self) -> bool:
        """Check if this execution signature was cached"""
        return self._is_cached
    
    def get_cache_key(self):
        """Get compilation cache key"""
        return self._cache_key

def create_tpu_sampling_context(model, 
                               latent_shape: Tuple[int, ...],
                               cfg_scale: float = 7.0,
                               scheduler: str = "default",
                               steps: int = 20) -> TPUSamplingContext:
    """Factory function to create TPU sampling context"""
    
    # Extract model identifier
    model_id = getattr(model, 'model_hash', 'unknown_model')
    if hasattr(model, 'model') and hasattr(model.model, '__class__'):
        model_id = model.model.__class__.__name__
    
    # Determine batch size from latent
    batch_size = latent_shape[0] if len(latent_shape) > 0 else 1
    
    return TPUSamplingContext(
        model_id=model_id,
        input_shape=latent_shape,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
        scheduler=scheduler,
        steps=steps
    )

def tpu_optimized_sampling_wrapper(original_sample_func):
    """Decorator to add TPU optimization to sampling functions"""
    
    def wrapper(model, noise, positive, negative, cfg, device, sampler, sigmas, 
                model_options=None, latent_image=None, denoise_mask=None, 
                callback=None, disable_pbar=False, seed=None, **kwargs):
        
        # Check if TPU optimization should be applied
        if not is_tpu_sampling_available():
            return original_sample_func(model, noise, positive, negative, cfg, device, sampler, sigmas,
                                     model_options, latent_image, denoise_mask, callback, disable_pbar, seed, **kwargs)
        
        from . import model_management
        if not model_management.is_tpu_device(device):
            return original_sample_func(model, noise, positive, negative, cfg, device, sampler, sigmas,
                                     model_options, latent_image, denoise_mask, callback, disable_pbar, seed, **kwargs)
        
        # Create TPU sampling context
        latent_shape = noise.shape if noise is not None else (1, 4, 64, 64)
        steps = len(sigmas) - 1 if sigmas is not None else 20
        
        tpu_context = create_tpu_sampling_context(
            model=model,
            latent_shape=latent_shape,
            cfg_scale=cfg,
            steps=steps
        )
        
        # Execute with TPU optimization
        with tpu_context:
            # Use autocast for mixed precision
            with tpu_context.autocast_context():
                # Move inputs to TPU device
                if hasattr(model_management.tpu_manager, 'to_device'):
                    if noise is not None:
                        noise = model_management.tpu_manager.to_device(noise)
                    if latent_image is not None:
                        latent_image = model_management.tpu_manager.to_device(latent_image)
                
                # Execute original sampling with step boundaries
                return original_sample_func(model, noise, positive, negative, cfg, device, sampler, sigmas,
                                         model_options, latent_image, denoise_mask, callback, disable_pbar, seed, **kwargs)
    
    return wrapper

def apply_tpu_step_boundary_to_unet(unet_forward):
    """Wrap UNet forward pass with TPU step boundary"""
    
    def wrapped_forward(*args, **kwargs):
        # Check if we should apply TPU optimization
        if not is_tpu_sampling_available():
            return unet_forward(*args, **kwargs)
        
        # Apply step boundary for each UNet call
        try:
            from .model_management import create_tpu_step_context
            with create_tpu_step_context("unet_forward"):
                return unet_forward(*args, **kwargs)
        except ImportError:
            return unet_forward(*args, **kwargs)
    
    return wrapped_forward

# Warmup utilities
def create_manual_warmup_workflow(model,
                                 target_shape: Tuple[int, ...] = (1, 4, 64, 64),
                                 cfg_scale: float = 7.0,
                                 scheduler: str = "euler",
                                 device = None) -> Dict[str, Any]:
    """Create a minimal workflow for manual TPU warmup"""
    
    if device is None:
        from . import model_management
        device = model_management.get_torch_device()
    
    # Create minimal inputs for warmup
    batch_size, channels, height, width = target_shape
    
    warmup_data = {
        "latent": torch.randn(target_shape, device=device, dtype=torch.bfloat16),
        "timestep": torch.tensor([500.0], device=device),
        "text_embedding": torch.randn((batch_size, 77, 768), device=device, dtype=torch.bfloat16),
        "cfg_scale": cfg_scale,
        "scheduler": scheduler
    }
    
    return warmup_data

def execute_manual_warmup(model, warmup_data: Dict[str, Any]) -> bool:
    """Execute manual warmup with given data"""
    
    try:
        if not is_tpu_sampling_available():
            logging.warning("TPU sampling not available for warmup")
            return False
        
        from . import model_management
        
        # Create warmup context
        tpu_context = create_tpu_sampling_context(
            model=model,
            latent_shape=warmup_data["latent"].shape,
            cfg_scale=warmup_data["cfg_scale"],
            steps=1  # Single step for warmup
        )
        
        with tpu_context:
            with tpu_context.step_boundary("warmup_step"):
                with tpu_context.autocast_context():
                    # Simple forward pass to trigger compilation
                    _ = model.model(
                        warmup_data["latent"],
                        warmup_data["timestep"],
                        warmup_data["text_embedding"]
                    )
        
        # Mark warmup as completed
        if hasattr(model_management, 'tpu_manager'):
            model_management.tpu_manager.complete_manual_warmup()
        
        logging.info("Manual TPU warmup completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Manual TPU warmup failed: {e}")
        return False