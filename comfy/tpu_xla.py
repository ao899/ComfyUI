"""
ComfyUI TPU/XLA Backend Support
Provides advanced TPU integration with PyTorch/XLA including:
- Manual warmup and compilation caching
- Shape-aware graph reuse
- Step boundary optimization
- Mixed precision for TPU
"""

import logging
import time
import torch
import hashlib
import weakref
from typing import Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from contextlib import contextmanager
import threading

# XLA imports with safe fallback
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    XLA_AVAILABLE = True
    logging.info("PyTorch/XLA detected and available")
except ImportError:
    XLA_AVAILABLE = False
    torch_xla = None
    xm = None
    met = None
    logging.info("PyTorch/XLA not available, TPU features disabled")

@dataclass
class CompilationCacheKey:
    """Key for XLA compilation caching based on execution signature"""
    model_id: str
    input_shape: Tuple[int, ...]
    batch_size: int
    cfg_scale: float
    has_guidance: bool
    vae_mode: str
    scheduler: str
    steps: int
    dtype: str
    
    def __post_init__(self):
        # Normalize shapes to common sizes to increase cache hits
        self.input_shape = self._normalize_shape(self.input_shape)
    
    def _normalize_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Normalize shapes to common resolutions to improve cache hits"""
        if len(shape) >= 2:  # Assume [H, W, ...] format
            h, w = shape[-2], shape[-1]
            # Round to common multiples of 64 for better cache hits
            h_norm = ((h + 63) // 64) * 64
            w_norm = ((w + 63) // 64) * 64
            return shape[:-2] + (h_norm, w_norm)
        return shape
    
    def __hash__(self):
        return hash((
            self.model_id, self.input_shape, self.batch_size,
            self.cfg_scale, self.has_guidance, self.vae_mode,
            self.scheduler, self.steps, self.dtype
        ))

@dataclass 
class WarmupState:
    """Tracks manual warmup completion and signatures"""
    completed: bool = False
    cache_key: Optional[CompilationCacheKey] = None
    timestamp: float = 0.0
    step_count: int = 0

class TPUXLAManager:
    """Advanced TPU/XLA management for ComfyUI"""
    
    def __init__(self):
        self.is_available = XLA_AVAILABLE
        self.device = None
        self.compilation_cache: Dict[CompilationCacheKey, bool] = {}
        self.warmup_state = WarmupState()
        self._lock = threading.RLock()
        self._step_contexts = {}
        self._current_autocast_dtype = torch.bfloat16
        
        if self.is_available:
            try:
                self.device = xm.xla_device()
                # Try to get ordinal, fallback if not available
                try:
                    self.device_ordinal = xm.get_ordinal()
                except AttributeError:
                    # Older versions of torch_xla may not have get_ordinal
                    self.device_ordinal = 0
                logging.info(f"TPU/XLA initialized: {self.device} (ordinal: {self.device_ordinal})")
            except Exception as e:
                logging.warning(f"TPU/XLA device initialization failed: {e}")
                self.is_available = False
    
    def get_device(self) -> torch.device:
        """Get XLA device or fallback"""
        if self.is_available and self.device:
            return self.device
        return torch.device("cpu")  # Fallback
    
    def is_tpu_device(self, device: Union[torch.device, str, None]) -> bool:
        """Check if device is TPU/XLA"""
        if not self.is_available:
            return False
        if device is None:
            return False
        if isinstance(device, str):
            return device.startswith("xla")
        return hasattr(device, 'type') and device.type == "xla"
    
    def to_device(self, tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """Move tensor to TPU device with optimal settings"""
        if not self.is_available:
            return tensor
        
        target_device = self.get_device()
        if tensor.device == target_device:
            return tensor
            
        # XLA doesn't support non_blocking transfers
        return tensor.to(target_device, non_blocking=False)
    
    def get_optimal_dtype(self, requested_dtype: Optional[torch.dtype] = None) -> torch.dtype:
        """Get optimal dtype for TPU operations"""
        if not self.is_available:
            return requested_dtype or torch.float32
        
        # TPU prefers bfloat16 for most operations
        if requested_dtype in [torch.float16, torch.bfloat16, None]:
            return torch.bfloat16
        return requested_dtype or torch.bfloat16
    
    @contextmanager
    def autocast_context(self, dtype: Optional[torch.dtype] = None):
        """TPU-optimized autocast context"""
        if not self.is_available:
            # Fallback to CPU/CUDA autocast
            with torch.autocast(device_type="cpu", dtype=dtype or torch.float32):
                yield
            return
        
        effective_dtype = dtype or self._current_autocast_dtype
        try:
            # XLA-specific autocast when available
            if hasattr(torch, 'autocast') and hasattr(torch.autocast, '__call__'):
                with torch.autocast(device_type="cpu", dtype=effective_dtype, enabled=True):
                    yield
            else:
                yield
        except Exception as e:
            logging.warning(f"TPU autocast failed, using no-op context: {e}")
            yield
    
    @contextmanager 
    def step_context(self, name: str = "default"):
        """XLA step boundary context for optimal compilation"""
        if not self.is_available:
            yield
            return
        
        step_id = f"{name}_{id(threading.current_thread())}"
        
        try:
            # Mark step boundary for XLA
            yield
        finally:
            if self.is_available:
                try:
                    # Ensure XLA operations are flushed
                    xm.mark_step()
                except Exception as e:
                    logging.warning(f"XLA step marking failed: {e}")
    
    def create_cache_key(self, 
                        model_id: str,
                        input_shape: Tuple[int, ...],
                        batch_size: int = 1,
                        cfg_scale: float = 7.0,
                        has_guidance: bool = True,
                        vae_mode: str = "decode",
                        scheduler: str = "default",
                        steps: int = 20,
                        dtype: torch.dtype = torch.bfloat16) -> CompilationCacheKey:
        """Create compilation cache key from execution parameters"""
        return CompilationCacheKey(
            model_id=model_id,
            input_shape=input_shape,
            batch_size=batch_size,
            cfg_scale=cfg_scale,
            has_guidance=has_guidance,
            vae_mode=vae_mode,
            scheduler=scheduler,
            steps=steps,
            dtype=str(dtype)
        )
    
    def is_cached(self, cache_key: CompilationCacheKey) -> bool:
        """Check if execution signature is already compiled"""
        with self._lock:
            return cache_key in self.compilation_cache
    
    def mark_compiled(self, cache_key: CompilationCacheKey):
        """Mark execution signature as compiled"""
        with self._lock:
            self.compilation_cache[cache_key] = True
            logging.debug(f"Marked TPU compilation cached: {cache_key.model_id} - {cache_key.input_shape}")
    
    def start_manual_warmup(self, cache_key: CompilationCacheKey):
        """Start manual warmup process"""
        with self._lock:
            self.warmup_state.completed = False
            self.warmup_state.cache_key = cache_key
            self.warmup_state.timestamp = time.time()
            self.warmup_state.step_count = 0
            logging.info(f"Starting manual TPU warmup for: {cache_key.model_id}")
    
    def complete_manual_warmup(self):
        """Complete manual warmup and mark cache"""
        with self._lock:
            if self.warmup_state.cache_key:
                self.mark_compiled(self.warmup_state.cache_key)
                duration = time.time() - self.warmup_state.timestamp
                logging.info(f"TPU manual warmup completed in {duration:.2f}s")
            
            self.warmup_state.completed = True
    
    def should_skip_warmup(self, cache_key: CompilationCacheKey) -> bool:
        """Check if warmup should be skipped for this signature"""
        with self._lock:
            # Skip if already warmed up with same signature
            if (self.warmup_state.completed and 
                self.warmup_state.cache_key == cache_key):
                return True
            
            # Skip if signature is already cached
            return self.is_cached(cache_key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get compilation cache statistics"""
        with self._lock:
            return {
                "available": self.is_available,
                "device": str(self.device) if self.device else None,
                "cached_signatures": len(self.compilation_cache),
                "warmup_completed": self.warmup_state.completed,
                "warmup_signature": str(self.warmup_state.cache_key) if self.warmup_state.cache_key else None
            }
    
    def synchronize(self):
        """Synchronize XLA operations"""
        if not self.is_available:
            return
            
        try:
            xm.wait_device_ops()
        except Exception as e:
            logging.warning(f"TPU synchronization failed: {e}")
    
    def clear_cache(self):
        """Clear compilation cache (for debugging)"""
        with self._lock:
            self.compilation_cache.clear()
            self.warmup_state = WarmupState()
            logging.info("TPU compilation cache cleared")

# Global TPU manager instance
tpu_manager = TPUXLAManager()

# Convenience functions
def is_tpu_available() -> bool:
    """Check if TPU is available"""
    return tpu_manager.is_available

def get_tpu_device() -> torch.device:
    """Get TPU device"""
    return tpu_manager.get_device()

def is_tpu_device(device) -> bool:
    """Check if device is TPU"""
    return tpu_manager.is_tpu_device(device)

def to_tpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to TPU"""
    return tpu_manager.to_device(tensor)

def tpu_autocast(dtype: Optional[torch.dtype] = None):
    """TPU autocast context"""
    return tpu_manager.autocast_context(dtype)

def tpu_step(name: str = "default"):
    """TPU step context"""
    return tpu_manager.step_context(name)

def create_execution_signature(model_id: str, **kwargs) -> CompilationCacheKey:
    """Create execution signature for caching"""
    return tpu_manager.create_cache_key(model_id, **kwargs)

def should_skip_warmup(cache_key: CompilationCacheKey) -> bool:
    """Check if warmup should be skipped"""
    return tpu_manager.should_skip_warmup(cache_key)

def complete_warmup():
    """Mark warmup as completed"""
    tpu_manager.complete_manual_warmup()

def get_tpu_stats() -> Dict[str, Any]:
    """Get TPU statistics"""
    return tpu_manager.get_cache_stats()