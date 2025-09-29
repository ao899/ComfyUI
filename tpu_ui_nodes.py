"""
TPU/XLA Control Nodes for ComfyUI
Provides UI controls for manual warmup, compilation caching, and TPU settings
"""

import logging
import torch
from typing import Dict, Any, Tuple

class TPUWarmupNode:
    """Manual TPU warmup control node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_warmup": ("BOOLEAN", {"default": True}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "scheduler": (["euler", "ddim", "dpm2", "karras", "normal"], {"default": "euler"}),
                "force_recompile": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "warmup_status")
    FUNCTION = "execute_warmup"
    CATEGORY = "advanced/tpu"
    
    def execute_warmup(self, model, enable_warmup, target_width, target_height, batch_size, cfg_scale, scheduler="euler", force_recompile=False):
        try:
            from comfy import model_management
            
            # Check TPU availability
            if not model_management.is_tpu():
                return (model, "TPU not available - skipping warmup")
                
            if not enable_warmup:
                return (model, "TPU warmup disabled by user")
            
            # Check if TPU sampling is available
            try:
                from comfy.tpu_sampling import (
                    is_tpu_sampling_available,
                    create_manual_warmup_workflow,
                    execute_manual_warmup,
                    create_tpu_sampling_context
                )
                
                if not is_tpu_sampling_available():
                    return (model, "TPU sampling utilities not available")
                
            except ImportError as e:
                return (model, f"TPU sampling import failed: {e}")
            
            # Create execution signature
            target_shape = (batch_size, 4, target_height // 8, target_width // 8)
            
            tpu_context = create_tpu_sampling_context(
                model=model,
                latent_shape=target_shape,
                cfg_scale=cfg_scale,
                scheduler=scheduler,
                steps=1  # Single step for warmup
            )
            
            # Check if already warmed up
            if not force_recompile and tpu_context.is_cache_hit():
                return (model, f"TPU already warmed up for signature: {target_width}x{target_height}, batch={batch_size}")
            
            # Execute warmup
            logging.info(f"Starting TPU manual warmup: {target_width}x{target_height}, batch={batch_size}")
            warmup_start = __import__('time').time()
            
            warmup_data = create_manual_warmup_workflow(
                model=model,
                target_shape=target_shape,
                cfg_scale=cfg_scale,
                scheduler=scheduler
            )
            
            success = execute_manual_warmup(model, warmup_data)
            warmup_time = __import__('time').time() - warmup_start
            
            if success:
                status = f"TPU warmup completed in {warmup_time:.2f}s for {target_width}x{target_height}"
                logging.info(status)
                return (model, status)
            else:
                status = f"TPU warmup failed after {warmup_time:.2f}s"
                logging.warning(status)
                return (model, status)
                
        except Exception as e:
            error_msg = f"TPU warmup error: {str(e)}"
            logging.error(error_msg)
            return (model, error_msg)

class TPUCompilationCacheNode:
    """TPU compilation cache control and monitoring node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["get_stats", "clear_cache", "cache_info"], {"default": "get_stats"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_info",)
    FUNCTION = "manage_cache"
    CATEGORY = "advanced/tpu"
    
    def manage_cache(self, action):
        try:
            from comfy.tpu_xla import get_tpu_stats, tpu_manager
            
            if action == "get_stats":
                stats = get_tpu_stats()
                info = []
                info.append(f"TPU Available: {stats.get('available', False)}")
                info.append(f"Device: {stats.get('device', 'None')}")
                info.append(f"Cached Signatures: {stats.get('cached_signatures', 0)}")
                info.append(f"Warmup Completed: {stats.get('warmup_completed', False)}")
                if stats.get('warmup_signature'):
                    info.append(f"Warmup Signature: {stats['warmup_signature']}")
                return ("\n".join(info),)
                
            elif action == "clear_cache":
                tpu_manager.clear_cache()
                return ("TPU compilation cache cleared",)
                
            elif action == "cache_info":
                stats = get_tpu_stats()
                cache_count = stats.get('cached_signatures', 0)
                return (f"TPU has {cache_count} cached compilation signatures",)
                
            else:
                return (f"Unknown action: {action}",)
                
        except Exception as e:
            return (f"TPU cache management error: {str(e)}",)

class TPUDeviceInfoNode:
    """TPU device information and status node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {}
    
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("device_info", "is_available", "xla_version")
    FUNCTION = "get_device_info"
    CATEGORY = "advanced/tpu"
    
    def get_device_info(self):
        try:
            from comfy import model_management
            
            is_tpu = model_management.is_tpu()
            device = model_management.get_torch_device()
            
            info_lines = []
            info_lines.append(f"TPU Available: {is_tpu}")
            info_lines.append(f"Current Device: {device}")
            
            # Get XLA version if available
            xla_version = "Not available"
            try:
                import torch_xla
                xla_version = getattr(torch_xla, '__version__', 'Unknown version')
                info_lines.append(f"XLA Version: {xla_version}")
                
                # Get TPU device details
                if is_tpu:
                    import torch_xla.core.xla_model as xm
                    ordinal = xm.get_ordinal()
                    info_lines.append(f"TPU Ordinal: {ordinal}")
                    
            except ImportError:
                info_lines.append("torch_xla not available")
            
            # Check advanced TPU features
            try:
                from comfy.tpu_xla import tpu_manager
                info_lines.append("Advanced TPU features: Available")
                info_lines.append(f"XLA Manager: {type(tpu_manager).__name__}")
            except ImportError:
                info_lines.append("Advanced TPU features: Not available")
            
            device_info = "\n".join(info_lines)
            
            return (device_info, is_tpu, xla_version)
            
        except Exception as e:
            error_info = f"Error getting TPU info: {str(e)}"
            return (error_info, False, "Error")

class TPUModelOptimizationNode:
    """TPU model optimization and configuration node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "autocast_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "enable_compilation_cache": ("BOOLEAN", {"default": True}),
                "shape_padding": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_optimization": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("optimized_model", "optimization_info")
    FUNCTION = "optimize_model"
    CATEGORY = "advanced/tpu"
    
    def optimize_model(self, model, autocast_dtype, enable_compilation_cache, shape_padding, custom_optimization=""):
        try:
            from comfy import model_management
            
            if not model_management.is_tpu():
                return (model, "TPU not available - no optimization applied")
            
            # Set TPU-specific optimizations
            optimization_info = []
            optimization_info.append(f"TPU Model Optimization Applied:")
            optimization_info.append(f"- Autocast dtype: {autocast_dtype}")
            optimization_info.append(f"- Compilation cache: {enable_compilation_cache}")
            optimization_info.append(f"- Shape padding: {shape_padding}")
            
            # Apply optimizations to model
            if hasattr(model, 'model_options'):
                if model.model_options is None:
                    model.model_options = {}
                
                # Set TPU-specific model options
                model.model_options['tpu_autocast_dtype'] = autocast_dtype
                model.model_options['tpu_compilation_cache'] = enable_compilation_cache
                model.model_options['tpu_shape_padding'] = shape_padding
                
            if custom_optimization:
                optimization_info.append(f"- Custom: {custom_optimization}")
            
            optimization_info.append("TPU optimizations configured")
            
            return (model, "\n".join(optimization_info))
            
        except Exception as e:
            error_msg = f"TPU model optimization error: {str(e)}"
            return (model, error_msg)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TPU Manual Warmup": TPUWarmupNode,
    "TPU Compilation Cache": TPUCompilationCacheNode,
    "TPU Device Info": TPUDeviceInfoNode,
    "TPU Model Optimization": TPUModelOptimizationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TPU Manual Warmup": "TPU Manual Warmup",
    "TPU Compilation Cache": "TPU Cache Manager",
    "TPU Device Info": "TPU Device Info",
    "TPU Model Optimization": "TPU Model Optimizer",
}