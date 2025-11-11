import torch
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_device_info() -> Dict[str, Any]:
    """Get information about available computing devices"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info.update({
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_cached': torch.cuda.memory_reserved()
        })
    
    return device_info

def optimize_model_performance(model: torch.nn.Module) -> torch.nn.Module:
    """Apply performance optimizations to the model"""
    # Enable cuDNN auto-tuner
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def convert_to_onnx(model: torch.nn.Module, input_size: int, output_path: str) -> bool:
    """Convert PyTorch model to ONNX format"""
    try:
        dummy_input = torch.randn(1, input_size)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model converted to ONNX: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        return False

def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model size in different units"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'parameters_mb': param_size / 1024**2,
        'buffers_mb': buffer_size / 1024**2,
        'total_mb': size_all_mb
    }