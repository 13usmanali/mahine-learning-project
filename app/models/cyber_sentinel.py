import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CyberSentinelModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self) -> None:
        """Load the Cyber Sentinel model from file"""
        try:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Log model information
            self._log_model_info()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _log_model_info(self) -> None:
        """Log model architecture and parameters"""
        if self.model is None:
            return
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Log layer information
        logger.info("Model layers:")
        for name, module in self.model.named_children():
            logger.info(f"  - {name}: {module}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and requirements"""
        if self.model is None:
            return {}
        
        sample_param = next(iter(self.model.parameters()))
        input_size = sample_param.shape[1] if len(sample_param.shape) > 1 else "Unknown"
        
        return {
            "input_size": input_size,
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters())
        }
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction with the model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            return output.cpu()
    
    def predict_batch(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Make batch predictions"""
        return self.predict(input_batch)