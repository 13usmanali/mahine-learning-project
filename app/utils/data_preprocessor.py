import numpy as np
import torch
import logging
from typing import Union, List, Dict, Any

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.required_input_size = None
    
    def process(self, raw_data: Union[List, np.ndarray, Dict], input_size: Any = None) -> torch.Tensor:
        """Process raw data into model-ready tensor"""
        try:
            # Convert to numpy array first
            if isinstance(raw_data, Dict):
                array_data = self._dict_to_array(raw_data)
            elif isinstance(raw_data, List):
                array_data = np.array(raw_data, dtype=np.float32)
            elif isinstance(raw_data, np.ndarray):
                array_data = raw_data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported data type: {type(raw_data)}")
            
            # Ensure correct shape and type
            tensor_data = self._array_to_tensor(array_data, input_size)
            
            return tensor_data
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _dict_to_array(self, data_dict: Dict) -> np.ndarray:
        """Convert dictionary to numpy array"""
        # Extract values and convert to array
        # You might need to adjust this based on your data structure
        if 'features' in data_dict:
            return np.array(data_dict['features'], dtype=np.float32)
        else:
            # Use all values if no specific key
            return np.array(list(data_dict.values()), dtype=np.float32)
    
    def _array_to_tensor(self, array_data: np.ndarray, input_size: Any) -> torch.Tensor:
        """Convert array to properly shaped tensor"""
        tensor_data = torch.from_numpy(array_data).float()
        
        # Handle different input dimensions
        if tensor_data.dim() == 1:
            # Single sample, add batch dimension
            tensor_data = tensor_data.unsqueeze(0)
        
        # Reshape if input size is known and doesn't match
        if input_size and isinstance(input_size, int):
            if tensor_data.shape[1] != input_size:
                logger.warning(f"Input shape {tensor_data.shape} doesn't match expected size {input_size}")
                # You might want to implement padding/truncation here
        
        return tensor_data
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features (adjust based on your model's needs)"""
        # Implement normalization logic here
        # This is a placeholder - adjust based on your data
        return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    def validate_input_shape(self, tensor: torch.Tensor, expected_size: int) -> bool:
        """Validate input tensor shape"""
        return tensor.shape[1] == expected_size