import torch
import numpy as np
import logging
from typing import Dict, List, Any, Union
from app.models.cyber_sentinel import CyberSentinelModel
from app.utils.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path: str):
        self.model = CyberSentinelModel(model_path)
        self.preprocessor = DataPreprocessor()
        self.model_info = self.model.get_model_info()
        
        logger.info("Prediction service initialized")
        logger.info(f"Model info: {self.model_info}")
    
    def preprocess_input(self, raw_data: Union[List, np.ndarray, Dict]) -> torch.Tensor:
        """Preprocess input data for the model"""
        return self.preprocessor.process(raw_data, self.model_info.get('input_size'))
    
    def predict(self, input_data: Union[List, np.ndarray, Dict]) -> Dict[str, Any]:
        """Make prediction with proper error handling"""
        try:
            # Preprocess input
            input_tensor = self.preprocess_input(input_data)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model.predict(input_tensor)
            
            # Convert to Python native types
            prediction_np = prediction.numpy()
            
            return {
                "success": True,
                "prediction": prediction_np.tolist(),
                "shape": list(prediction_np.shape),
                "model_info": self.model_info
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "prediction": None,
                "model_info": self.model_info
            }
    
    def batch_predict(self, batch_data: List) -> Dict[str, Any]:
        """Process multiple predictions"""
        try:
            processed_batch = [self.preprocess_input(data) for data in batch_data]
            input_tensor = torch.stack(processed_batch)
            
            with torch.no_grad():
                predictions = self.model.predict_batch(input_tensor)
            
            predictions_np = predictions.numpy()
            
            return {
                "success": True,
                "predictions": predictions_np.tolist(),
                "batch_size": len(batch_data),
                "shape": list(predictions_np.shape)
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": None
            }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get information about what the model can do"""
        return {
            "model_info": self.model_info,
            "supported_input_types": ["list", "numpy_array", "dict"],
            "batch_support": True,
            "device": self.model_info.get('device', 'cpu')
        }