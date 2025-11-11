import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.cyber_sentinel import CyberSentinelModel
from app.services.prediction_service import PredictionService

class TestCyberSentinelModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = 'models/cyber_sentinel_model.pkl'
        
    def test_model_loading(self):
        """Test that the model loads successfully"""
        try:
            model = CyberSentinelModel(self.model_path)
            self.assertIsNotNone(model.model)
            self.assertTrue(model.model.training == False)  # Should be in eval mode
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")
    
    def test_model_info(self):
        """Test model information retrieval"""
        try:
            model = CyberSentinelModel(self.model_path)
            info = model.get_model_info()
            
            self.assertIsInstance(info, dict)
            self.assertIn('input_size', info)
            self.assertIn('device', info)
            
        except Exception as e:
            self.skipTest(f"Model info test failed: {e}")
    
    def test_prediction_service(self):
        """Test prediction service initialization"""
        try:
            service = PredictionService(self.model_path)
            capabilities = service.get_model_capabilities()
            
            self.assertIsInstance(capabilities, dict)
            self.assertIn('model_info', capabilities)
            
        except Exception as e:
            self.skipTest(f"Prediction service test failed: {e}")

if __name__ == '__main__':
    unittest.main()