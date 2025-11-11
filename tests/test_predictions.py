import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.prediction_service import PredictionService

class TestPredictions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = 'models/cyber_sentinel_model.pkl'
        try:
            self.service = PredictionService(self.model_path)
            self.model_info = self.service.get_model_capabilities()
        except Exception as e:
            self.skipTest(f"Service initialization failed: {e}")
    
    def test_single_prediction(self):
        """Test single prediction"""
        input_size = self.model_info['model_info'].get('input_size', 10)
        
        # Create dummy input
        dummy_input = [0.5] * input_size
        
        result = self.service.predict(dummy_input)
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['prediction'])
        self.assertIsInstance(result['prediction'], list)
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        input_size = self.model_info['model_info'].get('input_size', 10)
        
        # Create dummy batch
        dummy_batch = [
            [0.1] * input_size,
            [0.2] * input_size,
            [0.3] * input_size
        ]
        
        result = self.service.batch_predict(dummy_batch)
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['predictions'])
        self.assertEqual(result['batch_size'], 3)

if __name__ == '__main__':
    unittest.main()