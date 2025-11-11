from flask import Blueprint, request, jsonify, render_template
import logging
import numpy as np
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

bp = Blueprint('api', __name__)

# Initialize prediction service
prediction_service = PredictionService('models/cyber_sentinel_model.pkl')

@bp.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@bp.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "service": "cyber_sentinel"
    })

@bp.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    capabilities = prediction_service.get_model_capabilities()
    return jsonify(capabilities)

@bp.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction"""
    try:
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({
                "success": False,
                "error": "No input data provided. Use 'input' key."
            }), 400
        
        input_data = data['input']
        
        # Make prediction
        result = prediction_service.predict(input_data)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@bp.route('/api/predict/batch', methods=['POST'])
def batch_predict():
    """Make batch predictions"""
    try:
        data = request.get_json()
        
        if not data or 'inputs' not in data:
            return jsonify({
                "success": False,
                "error": "No input data provided. Use 'inputs' key for batch processing."
            }), 400
        
        batch_data = data['inputs']
        
        if not isinstance(batch_data, list):
            return jsonify({
                "success": False,
                "error": "Batch data must be a list of inputs"
            }), 400
        
        # Make batch prediction
        result = prediction_service.batch_predict(batch_data)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@bp.route('/api/example', methods=['GET'])
def get_example():
    """Get example input format"""
    model_info = prediction_service.get_model_capabilities()
    input_size = model_info['model_info'].get('input_size', 'unknown')
    
    example_input = {
        "description": "Example input format for the model",
        "input_size": input_size,
        "single_prediction": {
            "input": [0.1] * (input_size if isinstance(input_size, int) else 10)
        },
        "batch_prediction": {
            "inputs": [
                [0.1] * (input_size if isinstance(input_size, int) else 10),
                [0.2] * (input_size if isinstance(input_size, int) else 10)
            ]
        }
    }
    
    return jsonify(example_input)