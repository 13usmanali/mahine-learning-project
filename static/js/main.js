// Cyber Sentinel App JavaScript

class CyberSentinelApp {
    constructor() {
        this.apiBaseUrl = window.location.origin + '/api';
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadModelInfo();
    }

    bindEvents() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });

        // Batch prediction
        document.getElementById('batchPredictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleBatchPrediction();
        });

        // Example input
        document.getElementById('loadExample').addEventListener('click', () => {
            this.loadExampleInput();
        });
    }

    async loadModelInfo() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/model/info`);
            const data = await response.json();
            
            this.displayModelInfo(data);
        } catch (error) {
            console.error('Failed to load model info:', error);
            this.showError('Failed to load model information');
        }
    }

    displayModelInfo(modelInfo) {
        const infoContainer = document.getElementById('modelInfo');
        
        if (modelInfo.model_info) {
            const info = modelInfo.model_info;
            infoContainer.innerHTML = `
                <div class="info-item">
                    <strong>Input Size:</strong> ${info.input_size || 'Unknown'}
                </div>
                <div class="info-item">
                    <strong>Device:</strong> ${info.device || 'Unknown'}
                </div>
                <div class="info-item">
                    <strong>Model Type:</strong> ${info.model_type || 'Unknown'}
                </div>
                <div class="info-item">
                    <strong>Parameters:</strong> ${info.parameters ? info.parameters.toLocaleString() : 'Unknown'}
                </div>
            `;
        }
    }

    async handlePrediction() {
        const inputText = document.getElementById('inputData').value;
        const resultsDiv = document.getElementById('predictionResults');
        const loadingDiv = document.getElementById('loading');
        
        this.showLoading(loadingDiv);
        resultsDiv.innerHTML = '';

        try {
            // Parse input data
            let inputData;
            try {
                inputData = JSON.parse(inputText);
            } catch {
                // If not JSON, try as space-separated numbers
                inputData = inputText.trim().split(/\s+/).map(Number);
            }

            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ input: inputData }),
            });

            const result = await response.json();
            this.displayPredictionResult(result, resultsDiv);

        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Prediction failed: ' + error.message, resultsDiv);
        } finally {
            this.hideLoading(loadingDiv);
        }
    }

    async handleBatchPrediction() {
        const inputText = document.getElementById('batchInputData').value;
        const resultsDiv = document.getElementById('batchPredictionResults');
        const loadingDiv = document.getElementById('batchLoading');
        
        this.showLoading(loadingDiv);
        resultsDiv.innerHTML = '';

        try {
            const inputData = JSON.parse(inputText);

            const response = await fetch(`${this.apiBaseUrl}/predict/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ inputs: inputData }),
            });

            const result = await response.json();
            this.displayBatchPredictionResult(result, resultsDiv);

        } catch (error) {
            console.error('Batch prediction error:', error);
            this.showError('Batch prediction failed: ' + error.message, resultsDiv);
        } finally {
            this.hideLoading(loadingDiv);
        }
    }

    displayPredictionResult(result, container) {
        if (result.success) {
            container.innerHTML = `
                <div class="prediction-result success">
                    <h3>Prediction Successful</h3>
                    <p><strong>Output Shape:</strong> [${result.shape.join(', ')}]</p>
                    <div class="prediction-values">
                        <strong>Prediction:</strong>
                        <pre>${JSON.stringify(result.prediction, null, 2)}</pre>
                    </div>
                </div>
            `;
        } else {
            this.showError(result.error || 'Prediction failed', container);
        }
    }

    displayBatchPredictionResult(result, container) {
        if (result.success) {
            container.innerHTML = `
                <div class="prediction-result success">
                    <h3>Batch Prediction Successful</h3>
                    <p><strong>Batch Size:</strong> ${result.batch_size}</p>
                    <p><strong>Output Shape:</strong> [${result.shape.join(', ')}]</p>
                    <div class="prediction-values">
                        <strong>Predictions:</strong>
                        <pre>${JSON.stringify(result.predictions, null, 2)}</pre>
                    </div>
                </div>
            `;
        } else {
            this.showError(result.error || 'Batch prediction failed', container);
        }
    }

    async loadExampleInput() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/example`);
            const example = await response.json();
            
            document.getElementById('inputData').value = JSON.stringify(example.single_prediction.input, null, 2);
            document.getElementById('batchInputData').value = JSON.stringify(example.batch_prediction.inputs, null, 2);
            
        } catch (error) {
            console.error('Failed to load example:', error);
            this.showError('Failed to load example input');
        }
    }

    showLoading(loadingElement) {
        loadingElement.style.display = 'block';
    }

    hideLoading(loadingElement) {
        loadingElement.style.display = 'none';
    }

    showError(message, container = null) {
        if (container) {
            container.innerHTML = `
                <div class="prediction-result error">
                    <h3>Error</h3>
                    <p>${message}</p>
                </div>
            `;
        } else {
            // Show global error notification
            alert('Error: ' + message);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CyberSentinelApp();
});