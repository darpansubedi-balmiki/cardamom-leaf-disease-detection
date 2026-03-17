import { useState, ChangeEvent } from 'react';
import { apiClient, PredictionResponse } from './api/client';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<PredictionResponse | null>(null);

  /**
   * Handle file selection.
   */
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file');
        return;
      }

      setSelectedFile(file);
      setError('');
      setResult(null);

      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  /**
   * Handle image analysis.
   */
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      console.log('Sending image to API:', selectedFile.name);
      const response = await apiClient.predict(selectedFile);
      console.log('Received response:', response);
      
      // Validate response
      if (!response || typeof response.confidence !== 'number') {
        throw new Error('Invalid response from server: missing or invalid confidence value');
      }
      
      setResult(response);
    } catch (err: any) {
      console.error('Prediction error:', err);
      console.error('Error details:', {
        message: err.message,
        response: err.response?.data,
        status: err.response?.status,
        code: err.code
      });

      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.');
      } else if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to server. Make sure the backend is running at http://localhost:8000');
      } else if (err.message) {
        setError(err.message);
      } else {
        setError('An error occurred during analysis. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Reset all state.
   */
  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setError('');
    setResult(null);

    // Clean up preview URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🌿 Cardamom Leaf Disease Detection</h1>
          <p>Upload a cardamom leaf image to detect diseases</p>
        </header>

        <div className="upload-section">
          <input
            type="file"
            id="file-input"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            📁 Choose Image
          </label>

          {selectedFile && (
            <span className="file-name">{selectedFile.name}</span>
          )}
        </div>

        {previewUrl && (
          <div className="preview-section">
            <h3>Preview</h3>
            <img src={previewUrl} alt="Preview" className="preview-image" />
          </div>
        )}

        <div className="action-buttons">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || isLoading}
            className="btn btn-primary"
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              'Analyze'
            )}
          </button>

          {(selectedFile || result) && (
            <button onClick={handleReset} className="btn btn-secondary">
              Reset
            </button>
          )}
        </div>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {result && (
          <div className="results-section">
            <h2>Results</h2>

            {result.warning && (
              <div className="warning-message" style={{
                backgroundColor: '#fff3cd',
                border: '1px solid #ffc107',
                borderRadius: '8px',
                padding: '12px 16px',
                marginBottom: '16px',
                color: '#856404'
              }}>
                {result.warning}
              </div>
            )}

            <div className="result-card">
              <div className="result-info">
                <div className="result-item">
                  <span className="label">Disease Class:</span>
                  <span className="value class-name">
                    {result.top_class || 'Unknown'}
                    {result.is_uncertain && ' (Uncertain)'}
                  </span>
                </div>

                <div className="result-item">
                  <span className="label">Confidence:</span>
                  <span className="value confidence">
                    {typeof result.top_probability === 'number' 
                      ? `${(result.top_probability * 100).toFixed(2)}%`
                      : 'N/A'
                    }
                  </span>
                </div>

                {typeof result.top_probability === 'number' && (
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{ width: `${result.top_probability * 100}%` }}
                    ></div>
                  </div>
                )}
              </div>

              {/* Show all predictions from top_k */}
              {result.top_k && result.top_k.length > 0 && (
                <div className="all-predictions">
                  <h4>All Predictions</h4>
                  <div className="predictions-list">
                    {result.top_k.map((pred, index) => (
                      <div key={index} className="prediction-item">
                        <div className="prediction-header">
                          <span className="prediction-name">{pred.class_name}</span>
                          <span className="prediction-probability">
                            {pred.probability_pct.toFixed(2)}%
                          </span>
                        </div>
                        <div className="prediction-bar">
                          <div
                            className="prediction-fill"
                            style={{ 
                              width: `${pred.probability_pct}%`,
                              backgroundColor: index === 0 ? '#4CAF50' : '#ddd'
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.is_uncertain && (
                <div className="uncertainty-warning" style={{
                  backgroundColor: '#fff3cd',
                  border: '1px solid #ffc107',
                  borderRadius: '8px',
                  padding: '12px 16px',
                  marginTop: '16px',
                  color: '#856404'
                }}>
                  ⚠️ The model is uncertain about this prediction (confidence below {(result.confidence_threshold * 100).toFixed(0)}%). 
                  Please verify with an expert.
                </div>
              )}

              <div className="heatmap-section">
                <h3>Grad-CAM Heatmap</h3>
                <p className="heatmap-description">
                  This visualization shows which regions of the leaf influenced the prediction
                </p>
                {result.heatmap ? (
                  <img
                    src={`data:image/png;base64,${result.heatmap}`}
                    alt="Grad-CAM Heatmap"
                    className="heatmap-image"
                  />
                ) : (
                  <p>Heatmap not available</p>
                )}
              </div>
            </div>
          </div>
        )}

        <footer className="footer">
          <p>Designed & Developed by Darpan Subedi</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
