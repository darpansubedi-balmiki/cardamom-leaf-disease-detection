import { useState, ChangeEvent } from 'react';
import { apiClient } from './api/client';
import './App.css';

interface PredictionResponse {
  class_name: string;
  confidence: number;
  heatmap: string;
}

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
      const response = await apiClient.predict(selectedFile);
      setResult(response);
    } catch (err: any) {
      console.error('Prediction error:', err);

      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.');
      } else if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to server. Make sure the backend is running.');
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

            <div className="result-card">
              <div className="result-info">
                <div className="result-item">
                  <span className="label">Disease Class:</span>
                  <span className="value class-name">{result.class_name}</span>
                </div>

                <div className="result-item">
                  <span className="label">Confidence:</span>
                  <span className="value confidence">
                    {(result.confidence * 100).toFixed(2)}%
                  </span>
                </div>

                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="heatmap-section">
                <h3>Grad-CAM Heatmap</h3>
                <p className="heatmap-description">
                  This visualization shows which regions of the leaf influenced the prediction
                </p>
                <img
                  src={`data:image/png;base64,${result.heatmap}`}
                  alt="Grad-CAM Heatmap"
                  className="heatmap-image"
                />
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
