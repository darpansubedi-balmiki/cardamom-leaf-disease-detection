import { useState } from 'react';
import type { ChangeEvent } from 'react';
import { apiClient } from './api/client';
import type { PredictionResponse } from './api/client';
import './App.css';

/** Human-readable stage labels. */
const STAGE_LABELS: Record<number, string> = {
  0: 'Stage 0 – Healthy (no lesion)',
  1: 'Stage 1 – Mild (1–10 %)',
  2: 'Stage 2 – Moderate (11–25 %)',
  3: 'Stage 3 – Severe (26–50 %)',
  4: 'Stage 4 – Very Severe (> 50 %)',
};

/** Colour class for each stage badge. */
const STAGE_COLOURS: Record<number, string> = {
  0: 'stage-healthy',
  1: 'stage-mild',
  2: 'stage-moderate',
  3: 'stage-severe',
  4: 'stage-very-severe',
};
interface PredictionResponse {
  class_name: string;
  confidence: number;
  heatmap: string;
  model_trained?: boolean;
  warning?: string | null;
}

type Advice = {
  nepaliName: string;
  prevention: string[];
  cure: string[];
};

const ADVICE_MAP: Record<string, Advice> = {
  'Colletotrichum Blight': {
    nepaliName: 'पात डढ्ने रोग',
    prevention: [
      'बोटबीचको दूरी मिलाएर हावापानी चल्ने बनाउनुहोस् (घना छायाँ/आर्द्रता कम गर्ने)।',
      'बिरामी पात/डाँठ तुरुन्त हटाएर नष्ट गर्नुहोस् (खेतमै थुपारेर नराख्नुहोस्)।',
      'बगैँचामा पानी जम्न नदिनुहोस्; ड्रेनेज राम्रो बनाउनुहोस्।',
      'बेलाबेलामा झार सफा गरी खेत सफा राख्नुहोस्।',
      'नर्सरी/रोपाइँ सामग्री स्वस्थ र रोगमुक्त प्रयोग गर्नुहोस्।',
    ],
    cure: [
      'संक्रमित पात काटेर हटाउनुहोस् र खेत बाहिर नष्ट गर्नुहोस्।',
      'रोग बढिरहेको बेला माथिबाट पानी छर्किने सिंचाइ (overhead irrigation) कम गर्नुहोस्।',
      'स्थानीय कृषि प्राविधिकको सल्लाह अनुसार उपयुक्त फफूँदनाशक (fungicide) छर्कनुहोस् (डोज/अन्तराल पालन गर्नुहोस्)।',
      'फेरि–फेरि एउटै औषधि मात्र नदोहो¥याई समूह परिवर्तन (rotation) गर्नुहोस् ताकि प्रतिरोध (resistance) नबढोस्।',
    ],
  },

  'Phyllosticta Leaf Spot': {
    nepaliName: 'पातमा दाग लाग्ने रोग',
    prevention: [
      'बोटलाई घना हुन नदिन छाँटाइ/सफाइ गरी हावा चल्ने बनाउनुहोस्।',
      'बिरामी पात संकलन गरेर नष्ट गर्नुहोस्।',
      'पात लामो समय भिजिरहने अवस्था (लगातार आर्द्रता) कम गर्ने व्यवस्थापन गर्नुहोस्।',
      'सन्तुलित मलखाद प्रयोग गर्नुहोस्; अत्यधिक नाइट्रोजनबाट नरम पात बढी संवेदनशील हुन सक्छ।',
    ],
    cure: [
      'रोग लागेको भाग काटेर हटाउनुहोस्।',
      'रोग फैलिएको अवस्थामा कृषि प्राविधिकको सल्लाह अनुसार उपयुक्त फफूँदनाशक छर्कनुहोस्।',
      'छर्काइपछि पनि सुधार नआएमा रोग पहिचान/डोज पुनः पुष्टि गर्न नजिकको कृषि कार्यालय/प्रयोगशालासँग परामर्श गर्नुहोस्।',
    ],
  },

  Healthy: {
    nepaliName: 'स्वस्थ (रोग देखिएको छैन)',
    prevention: [
      'बगैँचा सफा राख्नुहोस्, झारपात नियन्त्रण गर्नुहोस्।',
      'पानी जम्न नदिनुहोस्; राम्रो निकास (drainage) बनाउनुहोस्।',
      'बोटलाई धेरै घना हुन नदिन हल्का छाँटाइ गरी हावा चल्ने बनाउनुहोस्।',
      'सन्तुलित मलखाद/जैविक मल प्रयोग गर्नुहोस् र माटोको स्वास्थ्य सुधार गर्नुहोस्।',
      'समय–समयमा पात निरीक्षण गर्नुहोस्—दाग/किरा देखिनासाथ छिटो व्यवस्थापन गर्नुहोस्।',
    ],
    cure: [
      'हाल उपचार आवश्यक छैन।',
      'यदि भविष्यमा दाग/डढाइ/पहेंलोपन बढेमा फोटोसहित पुन: परीक्षण गर्नुहोस् वा कृषि प्राविधिकसँग सल्लाह लिनुहोस्।',
    ],
  },

  Uncertain: {
    nepaliName: 'अनिश्चित',
    prevention: [
      'फोटो स्पष्ट (फोकस भएको) र पात नजिकबाट खिचेर पुन: प्रयास गर्नुहोस्।',
      'धेरै पात/पृष्ठभूमि नआउने गरी एउटै पात केन्द्रमा राख्नुहोस्।',
      'प्राकृतिक उज्यालोमा फोटो खिच्नुहोस् (धेरै अँध्यारो/धेरै glare भएमा नतिजा बिग्रिन सक्छ)।',
    ],
    cure: [
      'यदि रोग शंका छ भने नजिकको कृषि प्राविधिक/कृषि कार्यालयमा देखाएर पुष्टि गर्नुहोस्।',
    ],
  },
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const advice = result ? ADVICE_MAP[result.class_name] : undefined;

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

  const hasSeverity =
    result !== null &&
    result.severity_stage !== null &&
    result.severity_percent !== null;

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Cardamom Leaf Disease Detection</h1>
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

          {selectedFile && <span className="file-name">{selectedFile.name}</span>}
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

        {error && <div className="error-message">⚠️ {error}</div>}

        {result && (
          <div className="results-section">
            <h2>Results</h2>

            {result.warning && (
              <div className="error-message" style={{ marginBottom: 12 }}>
                {result.warning}
              </div>
            )}

            <div className="result-card">
              <div className="result-info">
                {/* Disease class */}
                <div className="result-item">
                  <span className="label">Disease Class:</span>
                  <span className="value class-name">{result.top_class}</span>
                  <span className="value class-name">
                    {result.class_name}
                    {advice ? ` — ${advice.nepaliName}` : ''}
                  </span>
                </div>

                {/* Confidence */}
                <div className="result-item">
                  <span className="label">Confidence:</span>
                  <span className="value confidence">
                    {result.top_probability_pct.toFixed(2)}%
                  </span>
                </div>

                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.top_probability_pct}%` }}
                  ></div>
                </div>

                {/* Uncertainty warning */}
                {result.is_uncertain && (
                  <div className="uncertainty-warning">
                    ⚠️ Low confidence – prediction may be unreliable.
                  </div>
                )}

                {/* Severity section */}
                {hasSeverity && (
                  <div className="severity-section">
                    <h3>Severity Estimation</h3>

                    <div className="result-item">
                      <span className="label">Stage:</span>
                      <span
                        className={`severity-badge ${STAGE_COLOURS[result.severity_stage!]}`}
                      >
                        {STAGE_LABELS[result.severity_stage!] ?? `Stage ${result.severity_stage}`}
                      </span>
                    </div>

                    <div className="result-item">
                      <span className="label">Area Affected:</span>
                      <span className="value severity-percent">
                        {result.severity_percent!.toFixed(1)}%
                      </span>
                    </div>

                    <div className="severity-bar">
                      <div
                        className="severity-fill"
                        style={{ width: `${result.severity_percent}%` }}
                      ></div>
                    </div>

                    {result.severity_method === 'heuristic' && (
                      <p className="severity-disclaimer">
                        ℹ️ <strong>Estimate only.</strong> Severity was approximated from the
                        Grad-CAM heatmap (heuristic method) and does not reflect true lesion
                        area. For accurate quantification, use mask-based labelling.
                      </p>
                    )}
                  </div>
                )}

                {advice && (
                  <div className="recommendation-section">
                    <h3>सुझाव</h3>

                    <div className="advice-block">
                      <h4>रोकथाम (Prevention)</h4>
                      <ul>
                        {advice.prevention.map((tip, idx) => (
                          <li key={`prev-${idx}`}>{tip}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="advice-block">
                      <h4>उपचार/व्यवस्थापन (Cure / Management)</h4>
                      <ul>
                        {advice.cure.map((tip, idx) => (
                          <li key={`cure-${idx}`}>{tip}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="advice-block">
                      <h4>सूचना</h4>
                      <p style={{ margin: 0 }}>
                        यी सुझावहरू सामान्य जानकारीका लागि हुन्। स्थानीय कृषि प्राविधिक/कृषि
                        कार्यालयको सल्लाह अनुसार मात्र औषधि/छर्काइ प्रयोग गर्नुहोस्।
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Grad-CAM heatmap (returned with severity) */}
              {result.heatmap && (
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
              )}
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