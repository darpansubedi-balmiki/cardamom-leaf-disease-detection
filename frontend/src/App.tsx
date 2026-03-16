import { useRef, useState, useCallback } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import './App.css'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TopKItem {
  class_name: string
  probability: number
  probability_pct: number
}

interface PredictResponse {
  top_class: string
  top_probability: number
  top_probability_pct: number
  is_uncertain: boolean
  confidence_threshold: number
  top_k: TopKItem[]
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
const CLASS_COLOR: Record<string, string> = {
  'Colletotrichum Blight': '#e74c3c',
  'Healthy': '#27ae60',
  'Phyllosticta Leaf Spot': '#f39c12',
  Uncertain: '#95a5a6',
}

function classColor(name: string): string {
  return CLASS_COLOR[name] ?? '#3498db'
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.60)
  const [topK, setTopK] = useState(3)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File) => {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      setError('Please upload a JPEG, PNG, or WebP image.')
      return
    }
    setError(null)
    setResult(null)
    setImageFile(file)
    setImageUrl(URL.createObjectURL(file))
  }, [])

  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(true)
  }
  const onDragLeave = () => setDragging(false)

  const handleAnalyse = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('file', imageFile)
      form.append('confidence_threshold', String(confidenceThreshold))
      form.append('top_k', String(topK))

      const resp = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form })
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(detail?.detail ?? resp.statusText)
      }
      const data: PredictResponse = await resp.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImageUrl(null)
    setImageFile(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌿 Cardamom Leaf Disease Detection</h1>
        <p className="subtitle">Upload a leaf image to identify diseases with AI</p>
      </header>

      <main className="app-main">
        {/* Upload area */}
        <section className="upload-section">
          <div
            className={`dropzone${dragging ? ' dropzone--active' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            role="button"
            tabIndex={0}
            aria-label="Upload leaf image"
            onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
          >
            {imageUrl ? (
              <img src={imageUrl} alt="Uploaded leaf" className="preview-image" />
            ) : (
              <div className="dropzone-placeholder">
                <span className="dropzone-icon">📷</span>
                <p>Drop a leaf image here or click to browse</p>
                <p className="dropzone-hint">JPEG, PNG, WebP supported</p>
              </div>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={onInputChange}
            style={{ display: 'none' }}
          />
        </section>

        {/* Settings */}
        <section className="settings-section">
          <h2>Settings</h2>
          <div className="settings-grid">
            <label className="setting-item">
              <span>
                Confidence threshold:{' '}
                <strong>{(confidenceThreshold * 100).toFixed(0)}%</strong>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                aria-label="Confidence threshold"
              />
              <span className="setting-hint">
                Predictions below this threshold are shown as "Uncertain"
              </span>
            </label>

            <label className="setting-item">
              <span>
                Top-K results: <strong>{topK}</strong>
              </span>
              <input
                type="range"
                min={1}
                max={3}
                step={1}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                aria-label="Number of top predictions"
              />
            </label>
          </div>
        </section>

        {/* Actions */}
        <div className="actions">
          <button
            className="btn btn--primary"
            onClick={handleAnalyse}
            disabled={!imageFile || loading}
          >
            {loading ? 'Analysing…' : 'Analyse Leaf'}
          </button>
          {imageUrl && (
            <button className="btn btn--secondary" onClick={handleReset}>
              Reset
            </button>
          )}
        </div>

        {/* Error */}
        {error && <div className="alert alert--error" role="alert">⚠️ {error}</div>}

        {/* Results */}
        {result && (
          <section className="results-section" aria-live="polite">
            <h2>Results</h2>

            {/* Primary prediction */}
            <div
              className={`primary-prediction${result.is_uncertain ? ' primary-prediction--uncertain' : ''}`}
              style={{ borderColor: classColor(result.top_class) }}
            >
              <div className="primary-label">
                {result.is_uncertain && (
                  <span className="badge badge--uncertain">⚠ Low Confidence</span>
                )}
                <span
                  className="disease-class"
                  style={{ color: classColor(result.top_class) }}
                >
                  {result.top_class}
                </span>
              </div>
              <div className="primary-probability">
                {result.top_probability_pct.toFixed(1)}% confidence
              </div>
              {result.is_uncertain && (
                <p className="uncertain-note">
                  The model's top prediction is below the{' '}
                  {(result.confidence_threshold * 100).toFixed(0)}% threshold.
                  Consider reviewing the top-{topK} predictions below.
                </p>
              )}
            </div>

            {/* Top-K list */}
            <div className="topk-section">
              <h3>Top-{result.top_k.length} Predictions</h3>
              <ul className="topk-list">
                {result.top_k.map((item, idx) => (
                  <li key={item.class_name} className="topk-item">
                    <span className="topk-rank">#{idx + 1}</span>
                    <span className="topk-class">{item.class_name}</span>
                    <div className="topk-bar-container">
                      <div
                        className="topk-bar"
                        style={{
                          width: `${item.probability_pct}%`,
                          backgroundColor: classColor(item.class_name),
                        }}
                        role="progressbar"
                        aria-valuenow={item.probability_pct}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-label={`${item.class_name}: ${item.probability_pct.toFixed(1)}%`}
                      />
                    </div>
                    <span className="topk-pct">{item.probability_pct.toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by EfficientNetV2 · Cardamom Disease Detection System</p>
      </footer>
    </div>
  )
}
