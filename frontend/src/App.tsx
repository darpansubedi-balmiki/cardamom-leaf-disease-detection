import { useState, useCallback, useRef } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { apiClient } from "./api/client";
import type { PredictionResponse } from "./api/client";
import { ADVICE_MAP } from "./utils/AdviceMap";
import { STAGE_BADGE_TW, STAGE_LABELS } from "./utils/StageLabel";


export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const advice = result ? ADVICE_MAP[result.top_class] : undefined;

  const applyFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file");
      return;
    }
    setSelectedFile(file);
    setError("");
    setResult(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) applyFile(file);
  };

  /* ---- Drag-and-drop handlers ---- */
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) applyFile(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await apiClient.predict(selectedFile);
      setResult(response);
    } catch (err: any) {
      console.error("Prediction error:", err);

      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === "ECONNABORTED") {
        setError("Request timeout. Please try again.");
      } else if (err.code === "ERR_NETWORK") {
        setError("Cannot connect to server. Make sure the backend is running.");
      } else {
        setError("An error occurred during analysis. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl("");
    setError("");
    setResult(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  };

  const hasSeverity =
    result !== null && result.severity_stage !== null && result.severity_percent !== null;

  return (
    <div className="min-h-screen bg-linear-to-br from-indigo-500 to-purple-700 px-4 py-8">
      <div className="mx-auto max-w-225 rounded-2xl bg-white p-6 shadow-[0_20px_60px_rgba(0,0,0,0.3)] sm:p-10">
        {/* Header */}
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-bold text-slate-800 sm:text-4xl">
            Cardamom Leaf Disease Detection
          </h1>
          <p className="mt-2 text-base text-slate-600 sm:text-lg">
            Upload a cardamom leaf image to detect diseases
          </p>
        </header>

        {/* Drop zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={[
            "mb-6 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-6 py-10 transition",
            isDragging
              ? "border-indigo-500 bg-indigo-50"
              : "border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-indigo-50/40",
          ].join(" ")}
        >
          <svg className="h-10 w-10 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-sm text-slate-600">
            <span className="font-semibold text-indigo-600">Click to choose</span>
            {" "}or drag &amp; drop an image here
          </p>
          {selectedFile && (
            <span className="max-w-xs truncate text-xs text-slate-500">{selectedFile.name}</span>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          id="file-input"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />

        {/* Preview + Heatmap side-by-side */}
        {(previewUrl || (result && result.heatmap)) && (
          <div className={`grid gap-6 mb-8 ${result?.heatmap ? "sm:grid-cols-2" : "grid-cols-1"}`}>
            {previewUrl && (
              <div className="text-center">
                <h3 className="text-lg font-semibold text-slate-800">Original Image</h3>
                <p className="mt-1 text-xs italic text-slate-500">
                  The image you uploaded
                </p>
                <div className="relative max-w-full mx-auto mt-3">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="mx-auto max-h-80 w-full max-w-full rounded-xl object-contain"
                  />
                  {isLoading && (
                    <div className="absolute inset-0 rounded-xl overflow-hidden">
                      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
                      <div className="absolute left-0 w-full h-1 bg-green-400 animate-scan" />
                    </div>
                  )}
                </div>
              </div>
            )}

            {result?.heatmap && (
              <div className="text-center">
                <h3 className="text-lg font-semibold text-slate-800">
                  {result.cam_method === "gradcam++" ? "Grad-CAM++" : "Grad-CAM"} Heatmap
                </h3>
                <p className="mt-1 text-xs italic text-slate-500">
                  Regions that influenced the prediction
                </p>
                <img
                  src={`data:image/png;base64,${result.heatmap}`}
                  alt="Grad-CAM Heatmap"
                  className="mx-auto mt-3 max-h-80 w-full rounded-xl object-contain"
                />
              </div>
            )}
          </div>
        )}

        {/* Action buttons */}
        <div className="mb-8 flex flex-wrap justify-center gap-3">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || isLoading}
            className="inline-flex items-center gap-2 rounded-full bg-linear-to-br from-indigo-500 to-purple-700 px-10 py-3 text-base font-semibold text-white shadow-md transition hover:-translate-y-0.5 hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isLoading ? (
              <>
                <span className="h-4.5 w-4.5 animate-spin rounded-full border-[3px] border-white/30 border-t-white" />
                Analyzing...
              </>
            ) : (
              "Analyze"
            )}
          </button>

          {(selectedFile || result) && (
            <button
              onClick={handleReset}
              className="inline-flex items-center justify-center rounded-full bg-slate-100 px-10 py-3 text-base font-semibold text-slate-800 transition hover:-translate-y-0.5 hover:bg-slate-200"
            >
              Reset
            </button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 rounded-lg border-2 border-red-200 bg-red-50 p-4 text-center font-medium text-red-700">
            ⚠️ {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-10">
            <h2 className="mb-6 text-center text-2xl font-bold text-slate-800">Results</h2>

            {result.warning && (
              <div className="mb-3 rounded-lg border-2 border-amber-200 bg-amber-50 p-4 text-center font-medium text-amber-800">
                {result.warning}
              </div>
            )}

            <div className="rounded-xl bg-linear-to-br from-slate-50 to-indigo-100 p-6 shadow-md sm:p-8">
              <div className="space-y-4">
                {/* Disease class */}
                <div className="flex flex-col gap-2 rounded-lg bg-white p-3 sm:flex-row sm:items-center sm:justify-between">
                  <span className="text-base font-semibold text-slate-600">Disease Class:</span>
                  <span className="text-lg font-bold text-indigo-600">
                    {result.top_class}
                    {advice ? ` — ${advice.nepaliName}` : ""}
                  </span>
                </div>

                {/* Confidence */}
                <div className="flex flex-col gap-2 rounded-lg bg-white p-3 sm:flex-row sm:items-center sm:justify-between">
                  <span className="text-base font-semibold text-slate-600">Confidence:</span>
                  <span className="text-lg font-bold text-purple-700">
                    {result.top_probability_pct.toFixed(2)}%
                  </span>
                </div>

                {/* Confidence bar */}
                <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full bg-linear-to-r from-indigo-500 to-purple-700 transition-[width] duration-500 ease-out"
                    style={{ width: `${result.top_probability_pct}%` }}
                  />
                </div>

                {/* Uncertainty — enhanced: show top-K bars */}
                {result.is_uncertain && (
                  <div className="rounded-lg border border-amber-400 bg-amber-100 px-4 py-4 text-sm text-amber-900 space-y-3">
                    <p className="font-semibold">
                      ⚠️ Low confidence – the model is unsure. Please retake the photo in better lighting
                      or from a closer angle.
                    </p>
                    <p className="text-xs text-amber-800">Top candidates considered:</p>
                    <div className="space-y-2">
                      {result.top_k.map((item) => (
                        <div key={item.class_name}>
                          <div className="flex justify-between text-xs mb-0.5">
                            <span>{item.class_name}</span>
                            <span>{item.probability_pct.toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full rounded bg-amber-200 overflow-hidden">
                            <div
                              className="h-full rounded bg-amber-500 transition-[width] duration-500"
                              style={{ width: `${item.probability_pct}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Top-K breakdown (always shown) */}
                {!result.is_uncertain && result.top_k.length > 1 && (
                  <details className="rounded-lg border border-slate-200 bg-white p-3">
                    <summary className="cursor-pointer text-sm font-semibold text-slate-600">
                      All predictions
                    </summary>
                    <div className="mt-3 space-y-2">
                      {result.top_k.map((item) => (
                        <div key={item.class_name}>
                          <div className="flex justify-between text-xs mb-0.5 text-slate-700">
                            <span>{item.class_name}</span>
                            <span>{item.probability_pct.toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full rounded bg-slate-200 overflow-hidden">
                            <div
                              className="h-full rounded bg-indigo-400 transition-[width] duration-500"
                              style={{ width: `${item.probability_pct}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}

                {/* Severity section */}
                {hasSeverity && result.top_class !== "Healthy" && (
                  <div className="mt-6 border-t-2 border-dashed border-slate-200 pt-5">
                    <h3 className="mb-3 text-lg font-semibold text-slate-800">
                      Severity Estimation
                    </h3>

                    <div className="flex flex-col gap-2 rounded-lg bg-white p-3 sm:flex-row sm:items-center sm:justify-between">
                      <span className="text-base font-semibold text-slate-600">Stage:</span>
                      <span
                        className={[
                          "inline-flex rounded-full px-3 py-1 text-sm font-semibold",
                          STAGE_BADGE_TW[result.severity_stage!] ?? "bg-slate-100 text-slate-800",
                        ].join(" ")}
                      >
                        {STAGE_LABELS[result.severity_stage!] ??
                          `Stage ${result.severity_stage}`}
                      </span>
                    </div>

                    <div className="flex flex-col gap-2 rounded-lg bg-white p-3 sm:flex-row sm:items-center sm:justify-between">
                      <span className="text-base font-semibold text-slate-600">
                        Area Affected:
                      </span>
                      <span className="text-lg font-bold text-red-700">
                        {result.severity_percent!.toFixed(1)}%
                      </span>
                    </div>

                    <div className="mt-2 h-2.5 w-full overflow-hidden rounded bg-slate-200">
                      <div
                        className="h-full rounded bg-linear-to-r from-emerald-600 via-amber-500 to-red-600 transition-[width] duration-300 ease-out"
                        style={{ width: `${result.severity_percent}%` }}
                      />
                    </div>

                    {result.severity_method === "heuristic" && (
                      <p className="mt-3 rounded-lg border border-sky-200 bg-sky-50 px-4 py-3 text-sm leading-relaxed text-sky-900">
                        ℹ️ <strong>Estimate only.</strong> Severity was approximated from the
                        {result.cam_method === "gradcam++" ? " Grad-CAM++ " : " Grad-CAM "}
                        heatmap (heuristic method) and does not reflect true lesion
                        area. For accurate quantification, use mask-based labelling.
                      </p>
                    )}
                  </div>
                )}

                {/* Recommendations */}
                {advice && (
                  <div className="mt-6 rounded-xl border border-indigo-100 bg-indigo-50/50 p-4 sm:p-5">
                    <h3 className="text-lg font-semibold text-slate-900">सुझाव</h3>

                    <div className="mt-4 space-y-4 text-slate-800">
                      <div>
                        <h4 className="mb-2 font-semibold">रोकथाम (Prevention)</h4>
                        <ul className="list-disc space-y-1 pl-5">
                          {advice.prevention.map((tip, idx) => (
                            <li key={`prev-${idx}`}>{tip}</li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="mb-2 font-semibold">उपचार/व्यवस्थापन (Cure / Management)</h4>
                        <ul className="list-disc space-y-1 pl-5">
                          {advice.cure.map((tip, idx) => (
                            <li key={`cure-${idx}`}>{tip}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 leading-relaxed text-red-900">
                        ℹ️ <strong>सूचना: </strong>
                        यी सुझावहरू सामान्य जानकारीका लागि हुन्। स्थानीय कृषि
                        प्राविधिक/कृषि कार्यालयको सल्लाह अनुसार मात्र औषधि/छर्काइ
                        प्रयोग गर्नुहोस्।
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 border-t-2 border-slate-100 pt-6 text-center text-sm text-slate-400">
          <p>Designed &amp; Developed by Darpan Subedi</p>
        </footer>
      </div>
      <style>{`
        @keyframes scan {
          0% { top: 0%; }
          100% { top: 100%; }
        }
        .animate-scan {
          animation: scan 2s linear infinite;
        }
      `}</style>
    </div>
  );
}