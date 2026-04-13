/**
 * API client for communicating with the backend server.
 */
import axios from 'axios';

// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Prediction response interface – mirrors PredictResponse from the backend.
export interface PredictionResponse {
  // Core classification fields
  top_class: string;
  top_probability: number;
  top_probability_pct: number;
  is_uncertain: boolean;
  confidence_threshold: number;
  top_k: Array<{ class_name: string; probability: number; probability_pct: number }>;
  // Severity fields – present only when include_severity=true was sent
  heatmap: string | null;
  severity_stage: number | null;
  severity_percent: number | null;
  severity_method: string;
  // CAM method used ('gradcam' | 'gradcam++' | 'none')
  cam_method: string;
  // Model version from model_metadata.json (may be absent)
  model_version: string | null;
  warning: string[] | null; // Any warnings or notes about the prediction
}

// Health check response interface
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_classes: string[];
  device: string;
  model_version: string | null;
  model_accuracy: number | null;
}

/**
 * API client class with configured axios instance.
 */
class ApiClient {
  private client: ReturnType<typeof axios.create>;

  constructor(baseURL: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL,
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  /**
   * Health check endpoint.
   */
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  /**
   * Send image for disease prediction.
   *
   * @param file             - The image file to analyse.
   * @param includeSeverity  - When true, the response will include
   *                           severity_stage, severity_percent, severity_method
   *                           and a Grad-CAM heatmap overlay.
   * @param useTta           - When true, run Test-Time Augmentation on the backend.
   * @param camMethod        - 'gradcam' (default) or 'gradcam++'.
   */
  async predict(
    file: File,
    includeSeverity = true,
    useTta = false,
    camMethod: 'gradcam' | 'gradcam++' = 'gradcam',
  ): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('include_severity', includeSeverity ? 'true' : 'false');
    formData.append('use_tta', useTta ? 'true' : 'false');
    formData.append('cam_method', camMethod);

    const response = await this.client.post<PredictionResponse>(
      '/predict',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
