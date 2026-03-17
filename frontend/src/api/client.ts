/**
 * API client for communicating with the backend server.
 */
import axios from 'axios';

// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Prediction response interface
// Backend response structure
export interface TopKPrediction {
  class_name: string;
  probability: number;
  probability_pct: number;
}

export interface PredictionResponse {
  top_class: string;
  top_probability: number;
  top_probability_pct: number;
  is_uncertain: boolean;
  confidence_threshold: number;
  top_k: TopKPrediction[];
  heatmap?: string;
  model_trained?: boolean;
  warning?: string;
}

// Health check response interface
export interface HealthResponse {
  status: string;
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
   */
  async predict(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<PredictionResponse>(
      '/predict',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    // Validate response data
    const data = response.data;
    if (!data || typeof data.top_probability !== 'number' || !data.top_class) {
      console.error('Invalid API response:', data);
      throw new Error('Invalid response from server: missing or invalid prediction data');
    }

    return data;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
