/**
 * API client for communicating with the backend server.
 */
import axios from 'axios';

// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Prediction response interface
export interface PredictionResponse {
  class_name: string;
  confidence: number;
  heatmap: string;
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
    if (!data || typeof data.confidence !== 'number' || !data.class_name) {
      console.error('Invalid API response:', data);
      throw new Error('Invalid response from server');
    }

    return data;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
