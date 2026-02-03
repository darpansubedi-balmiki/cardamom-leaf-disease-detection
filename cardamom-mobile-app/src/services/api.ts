/**
 * API Client for communicating with the FastAPI backend
 */
import axios from 'axios';
import { PredictionResponse } from '../types';

// API Configuration
const API_BASE_URL = 'http://192.168.1.64:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

/**
 * Health check endpoint
 */
export const healthCheck = async (): Promise<{ status: string }> => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

/**
 * Send image for disease prediction
 * @param imageUri - Local file URI of the image
 * @returns Prediction response with class name, confidence, and heatmap
 */
export const predictDisease = async (imageUri: string): Promise<PredictionResponse> => {
  try {
    // Create FormData
    const formData = new FormData();

    // Extract filename from URI
    const filename = imageUri.split('/').pop() || 'image.jpg';

    // For React Native, we need to append the file with proper format
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: filename,
    } as any);

    const response = await apiClient.post<PredictionResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error: any) {
    console.error('Prediction failed:', error);

    // Handle specific error cases
    if (error.response) {
      // Server responded with error
      throw new Error(`Server error: ${error.response.data.detail || error.response.statusText}`);
    } else if (error.request) {
      // Request made but no response
      throw new Error('Cannot connect to server. Make sure the backend is running on http://localhost:8000');
    } else {
      // Something else happened
      throw new Error(`Error: ${error.message}`);
    }
  }
};

/**
 * Configure API base URL (useful for different environments)
 * @param baseUrl - New base URL
 */
export const setApiBaseUrl = (baseUrl: string) => {
  apiClient.defaults.baseURL = baseUrl;
};

export default {
  healthCheck,
  predictDisease,
  setApiBaseUrl,
};
