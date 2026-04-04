/**
 * API Client for communicating with the FastAPI backend
 */
import axios from 'axios';
import { Platform } from 'react-native';
import { PredictionResponse } from '../types';

// API Configuration
// On web (browser) the backend is on the same machine → localhost.
// On native (iOS/Android) devices the backend is on the LAN → use the
// machine's local IP address.  Change the IP below if your machine's IP
// is different.
const API_BASE_URL =
  Platform.OS === 'web' ? 'http://localhost:8000' : 'http://192.168.1.64:8000';

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
 * Build a FormData payload from an image URI that works on both web and
 * native.
 *
 * On native React Native the XHR adapter accepts the `{ uri, type, name }`
 * object directly.  In a browser the XHR adapter requires a real Blob/File,
 * so we fetch the blob: URI first to materialise the data.
 */
const buildImageFormData = async (imageUri: string): Promise<FormData> => {
  const formData = new FormData();
  const filename = imageUri.split('/').pop() || 'image.jpg';

  if (Platform.OS === 'web') {
    // On web, expo-image-picker returns a blob: or data: URI.
    // We must convert it to a real Blob so the browser XHR adapter can
    // send it correctly.
    const response = await fetch(imageUri);
    const blob = await response.blob();
    const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
    formData.append('file', file);
  } else {
    // React Native native format – accepted by RN's XHR adapter.
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: filename,
    } as any);
  }

  return formData;
};

/**
 * Send image for disease prediction
 * @param imageUri - Local file URI of the image
 * @returns Prediction response with classification, severity, heatmap, and advice fields
 */
export const predictDisease = async (imageUri: string): Promise<PredictionResponse> => {
  try {
    const formData = await buildImageFormData(imageUri);

    // Request severity estimation to match web frontend behaviour
    formData.append('include_severity', 'true');

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
      throw new Error('Cannot connect to server. Make sure the backend is running.');
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
