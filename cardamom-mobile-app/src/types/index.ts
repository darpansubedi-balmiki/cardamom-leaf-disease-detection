/**
 * TypeScript type definitions for the Cardamom Disease Detection App
 */

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
  heatmap?: string; // Base64 encoded PNG
  model_trained?: boolean;
  warning?: string;
}

export interface DiseaseInfo {
  id: string;
  nameEnglish: string;
  nameNepali: string;
  descriptionNepali: string;
  symptomsNepali: string[];
  causesNepali: string;
  treatmentNepali: string[];
  preventionNepali: string[];
  whenToActNepali: string;
  severity: 'low' | 'medium' | 'high';
  imageUrl?: string;
}

export type RootStackParamList = {
  Home: undefined;
  Camera: undefined;
  Result: {
    imageUri: string;
    prediction: PredictionResponse;
  };
  DiseaseInfo: {
    diseaseId: string;
  };
};
