/**
 * TypeScript type definitions for the Cardamom Disease Detection App
 */

export interface PredictionResponse {
  // Core classification fields
  top_class: string;
  top_probability: number;
  top_probability_pct: number;
  is_uncertain: boolean;
  confidence_threshold: number;
  top_k: Array<{ class_name: string; probability: number; probability_pct: number }>;
  // Heatmap and severity fields – present only when include_severity=true was sent
  heatmap: string | null;
  severity_stage: number | null;
  severity_percent: number | null;
  severity_method: string;
  warning: string[] | null;
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
