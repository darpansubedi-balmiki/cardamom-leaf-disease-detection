/**
 * TypeScript type definitions for the Cardamom Disease Detection App
 */

export interface PredictionResponse {
  class_name: string;
  confidence: number;
  heatmap: string; // Base64 encoded PNG
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
