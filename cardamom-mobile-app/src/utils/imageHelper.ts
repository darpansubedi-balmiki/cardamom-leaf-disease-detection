/**
 * Image helper utilities
 */

/**
 * Convert data URI to blob for upload
 * @param dataURI - Data URI string
 */
export const dataURItoBlob = (dataURI: string): Blob => {
  const byteString = atob(dataURI.split(',')[1]);
  const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  
  return new Blob([ab], { type: mimeString });
};

/**
 * Format confidence score to percentage
 * @param confidence - Confidence value (0-1)
 * @param decimals - Number of decimal places
 */
export const formatConfidence = (confidence: number, decimals: number = 2): string => {
  return `${(confidence * 100).toFixed(decimals)}%`;
};

/**
 * Get severity color based on confidence
 * @param confidence - Confidence value (0-1)
 */
export const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return '#4caf50'; // Green - High confidence
  if (confidence >= 0.6) return '#ff9800'; // Orange - Medium confidence
  return '#f44336'; // Red - Low confidence
};

/**
 * Get severity badge color
 * @param severity - Severity level
 */
export const getSeverityColor = (severity: 'low' | 'medium' | 'high'): string => {
  switch (severity) {
    case 'low':
      return '#4caf50';
    case 'medium':
      return '#ff9800';
    case 'high':
      return '#f44336';
    default:
      return '#9e9e9e';
  }
};

/**
 * Get severity label in Nepali
 * @param severity - Severity level
 */
export const getSeverityLabelNepali = (severity: 'low' | 'medium' | 'high'): string => {
  switch (severity) {
    case 'low':
      return 'कम गम्भीर';
    case 'medium':
      return 'मध्यम गम्भीर';
    case 'high':
      return 'अत्यन्त गम्भीर';
    default:
      return 'अज्ञात';
  }
};
