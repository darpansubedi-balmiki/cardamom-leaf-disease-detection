/**
 * Heatmap Viewer Component
 */
import React from 'react';
import { View, Image, StyleSheet, Text } from 'react-native';

interface HeatmapViewerProps {
  heatmapBase64: string;
  style?: any;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({ heatmapBase64, style }) => {
  return (
    <View style={[styles.container, style]}>
      <Text style={styles.title}>ग्रेड-CAM हिटम्याप (Grad-CAM Heatmap)</Text>
      <Text style={styles.subtitle}>
        यो दृश्यले देखाउँछ कि कुन क्षेत्रहरूले भविष्यवाणीमा प्रभाव पारेको छ
      </Text>
      <Image
        source={{ uri: `data:image/png;base64,${heatmapBase64}` }}
        style={styles.heatmap}
        resizeMode="contain"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 12,
    color: '#666',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  heatmap: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 8,
  },
});
