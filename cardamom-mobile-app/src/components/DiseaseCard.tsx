/**
 * Disease Card Component
 */
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { DiseaseInfo } from '../types';
import { getSeverityColor, getSeverityLabelNepali } from '../utils/imageHelper';

interface DiseaseCardProps {
  disease: DiseaseInfo;
  onPress?: () => void;
}

export const DiseaseCard: React.FC<DiseaseCardProps> = ({ disease, onPress }) => {
  return (
    <TouchableOpacity
      style={styles.card}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <View style={styles.titleContainer}>
          <Text style={styles.nameEnglish}>{disease.nameEnglish}</Text>
          <Text style={styles.nameNepali}>{disease.nameNepali}</Text>
        </View>
        <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(disease.severity) }]}>
          <Text style={styles.severityText}>{getSeverityLabelNepali(disease.severity)}</Text>
        </View>
      </View>
      
      <Text style={styles.description} numberOfLines={3}>
        {disease.descriptionNepali}
      </Text>
      
      <View style={styles.footer}>
        <Text style={styles.viewMore}>थप जानकारी हेर्नुहोस् →</Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  titleContainer: {
    flex: 1,
  },
  nameEnglish: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  nameNepali: {
    fontSize: 16,
    fontWeight: '600',
    color: '#667eea',
  },
  severityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    marginLeft: 8,
  },
  severityText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: 'bold',
  },
  description: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 12,
  },
  footer: {
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
    paddingTop: 12,
  },
  viewMore: {
    fontSize: 14,
    color: '#667eea',
    fontWeight: '600',
  },
});
