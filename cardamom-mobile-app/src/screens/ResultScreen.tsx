/**
 * Result Screen
 * Display prediction results with heatmap
 */
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { RootStackParamList } from '../types';
import { ImagePreview } from '../components/ImagePreview';
import { HeatmapViewer } from '../components/HeatmapViewer';
import { DiseaseCard } from '../components/DiseaseCard';
import { getDiseaseInfoByName } from '../data/diseaseInfo';
import { formatConfidence, getConfidenceColor } from '../utils/imageHelper';

type ResultScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Result'>;
type ResultScreenRouteProp = RouteProp<RootStackParamList, 'Result'>;

interface ResultScreenProps {
  navigation: ResultScreenNavigationProp;
  route: ResultScreenRouteProp;
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export const ResultScreen: React.FC<ResultScreenProps> = ({ navigation, route }) => {
  const { imageUri, prediction } = route.params;
  const diseaseInfo = getDiseaseInfoByName(prediction.class_name);

  const handleViewDetails = () => {
    if (diseaseInfo) {
      navigation.navigate('DiseaseInfo', { diseaseId: diseaseInfo.id });
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>परिणाम</Text>
        <View style={styles.placeholder} />
      </View>

      {/* Original Image */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>तस्बिर (Original Image)</Text>
        <ImagePreview imageUri={imageUri} />
      </View>

      {/* Prediction Result */}
      <View style={styles.resultCard}>
        <Text style={styles.resultLabel}>रोगको नाम (Disease)</Text>
        <Text style={styles.diseaseNameEnglish}>{prediction.class_name}</Text>
        {diseaseInfo && (
          <Text style={styles.diseaseNameNepali}>{diseaseInfo.nameNepali}</Text>
        )}

        <View style={styles.confidenceContainer}>
          <View style={styles.confidenceHeader}>
            <Text style={styles.confidenceLabel}>विश्वास स्तर (Confidence)</Text>
            <Text
              style={[
                styles.confidenceValue,
                { color: getConfidenceColor(prediction.confidence) },
              ]}
            >
              {formatConfidence(prediction.confidence)}
            </Text>
          </View>
          <View style={styles.confidenceBar}>
            <View
              style={[
                styles.confidenceFill,
                {
                  width: `${prediction.confidence * 100}%`,
                  backgroundColor: getConfidenceColor(prediction.confidence),
                },
              ]}
            />
          </View>
        </View>

        {prediction.confidence < 0.6 && (
          <View style={styles.warningBox}>
            <Ionicons name="warning" size={20} color="#f57c00" />
            <Text style={styles.warningText}>
              कम विश्वास स्तर। कृपया स्पष्ट तस्बिरसँग पुन: प्रयास गर्नुहोस्।
            </Text>
          </View>
        )}
      </View>

      {/* Heatmap */}
      <HeatmapViewer heatmapBase64={prediction.heatmap} />

      {/* Disease Information Card */}
      {diseaseInfo && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>रोग जानकारी</Text>
          <DiseaseCard disease={diseaseInfo} onPress={handleViewDetails} />
        </View>
      )}

      {/* Action Buttons */}
      <View style={styles.actionsContainer}>
        <TouchableOpacity
          style={[styles.actionButton, styles.primaryButton]}
          onPress={handleViewDetails}
        >
          <Ionicons name="information-circle" size={24} color="#fff" />
          <Text style={styles.actionButtonText}>विस्तृत जानकारी</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.secondaryButton]}
          onPress={() => navigation.navigate('Home')}
        >
          <Ionicons name="camera" size={24} color="#667eea" />
          <Text style={styles.actionButtonTextSecondary}>नयाँ तस्बिर</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  contentContainer: {
    paddingBottom: 32,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  placeholder: {
    width: 40,
  },
  section: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  resultCard: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 20,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  resultLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  diseaseNameEnglish: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  diseaseNameNepali: {
    fontSize: 20,
    fontWeight: '600',
    color: '#667eea',
    marginBottom: 20,
  },
  confidenceContainer: {
    marginTop: 16,
  },
  confidenceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#666',
  },
  confidenceValue: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  warningBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff3e0',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  warningText: {
    flex: 1,
    fontSize: 13,
    color: '#f57c00',
    marginLeft: 8,
  },
  actionsContainer: {
    padding: 16,
    gap: 12,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    gap: 8,
  },
  primaryButton: {
    backgroundColor: '#667eea',
  },
  secondaryButton: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#667eea',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
  },
  actionButtonTextSecondary: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#667eea',
  },
});
