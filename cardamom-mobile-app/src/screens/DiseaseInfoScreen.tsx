/**
 * Disease Info Screen
 * Detailed disease information in Nepali
 */
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { RootStackParamList } from '../types';
import { diseaseDatabase } from '../data/diseaseInfo';
import { getSeverityColor, getSeverityLabelNepali } from '../utils/imageHelper';

type DiseaseInfoScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'DiseaseInfo'>;
type DiseaseInfoScreenRouteProp = RouteProp<RootStackParamList, 'DiseaseInfo'>;

interface DiseaseInfoScreenProps {
  navigation: DiseaseInfoScreenNavigationProp;
  route: DiseaseInfoScreenRouteProp;
}

export const DiseaseInfoScreen: React.FC<DiseaseInfoScreenProps> = ({ navigation, route }) => {
  const { diseaseId } = route.params;
  const disease = diseaseDatabase[diseaseId];

  if (!disease) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>रोग जानकारी फेला परेन</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>{disease.nameEnglish}</Text>
          <Text style={styles.headerSubtitle}>{disease.nameNepali}</Text>
        </View>
      </View>

      {/* Severity Badge */}
      <View style={styles.severityContainer}>
        <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(disease.severity) }]}>
          <Ionicons name="alert-circle" size={20} color="#fff" />
          <Text style={styles.severityText}>{getSeverityLabelNepali(disease.severity)}</Text>
        </View>
      </View>

      {/* Description */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="document-text" size={24} color="#667eea" />
          <Text style={styles.sectionTitle}>विवरण</Text>
        </View>
        <Text style={styles.text}>{disease.descriptionNepali}</Text>
      </View>

      {/* Symptoms */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="medical" size={24} color="#f44336" />
          <Text style={styles.sectionTitle}>लक्षणहरू</Text>
        </View>
        {disease.symptomsNepali.map((symptom, index) => (
          <View key={index} style={styles.listItem}>
            <Text style={styles.bullet}>•</Text>
            <Text style={styles.listText}>{symptom}</Text>
          </View>
        ))}
      </View>

      {/* Causes */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="analytics" size={24} color="#ff9800" />
          <Text style={styles.sectionTitle}>कारण</Text>
        </View>
        <Text style={styles.text}>{disease.causesNepali}</Text>
      </View>

      {/* Treatment */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="fitness" size={24} color="#4caf50" />
          <Text style={styles.sectionTitle}>उपचार तथा व्यवस्थापन</Text>
        </View>
        {disease.treatmentNepali.map((treatment, index) => (
          <View key={index} style={styles.listItem}>
            <Text style={styles.numberBullet}>{index + 1}.</Text>
            <Text style={styles.listText}>{treatment}</Text>
          </View>
        ))}
      </View>

      {/* Prevention */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="shield-checkmark" size={24} color="#2196f3" />
          <Text style={styles.sectionTitle}>रोकथाम</Text>
        </View>
        {disease.preventionNepali.map((prevention, index) => (
          <View key={index} style={styles.listItem}>
            <Text style={styles.checkmark}>✓</Text>
            <Text style={styles.listText}>{prevention}</Text>
          </View>
        ))}
      </View>

      {/* When to Act */}
      <View style={[styles.section, styles.actionSection]}>
        <View style={styles.sectionHeader}>
          <Ionicons name="time" size={24} color="#9c27b0" />
          <Text style={styles.sectionTitle}>कहिले कारबाही गर्ने</Text>
        </View>
        <Text style={[styles.text, styles.actionText]}>{disease.whenToActNepali}</Text>
      </View>

      {/* Back Button */}
      <TouchableOpacity
        style={styles.backToHomeButton}
        onPress={() => navigation.navigate('Home')}
      >
        <Ionicons name="home" size={24} color="#fff" />
        <Text style={styles.backToHomeText}>मुख्य पृष्ठमा फर्कनुहोस्</Text>
      </TouchableOpacity>
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
    backgroundColor: '#667eea',
    padding: 20,
    paddingTop: 40,
    flexDirection: 'row',
    alignItems: 'center',
  },
  backButton: {
    padding: 8,
    marginRight: 12,
  },
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 18,
    color: '#fff',
    opacity: 0.9,
  },
  severityContainer: {
    padding: 16,
    alignItems: 'center',
  },
  severityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    gap: 8,
  },
  severityText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  section: {
    backgroundColor: '#fff',
    margin: 16,
    marginTop: 8,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    gap: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  text: {
    fontSize: 15,
    color: '#666',
    lineHeight: 24,
  },
  listItem: {
    flexDirection: 'row',
    marginBottom: 12,
    alignItems: 'flex-start',
  },
  bullet: {
    fontSize: 20,
    color: '#667eea',
    marginRight: 8,
    marginTop: -2,
  },
  numberBullet: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#667eea',
    marginRight: 8,
    minWidth: 24,
  },
  checkmark: {
    fontSize: 18,
    color: '#4caf50',
    marginRight: 8,
    fontWeight: 'bold',
  },
  listText: {
    flex: 1,
    fontSize: 15,
    color: '#666',
    lineHeight: 22,
  },
  actionSection: {
    backgroundColor: '#fff3e0',
    borderLeftWidth: 4,
    borderLeftColor: '#ff9800',
  },
  actionText: {
    fontWeight: '600',
    color: '#f57c00',
  },
  errorText: {
    fontSize: 16,
    color: '#f44336',
    textAlign: 'center',
    marginTop: 50,
  },
  backToHomeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#667eea',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    gap: 8,
  },
  backToHomeText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
  },
});
