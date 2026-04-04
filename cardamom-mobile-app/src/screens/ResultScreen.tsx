/**
 * Result Screen
 * Display prediction results identical to the web React frontend.
 */
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { RootStackParamList } from '../types';
import { getDiseaseInfoByName } from '../data/diseaseInfo';
import { ADVICE_MAP } from '../utils/AdviceMap';
import { STAGE_LABELS, STAGE_BADGE_COLORS } from '../utils/StageLabel';

type ResultScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Result'>;
type ResultScreenRouteProp = RouteProp<RootStackParamList, 'Result'>;

interface ResultScreenProps {
  navigation: ResultScreenNavigationProp;
  route: ResultScreenRouteProp;
}

export const ResultScreen: React.FC<ResultScreenProps> = ({ navigation, route }) => {
  const { imageUri, prediction } = route.params;
  const diseaseInfo = getDiseaseInfoByName(prediction.top_class);
  const advice = ADVICE_MAP[prediction.top_class];

  const hasSeverity =
    prediction.severity_stage !== null &&
    prediction.severity_stage !== undefined &&
    prediction.severity_percent !== null &&
    prediction.severity_percent !== undefined;

  const handleViewDetails = () => {
    if (diseaseInfo) {
      navigation.navigate('DiseaseInfo', { diseaseId: diseaseInfo.id });
    }
  };

  // Confidence bar color (mirrors web indigo→purple gradient – approximated with solid indigo)
  const confidenceBarColor = '#667eea';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>परिणाम / Results</Text>
        <View style={styles.placeholder} />
      </View>

      {/* Heatmap (shown when available, mirrors web which shows heatmap after analysis) */}
      {prediction.heatmap ? (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Grad-CAM Heatmap</Text>
          <Text style={styles.sectionSubtitle}>
            This visualization shows which regions of the leaf influenced the prediction
          </Text>
          <Image
            source={{ uri: `data:image/png;base64,${prediction.heatmap}` }}
            style={styles.heatmapImage}
            resizeMode="contain"
          />
        </View>
      ) : (
        /* Fallback: show original image when heatmap is not available */
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Preview</Text>
          <Image source={{ uri: imageUri }} style={styles.heatmapImage} resizeMode="contain" />
        </View>
      )}

      {/* Results card */}
      <View style={styles.resultsCard}>
        <Text style={styles.resultsHeading}>Results</Text>

        {/* API warning banner */}
        {prediction.warning && prediction.warning.length > 0 && (
          <View style={styles.warningBanner}>
            <Text style={styles.warningBannerText}>{prediction.warning.join(' ')}</Text>
          </View>
        )}

        <View style={styles.resultItems}>
          {/* Disease class */}
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Disease Class:</Text>
            <Text style={styles.resultValuePrimary}>
              {prediction.top_class}
              {advice ? ` — ${advice.nepaliName}` : ''}
            </Text>
          </View>

          {/* Confidence */}
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Confidence:</Text>
            <Text style={styles.resultValuePurple}>
              {prediction.top_probability_pct.toFixed(2)}%
            </Text>
          </View>

          {/* Confidence bar */}
          <View style={styles.confidenceBarTrack}>
            <View
              style={[
                styles.confidenceBarFill,
                {
                  width: `${prediction.top_probability_pct}%` as any,
                  backgroundColor: confidenceBarColor,
                },
              ]}
            />
          </View>

          {/* Uncertainty warning */}
          {prediction.is_uncertain && (
            <View style={styles.uncertaintyBox}>
              <Text style={styles.uncertaintyText}>
                ⚠️ Low confidence – prediction may be unreliable.
              </Text>
            </View>
          )}

          {/* Severity section (shown for non-Healthy classes only, mirrors web) */}
          {hasSeverity && prediction.top_class !== 'Healthy' && (
            <View style={styles.severitySection}>
              <Text style={styles.severityHeading}>Severity Estimation</Text>

              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Stage:</Text>
                <View
                  style={[
                    styles.stageBadge,
                    {
                      backgroundColor:
                        STAGE_BADGE_COLORS[prediction.severity_stage!]?.bg ?? '#f1f5f9',
                    },
                  ]}
                >
                  <Text
                    style={[
                      styles.stageBadgeText,
                      {
                        color:
                          STAGE_BADGE_COLORS[prediction.severity_stage!]?.text ?? '#334155',
                      },
                    ]}
                  >
                    {STAGE_LABELS[prediction.severity_stage!] ??
                      `Stage ${prediction.severity_stage}`}
                  </Text>
                </View>
              </View>

              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Area Affected:</Text>
                <Text style={styles.areaAffectedValue}>
                  {prediction.severity_percent!.toFixed(1)}%
                </Text>
              </View>

              {/* Severity bar: green → amber → red */}
              <View style={styles.severityBarTrack}>
                <View
                  style={[
                    styles.severityBarFill,
                    { width: `${prediction.severity_percent}%` as any },
                  ]}
                />
              </View>

              {prediction.severity_method === 'heuristic' && (
                <View style={styles.heuristicNote}>
                  <Text style={styles.heuristicText}>
                    ℹ️ <Text style={{ fontWeight: 'bold' }}>Estimate only.</Text> Severity was
                    approximated from the Grad-CAM heatmap (heuristic method) and does not reflect
                    true lesion area. For accurate quantification, use mask-based labelling.
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* Recommendations (सुझाव) – mirrors web frontend exactly */}
          {advice && (
            <View style={styles.adviceSection}>
              <Text style={styles.adviceHeading}>सुझाव</Text>

              <View style={styles.adviceBlock}>
                <Text style={styles.adviceSubHeading}>रोकथाम (Prevention)</Text>
                {advice.prevention.map((tip, idx) => (
                  <View key={`prev-${idx}`} style={styles.adviceItem}>
                    <Text style={styles.bullet}>•</Text>
                    <Text style={styles.adviceText}>{tip}</Text>
                  </View>
                ))}
              </View>

              <View style={styles.adviceBlock}>
                <Text style={styles.adviceSubHeading}>
                  उपचार/व्यवस्थापन (Cure / Management)
                </Text>
                {advice.cure.map((tip, idx) => (
                  <View key={`cure-${idx}`} style={styles.adviceItem}>
                    <Text style={styles.bullet}>•</Text>
                    <Text style={styles.adviceText}>{tip}</Text>
                  </View>
                ))}
              </View>

              <View style={styles.adviceDisclaimer}>
                <Text style={styles.adviceDisclaimerText}>
                  ℹ️ <Text style={{ fontWeight: 'bold' }}>सूचना: </Text>
                  यी सुझावहरू सामान्य जानकारीका लागि हुन्। स्थानीय कृषि
                  प्राविधिक/कृषि कार्यालयको सल्लाह अनुसार मात्र औषधि/छर्काइ
                  प्रयोग गर्नुहोस्।
                </Text>
              </View>
            </View>
          )}
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actionsContainer}>
        {diseaseInfo && (
          <TouchableOpacity
            style={[styles.actionButton, styles.primaryButton]}
            onPress={handleViewDetails}
          >
            <Ionicons name="information-circle" size={24} color="#fff" />
            <Text style={styles.actionButtonText}>विस्तृत जानकारी</Text>
          </TouchableOpacity>
        )}

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
    fontSize: 18,
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
    fontSize: 18,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 4,
    textAlign: 'center',
  },
  sectionSubtitle: {
    fontSize: 13,
    color: '#64748b',
    fontStyle: 'italic',
    textAlign: 'center',
    marginBottom: 12,
  },
  heatmapImage: {
    width: '100%',
    height: 300,
    borderRadius: 12,
    backgroundColor: '#e2e8f0',
  },
  resultsCard: {
    margin: 16,
    backgroundColor: '#f8fafc',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  resultsHeading: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1e293b',
    textAlign: 'center',
    marginBottom: 16,
  },
  warningBanner: {
    backgroundColor: '#fffbeb',
    borderWidth: 2,
    borderColor: '#fde68a',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  warningBannerText: {
    color: '#92400e',
    fontWeight: '600',
    textAlign: 'center',
    fontSize: 14,
  },
  resultItems: {
    gap: 8,
  },
  resultRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    flexWrap: 'wrap',
    gap: 4,
  },
  resultLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },
  resultValuePrimary: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#4f46e5',
    flexShrink: 1,
    textAlign: 'right',
  },
  resultValuePurple: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#7c3aed',
  },
  confidenceBarTrack: {
    height: 10,
    backgroundColor: '#e2e8f0',
    borderRadius: 999,
    overflow: 'hidden',
  },
  confidenceBarFill: {
    height: '100%',
    borderRadius: 999,
  },
  uncertaintyBox: {
    backgroundColor: '#fef3c7',
    borderWidth: 1,
    borderColor: '#fbbf24',
    borderRadius: 8,
    padding: 12,
  },
  uncertaintyText: {
    color: '#78350f',
    fontSize: 13,
  },
  // ── Severity ──────────────────────────────────────────────────────────────
  severitySection: {
    borderTopWidth: 2,
    borderTopColor: '#e2e8f0',
    borderStyle: 'dashed',
    paddingTop: 16,
    marginTop: 8,
    gap: 8,
  },
  severityHeading: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 4,
  },
  stageBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 999,
  },
  stageBadgeText: {
    fontSize: 13,
    fontWeight: '600',
  },
  areaAffectedValue: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#dc2626',
  },
  severityBarTrack: {
    height: 8,
    backgroundColor: '#e2e8f0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  severityBarFill: {
    height: '100%',
    borderRadius: 4,
    // Approximation of web's emerald→amber→red gradient using a fixed amber-red color
    backgroundColor: '#f59e0b',
  },
  heuristicNote: {
    backgroundColor: '#e0f2fe',
    borderWidth: 1,
    borderColor: '#bae6fd',
    borderRadius: 8,
    padding: 12,
  },
  heuristicText: {
    fontSize: 13,
    color: '#0c4a6e',
    lineHeight: 20,
  },
  // ── Recommendations ───────────────────────────────────────────────────────
  adviceSection: {
    marginTop: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e7ff',
    backgroundColor: 'rgba(238, 242, 255, 0.5)',
    padding: 16,
    gap: 12,
  },
  adviceHeading: {
    fontSize: 17,
    fontWeight: '600',
    color: '#1e293b',
  },
  adviceBlock: {
    gap: 6,
  },
  adviceSubHeading: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 4,
  },
  adviceItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 6,
  },
  bullet: {
    fontSize: 16,
    color: '#4f46e5',
    lineHeight: 22,
  },
  adviceText: {
    flex: 1,
    fontSize: 14,
    color: '#334155',
    lineHeight: 22,
  },
  adviceDisclaimer: {
    backgroundColor: '#fef2f2',
    borderWidth: 1,
    borderColor: '#fecaca',
    borderRadius: 8,
    padding: 12,
  },
  adviceDisclaimerText: {
    fontSize: 13,
    color: '#7f1d1d',
    lineHeight: 20,
  },
  // ── Action buttons ────────────────────────────────────────────────────────
  actionsContainer: {
    paddingHorizontal: 16,
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
