/**
 * Home Screen
 * Main screen with camera and gallery options
 */
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
  ScrollView,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { RootStackParamList } from '../types';
import { predictDisease } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

interface HomeScreenProps {
  navigation: HomeScreenNavigationProp;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    // Request permissions on mount
    (async () => {
      if (Platform.OS !== 'web') {
        const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
        const { status: galleryStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        
        if (cameraStatus !== 'granted' || galleryStatus !== 'granted') {
          Alert.alert(
            'अनुमति आवश्यक छ',
            'कृपया क्यामेरा र ग्यालेरी पहुँचको लागि अनुमति दिनुहोस्।'
          );
        }
      }
    })();
  }, []);

  const handleImageSelected = async (imageUri: string) => {
    setIsProcessing(true);
    
    try {
      // Call API to predict disease
      const prediction = await predictDisease(imageUri);
      
      // Navigate to result screen
      navigation.navigate('Result', {
        imageUri,
        prediction,
      });
    } catch (error: any) {
      Alert.alert('त्रुटि', error.message || 'भविष्यवाणी असफल भयो। कृपया पुन: प्रयास गर्नुहोस्।');
    } finally {
      setIsProcessing(false);
    }
  };

  const openCamera = async () => {
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        await handleImageSelected(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('त्रुटि', 'क्यामेरा खोल्न सकिएन।');
    }
  };

  const openGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        await handleImageSelected(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('त्रुटि', 'ग्यालेरी खोल्न सकिएन।');
    }
  };

  if (isProcessing) {
    return (
      <LoadingSpinner
        message="तस्बिर विश्लेषण गर्दै..."
      />
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.header}>
        <Text style={styles.title}>अलैंची रोग पहिचान</Text>
        <Text style={styles.subtitle}>Cardamom Disease Detection</Text>
        <Text style={styles.description}>
          अलैंची बिरुवाको पातको तस्बिर खिच्नुहोस् वा ग्यालेरीबाट छान्नुहोस्
        </Text>
      </View>

      <View style={styles.buttonsContainer}>
        <TouchableOpacity
          style={[styles.button, styles.cameraButton]}
          onPress={openCamera}
          activeOpacity={0.8}
        >
          <Ionicons name="camera" size={48} color="#fff" />
          <Text style={styles.buttonText}>क्यामेरा खोल्नुहोस्</Text>
          <Text style={styles.buttonSubtext}>Open Camera</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.galleryButton]}
          onPress={openGallery}
          activeOpacity={0.8}
        >
          <Ionicons name="images" size={48} color="#fff" />
          <Text style={styles.buttonText}>ग्यालेरीबाट छान्नुहोस्</Text>
          <Text style={styles.buttonSubtext}>Choose from Gallery</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.infoSection}>
        <Text style={styles.infoTitle}>समर्थित रोगहरू:</Text>
        <View style={styles.diseaseList}>
          <Text style={styles.diseaseItem}>• कोलेटोट्रिकम ब्लाइट (Colletotrichum Blight)</Text>
          <Text style={styles.diseaseItem}>• फाइलोस्टिक्टा पात दाग (Phyllosticta Leaf Spot)</Text>
          <Text style={styles.diseaseItem}>• स्वस्थ (Healthy)</Text>
        </View>
      </View>

      <View style={styles.tipsSection}>
        <Text style={styles.tipsTitle}>राम्रो नतिजाको लागि:</Text>
        <Text style={styles.tip}>• स्पष्ट र फोकसमा रहेको तस्बिर खिच्नुहोस्</Text>
        <Text style={styles.tip}>• राम्रो प्रकाशमा तस्बिर खिच्नुहोस्</Text>
        <Text style={styles.tip}>• पातको नजिकबाट तस्बिर खिच्नुहोस्</Text>
        <Text style={styles.tip}>• पूरै पात फ्रेममा देखिने गरी खिच्नुहोस्</Text>
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
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
    marginTop: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    color: '#667eea',
    fontWeight: '600',
    marginBottom: 12,
  },
  description: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
  },
  buttonsContainer: {
    marginBottom: 32,
  },
  button: {
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  cameraButton: {
    backgroundColor: '#667eea',
  },
  galleryButton: {
    backgroundColor: '#764ba2',
  },
  buttonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 12,
  },
  buttonSubtext: {
    fontSize: 13,
    color: '#fff',
    opacity: 0.9,
    marginTop: 4,
  },
  infoSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  diseaseList: {
    marginLeft: 8,
  },
  diseaseItem: {
    fontSize: 14,
    color: '#666',
    marginBottom: 6,
    lineHeight: 20,
  },
  tipsSection: {
    backgroundColor: '#e8f5e9',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  tipsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 12,
  },
  tip: {
    fontSize: 13,
    color: '#2e7d32',
    marginBottom: 6,
    lineHeight: 18,
  },
});
