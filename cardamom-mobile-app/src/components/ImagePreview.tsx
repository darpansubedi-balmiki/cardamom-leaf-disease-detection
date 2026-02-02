/**
 * Image Preview Component
 */
import React from 'react';
import { View, Image, StyleSheet, Dimensions } from 'react-native';

interface ImagePreviewProps {
  imageUri: string;
  style?: any;
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export const ImagePreview: React.FC<ImagePreviewProps> = ({ imageUri, style }) => {
  return (
    <View style={[styles.container, style]}>
      <Image
        source={{ uri: imageUri }}
        style={styles.image}
        resizeMode="contain"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    aspectRatio: 1,
    backgroundColor: '#f0f0f0',
    borderRadius: 12,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
});
