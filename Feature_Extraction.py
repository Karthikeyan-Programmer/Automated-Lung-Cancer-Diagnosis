import cv2
import numpy as np
import matplotlib.pyplot as plt
def MIR_GLCM():
    image_path = 'filtered_image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    def calculate_glcm(image, distance, angle):
        shifted = np.roll(image, shift=(distance, distance), axis=(0, 1))
        co_occurrence = np.histogram2d(image.flatten(), shifted.flatten(), bins=(256, 256))[0]
        return co_occurrence
    def extract_mir_glcm_features(image):
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        features = {}
        for distance in distances:
            for angle in angles:
                glcm = calculate_glcm(image, distance, angle)
                contrast = np.sum((np.arange(glcm.shape[0]) - np.arange(glcm.shape[1]))**2 * glcm)
                dissimilarity = np.sum(np.abs(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1])) * glcm)
                homogeneity = np.sum(glcm / (1 + np.abs(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1]))))
                energy = np.sum(glcm**2)
                correlation = np.sum((np.arange(glcm.shape[0]) * np.arange(glcm.shape[1]) * glcm) - np.sum(glcm)) / (np.sqrt(np.sum(np.arange(glcm.shape[0])**2 * np.sum(glcm, axis=1))) * np.sqrt(np.sum(np.arange(glcm.shape[1])**2 * np.sum(glcm, axis=0))))
                features[f'contrast_{distance}_{angle}'] = contrast
                features[f'dissimilarity_{distance}_{angle}'] = dissimilarity
                features[f'homogeneity_{distance}_{angle}'] = homogeneity
                features[f'energy_{distance}_{angle}'] = energy
                features[f'correlation_{distance}_{angle}'] = correlation
        return features
    mir_glcm_features = extract_mir_glcm_features(image)
    print("Extracted MIR-GLCM Features:")
    for key, value in mir_glcm_features.items():
        print(f'{key}: {value}')
    fig, axs = plt.subplots(len(mir_glcm_features) // 2, 2, figsize=(12, 2 * len(mir_glcm_features) // 2))
    fig.suptitle('MIR-GLCM Features', y=1.02)
    for i, (key, value) in enumerate(mir_glcm_features.items()):
        ax = axs[i // 2, i % 2]
        ax.bar(['Feature'], [value], color='blue')
        ax.set_title(key)
        ax.set_ylabel('Value')
    plt.tight_layout()
    plt.show()
