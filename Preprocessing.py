import cv2
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
def Butterworth_Smooth_filter():
    def butterworth_smooth_filter(image, cutoff_frequency, order=3):
        rows, cols = image.shape
        center = (rows // 2, cols // 2)
        butter_filter = np.ones((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                butter_filter[i, j] = 1 / (1 + (distance / cutoff_frequency) ** (2 * order))
        return butter_filter
    def apply_filter(image, butter_filter):
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)
        fft_filtered = fft_shifted * butter_filter
        result_image = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real
        result_image = np.uint8(result_image)
        return result_image
    image_path = "IpImg.png"
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cutoff_frequency = 30
    order = 3
    butter_filter = butterworth_smooth_filter(original_image, cutoff_frequency, order)
    filtered_image = apply_filter(original_image, butter_filter)
    cv2.imwrite("filtered_image.png", filtered_image)
    plt.figure(figsize=(9, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    plt.show()
