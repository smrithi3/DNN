#Demonstrate enchancing of segementing low contrast 2 D images 

import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_and_segment(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found or could not be loaded.")
        return
   
    # Step 1: Enhance the image using Histogram Equalization
    enhanced_image = cv2.equalizeHist(image)

    # Step 2: Segment the enhanced image using Otsu's Thresholding
    # Otsu's thresholding automatically determines the optimal threshold value
    _, segmented_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the original, enhanced, and segmented images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Enhanced Image (Histogram Equalization)")
    plt.imshow(enhanced_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Segmented Image (Otsu's Thresholding)")
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Path to the low-contrast image
image_path = 'images.jpg'  # Replace with your image path
enhance_and_segment(image_path)