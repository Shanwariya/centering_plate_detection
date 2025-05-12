# plate_detection.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os

def detect_plates(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image.")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    horizontal_sum = np.sum(edges, axis=1)
    smoothed = gaussian_filter1d(horizontal_sum, sigma=8)

    peaks, _ = find_peaks(smoothed, distance=25, prominence=40)

    for y in peaks:
        cv2.line(image, (0, y), (image.shape[1], y), (0, 255, 0), 2)

    estimated_plates = max(1, len(peaks) - 1)
    return image, estimated_plates

def main():
    image_path = input("Enter the path of the image file (e.g., test.jpg): ")
    
    if not os.path.exists(image_path):
        print("❌ File does not exist.")
        return

    output_img, plate_count = detect_plates(image_path)
    if output_img is not None:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Estimated Plates: {plate_count}")
        plt.axis('off')
        plt.show()

        output_filename = "processed_output.jpg"
        cv2.imwrite(output_filename, output_img)
        print(f"✅ Output saved as {output_filename}")

if __name__ == "__main__":
    main()
