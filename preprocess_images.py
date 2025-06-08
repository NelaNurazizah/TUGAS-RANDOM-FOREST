import os
import cv2
import numpy as np

def load_and_preprocess_images(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, label)
        if not os.path.isdir(class_dir):
            continue
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Baca sebagai grayscale
            img = cv2.resize(img, (64, 64))  # Resize ke ukuran seragam
            img = img.flatten() / 255.0  # Flatten dan normalisasi
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Contoh penggunaan
X, y = load_and_preprocess_images('dataset/train')
print("Jumlah data:", len(X))