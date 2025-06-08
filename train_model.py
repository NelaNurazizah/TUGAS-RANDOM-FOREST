import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_preprocess_images(folder_path):
    images = []
    labels = []

    for label, class_name in enumerate(sorted(os.listdir(folder_path))):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Baca sebagai grayscale
            img = cv2.resize(img, (64, 64))                   # Resize
            img = img.flatten() / 255.0                       # Flatten + normalisasi
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# 1. Muat dataset
DATASET_PATH = 'dataset/train'
X, y = load_and_preprocess_images(DATASET_PATH)

# 2. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi model
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# 5. Simpan model
joblib.dump(model, 'models/rf_model.pkl')

# (Opsional) Simpan nama kelas untuk referensi saat prediksi
class_names = sorted(os.listdir(DATASET_PATH))
np.save('models/class_names.npy', class_names)

print("✅ Model berhasil dilatih dan disimpan!")