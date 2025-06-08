from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
import os
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# URL Google Drive (direct download)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1u1jtcrBEHqLtu3x3c7HXTn9G9UM9mFrY"
CLASS_NAMES_URL = "https://drive.google.com/uc?export=download&id=1-9PQ70gQtz2AeevqAK5XUMheUMtxM1hz"

# Folder untuk model
os.makedirs('models', exist_ok=True)

# Fungsi download jika belum ada
def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"🔽 Mengunduh {destination} ...")
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"✅ Selesai mengunduh {destination}")

# Download model & class names jika belum ada
download_file(MODEL_URL, 'models/rf_model.pkl')
download_file(CLASS_NAMES_URL, 'models/class_names.npy')

# Muat model dan nama kelas
model = joblib.load('models/rf_model.pkl')
class_names = np.load('models/class_names.npy')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            image_url = '/' + filepath

            # Proses gambar
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            img = img.flatten() / 255.0
            img = img.reshape(1, -1)

            predicted_index = model.predict(img)[0]
            prediction = class_names[predicted_index]

    return render_template('index.html', prediction=prediction, image_path=image_url)

if __name__ == '__main__':
    app.run(debug=True)
