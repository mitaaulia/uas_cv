from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = tf.keras.models.load_model('model/model.h5')

classes = ['Curly', 'Dreadlocks', 'Kinky', 'Straight', 'Wavy']

recommendations = {
    'Curly': 'Gunakan produk bebas sulfat dan rutin lakukan deep conditioning.',
    'Dreadlocks': 'Cuci secara teratur dan gunakan minyak alami untuk menjaga kelembapan.',
    'Kinky': 'Lakukan moisturizing setiap hari dan hindari panas berlebih.',
    'Straight': 'Gunakan sampo ringan dan potong ujung rambut secara berkala.',
    'Wavy': 'Gunakan leave-in conditioner dan hindari menyisir saat basah.'
}

def predict_hair_type(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = float(prediction[predicted_index])
    return predicted_class, confidence

def get_hair_care_recommendation(hair_type):
    return recommendations.get(hair_type, "Tidak ada rekomendasi tersedia.")

@app.route('/')
def root():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/scan-page')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'file' not in request.files:
        return redirect('/scan-page')
    file = request.files['file']
    if file.filename == '':
        return redirect('/scan-page')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)

        label, confidence = predict_hair_type(filepath)
        care_tip = get_hair_care_recommendation(label)
        return render_template('result.html', label=label, confidence=round(confidence * 100, 2),
                               filename=filename, care_tip=care_tip)
    return redirect('/scan-page')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
