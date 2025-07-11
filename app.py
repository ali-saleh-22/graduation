import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import gdown
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

# 🟡 قراءة MODEL_ID من Environment Variable
MODEL_ID = os.getenv("MODEL_ID")
if not MODEL_ID:
    raise ValueError("❌ لم يتم تعيين المتغير البيئي MODEL_ID")

MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "Detection_model.keras"

# تحميل النموذج لو مش موجود
if not os.path.exists(MODEL_PATH):
    print("جاري تحميل النموذج من Google Drive باستخدام gdown...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=True)
    print("✅ تم تحميل النموذج بنجاح.")

# Lazy load للموديل
model = None

# تحميل كاشف الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# إنشاء التطبيق
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        print("⏳ تحميل الموديل عند أول استخدام...")
        model = load_model(MODEL_PATH)
        print("✅ تم تحميل الموديل.")

    if "image" not in request.files:
        return jsonify({"error": "لم يتم رفع أي صورة"}), 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # كشف الوجه
    faces = face_cascade.detectMultiScale(opencv_image, scaleFactor=1.1, minNeighbors=3)

    if len(faces) == 0:
        return jsonify({"error": "لم يتم التعرف على الوجه، الرجاء رفع صورة واضحة للطفل"}), 400

    # تجهيز الصورة
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    label = "مصاب" if confidence > 0.5 else "غير مصاب"

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(port=5000)
