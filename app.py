import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import requests
import gdown
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

MODEL_ID = "1aT-94D1rSPydiY4ulmhVH8NLVpAtv6EJ"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "Detection_model.keras"

# تحميل النموذج لو مش موجود
if not os.path.exists(MODEL_PATH):
    print("جاري تحميل النموذج من Google Drive باستخدام gdown...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=True)
    print("✅ تم تحميل النموذج بنجاح.")

# تحميل النموذج
model = load_model(MODEL_PATH)

# تحميل كاشف الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "لم يتم رفع أي صورة"}), 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    faces = face_cascade.detectMultiScale(opencv_image, scaleFactor=1.1, minNeighbors=3)

    if len(faces) == 0:
        return jsonify({"error": "لم يتم التعرف على الصورة، الرجاء إدخال صورة واضحة للطفل"}), 400

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    label = "مصاب" if confidence > 0.5 else "غير مصاب"

    return jsonify({"label": label, "confidence": round(confidence * 100, 2)})

if __name__ == "__main__":
    app.run(port=5000)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import requests
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import cv2
# import gdown
#
# MODEL_ID = "1aT-94D1rSPydiY4ulmhVH8NLVpAtv6EJ"
# MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
# MODEL_PATH = "Detection_model.keras"
#
# if not os.path.exists(MODEL_PATH):
#     print("جاري تحميل النموذج من Google Drive باستخدام gdown...")
#     gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
#     print("✅ تم تحميل النموذج بنجاح.")
#
#
# model = load_model(MODEL_PATH)
#
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#
# app = Flask(__name__)
#
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "لم يتم رفع أي صورة"}), 400
#
#     file = request.files["image"]
#     image = Image.open(file).convert("RGB")
#     opencv_image = np.array(image)
#     opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
#
#     faces = face_cascade.detectMultiScale(opencv_image, scaleFactor=1.1, minNeighbors=3)
#
#     if len(faces) == 0:
#         return jsonify({"error": "لم يتم التعرف على الصورة، الرجاء إدخال صورة واضحة للطفل"}), 400
#
#     image = image.resize((224, 224))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#
#     prediction = model.predict(img_array)
#     confidence = float(prediction[0][0])
#     label = "مصاب" if confidence > 0.5 else "غير مصاب"
#
#     return jsonify({"label": label, "confidence": round(confidence * 100, 2)})
#
#
# if __name__ == "__main__":
#     app.run(port=5000)
