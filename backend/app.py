from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
import tensorflow as tf
import traceback

app = Flask(__name__, static_folder="../frontend/build", static_url_path="")
CORS(app)

# -------------------------------
# 1️⃣ Load TensorFlow SavedModel
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "my_model_tf")

print("🔍 Checking if model exists at:", MODEL_DIR)

if not os.path.exists(MODEL_DIR):
    print("❌ Model folder does not exist! Check path and commit model to repo.")
    model_loaded = False
else:
    files_in_model = os.listdir(MODEL_DIR)
    print("📂 Files in model folder:", files_in_model)
    try:
        saved = tf.saved_model.load(MODEL_DIR)
        infer = saved.signatures.get("serving_default") or list(saved.signatures.values())[0]
        print("✅ Loaded TensorFlow SavedModel successfully.")
        model_loaded = True
    except Exception as e:
        print("❌ Failed to load model:", e)
        model_loaded = False

# -------------------------------
# 2️⃣ Haar cascades for face/eyes
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_prob_open(processed):
    t = tf.convert_to_tensor(processed, dtype=tf.float32)
    try:
        pred = infer(t)
    except Exception:
        pred = infer(**{list(infer.structured_input_signature[1].keys())[0]: t})
    if isinstance(pred, dict):
        first_val = next(iter(pred.values()))
        arr = first_val.numpy() if tf.is_tensor(first_val) else np.array(first_val)
    elif tf.is_tensor(pred):
        arr = pred.numpy()
    elif isinstance(pred, (list, tuple)):
        arr = pred[0].numpy() if tf.is_tensor(pred[0]) else np.array(pred[0])
    else:
        arr = np.array(pred)
    arr = np.array(arr).reshape(-1)
    return float(arr[0]) if arr.size > 0 else 0.0

# -------------------------------
# 3️⃣ Serve React frontend
# -------------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

# -------------------------------
# 4️⃣ Predict endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        payload = request.get_json(force=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "No image provided"}), 400

        data = payload["image"]
        if "," in data:
            _, b64 = data.split(",", 1)
        else:
            b64 = data

        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            img_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
            return jsonify({"status": "Not Detected", "probability_open": 0.0, "frame": img_str})

        x, y, w, h = faces[0]
        roi_color = frame[y:y+h, x:x+w]
        processed = preprocess_frame(roi_color)
        prob_open = predict_prob_open(processed)
        status = "Open Eyes" if prob_open >= 0.65 else "Closed Eyes"

        _, buffer = cv2.imencode(".jpg", frame)
        img_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        return jsonify({"status": status, "probability_open": prob_open, "frame": img_str})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------------------
# 5️⃣ Run server
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
