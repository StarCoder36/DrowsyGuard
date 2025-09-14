from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import traceback

app = Flask(__name__)

# ✅ Allow requests from your current frontend URL
CORS(app, origins=["https://drowsy-guard-opal.vercel.app"], supports_credentials=True)

# ---------------- Your existing model & routes code ----------------
# (keep all your model loading, predict route, etc. exactly as before)


# Load model (try TFSMLayer first, fallback to saved_model.load)
model = None
infer = None
MODEL_DIR = "my_model_tf"

try:
    # TFSMLayer is the Keras 3 friendly wrapper for SavedModel
    model = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
    print("Loaded model with TFSMLayer.")
except Exception as e:
    print("TFSMLayer load failed, trying tf.saved_model.load() fallback:", e)
    try:
        saved = tf.saved_model.load(MODEL_DIR)
        # try to get the serving_default signature
        infer = saved.signatures.get("serving_default", None)
        if infer is None:
            # pick any callable attribute if available
            infer = list(saved.signatures.values())[0] if saved.signatures else None
        print("Loaded SavedModel; using signature for inference.")
    except Exception as e2:
        print("Failed to load model:", e2)
        raise

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def preprocess_frame(frame):
    """Resize and normalize frame for model input (returns shape (1,224,224,3))."""
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_prob_open(processed):
    """
    Call the model (TFSMLayer or signature) and return scalar prob_open (float).
    Handles tensor, dict, list/tuple return types robustly.
    """
    # Call model
    if model is not None:
        pred = model(processed, training=False)
    elif infer is not None:
        # signature expects a tf.Tensor input keyed by input name or positional; try both
        t = tf.convert_to_tensor(processed, dtype=tf.float32)
        try:
            # many signatures accept the tensor as the first arg (positional)
            out = infer(t)
        except Exception:
            # or as a dict with input name
            # try to find input key from signature
            out = infer(**{list(infer.structured_input_signature[1].keys())[0]: t})
        pred = out
    else:
        raise RuntimeError("No model available for prediction")

    # Normalize pred -> numpy array
    try:
        if isinstance(pred, dict):
            first_val = next(iter(pred.values()))
            arr = first_val.numpy() if tf.is_tensor(first_val) else np.array(first_val)
        elif tf.is_tensor(pred):
            arr = pred.numpy()
        elif isinstance(pred, (list, tuple)):
            first = pred[0]
            arr = first.numpy() if tf.is_tensor(first) else np.array(first)
        else:
            arr = np.array(pred)
    except Exception:
        # Final fallback: convert to numpy via np.array
        arr = np.array(pred)

    # Expect arr to contain at least one scalar probability at index [0][0] or [0]
    arr = np.array(arr).reshape(-1)
    if arr.size == 0:
        return 0.0
    prob_open = float(arr[0])
    return prob_open

@app.route("/", methods=["GET"])
def home():
    return (
        "✅ Driver Drowsiness Detection Backend Running\n"
        "POST an image (JSON) to /predict with: {'image': 'data:image/jpeg;base64,...'}"
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "Request must be JSON with key 'image' (base64 data URL)"}), 400

        data = payload["image"]
        # Accept either "data:image/jpeg;base64,...." or raw base64
        if "," in data:
            _, b64 = data.split(",", 1)
        else:
            b64 = data

        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Detect face & eyes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            # No face — return original frame and Not Detected
            _, buffer = cv2.imencode(".jpg", frame)
            img_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
            return jsonify({"status": "Not Detected", "probability_open": 0.0, "frame": img_str})

        # Use first face
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Predict probability on face ROI
        processed = preprocess_frame(roi_color)
        prob_open = predict_prob_open(processed)
        status = "Open Eyes" if prob_open >= 0.65 else "Closed Eyes"

        # Encode annotated frame back to base64
        _, buffer = cv2.imencode(".jpg", frame)
        img_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        return jsonify({"status": status, "probability_open": prob_open, "frame": img_str})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # bind to 0.0.0.0 so other devices (or React on same machine) can reach it,
    # and set debug True during development.
    app.run(host="0.0.0.0", port=5000, debug=True)
