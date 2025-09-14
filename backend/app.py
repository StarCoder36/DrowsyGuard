from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import traceback
import threading
import time
import os

app = Flask(__name__)

# ✅ CORS configuration
CORS(app, 
     origins=["https://drowsy-guard-opal.vercel.app", "http://localhost:3000", "*"], 
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global variables for model and cascades
model = None
infer = None
face_cascade = None
eye_cascade = None
model_loaded = False

def load_model_and_cascades():
    """Load model and cascades in a separate thread to avoid blocking startup"""
    global model, infer, face_cascade, eye_cascade, model_loaded
    
    MODEL_DIR = "my_model_tf"
    
    try:
        print("🔄 Loading model...")
        
        # Try to load model
        try:
            model = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
            print("✅ Loaded model with TFSMLayer.")
        except Exception as e:
            print(f"TFSMLayer failed: {e}")
            try:
                saved = tf.saved_model.load(MODEL_DIR)
                infer = saved.signatures.get("serving_default", None)
                if infer is None:
                    infer = list(saved.signatures.values())[0] if saved.signatures else None
                print("✅ Loaded SavedModel with signature.")
            except Exception as e2:
                print(f"❌ Model loading failed: {e2}")
                return
        
        # Load Haar cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        
        if face_cascade.empty() or eye_cascade.empty():
            print("❌ Failed to load Haar cascades")
            return
            
        model_loaded = True
        print("✅ All components loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error in loading: {e}")
        traceback.print_exc()

def preprocess_frame(frame):
    """Optimized preprocessing"""
    try:
        # Resize more efficiently
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def predict_prob_open(processed):
    """Optimized prediction with timeout"""
    if processed is None:
        return 0.0
        
    try:
        # Set a timeout for prediction
        if model is not None:
            pred = model(processed, training=False)
        elif infer is not None:
            t = tf.convert_to_tensor(processed, dtype=tf.float32)
            try:
                pred = infer(t)
            except Exception:
                pred = infer(**{list(infer.structured_input_signature[1].keys())[0]: t})
        else:
            return 0.0

        # Process prediction result
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

        arr = np.array(arr).reshape(-1)
        return float(arr[0]) if arr.size > 0 else 0.0
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0

# Keep-alive mechanism to prevent cold starts
def keep_alive():
    """Keep the server warm"""
    while True:
        time.sleep(600)  # Every 10 minutes
        try:
            print("🔄 Keep-alive ping")
        except:
            pass

@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    return jsonify({
        "message": "✅ Driver Drowsiness Detection Backend",
        "status": "running",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """Quick health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    # Quick check if model is loaded
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded yet, please wait and try again",
            "status": "loading"
        }), 503
    
    start_time = time.time()
    
    try:
        # Parse request with timeout
        payload = request.get_json(force=True, silent=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "Missing image data"}), 400

        data = payload["image"]
        if "," in data:
            _, b64 = data.split(",", 1)
        else:
            b64 = data

        # Decode image
        try:
            img_bytes = base64.b64decode(b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Image decode failed: {str(e)}"}), 400

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Quick face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))

        if len(faces) == 0:
            # No face detected - quick response
            processing_time = time.time() - start_time
            return jsonify({
                "status": "Not Detected",
                "probability_open": 0.0,
                "processing_time": round(processing_time, 3)
            }), 200

        # Process first face only for speed
        x, y, w, h = faces[0]
        roi_color = frame[y:y+h, x:x+w]

        # Quick eye detection (optional - skip if too slow)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 2, minSize=(20, 20))

        # Predict on face ROI
        processed = preprocess_frame(roi_color)
        prob_open = predict_prob_open(processed)
        
        status = "Open Eyes" if prob_open >= 0.65 else "Closed Eyes"
        processing_time = time.time() - start_time

        return jsonify({
            "status": status,
            "probability_open": prob_open,
            "face_detected": True,
            "eyes_detected": len(eyes),
            "processing_time": round(processing_time, 3)
        }), 200

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        
        return jsonify({
            "error": f"Processing failed: {str(e)}",
            "processing_time": round(processing_time, 3)
        }), 500

if __name__ == "__main__":
    # Start model loading in background
    loading_thread = threading.Thread(target=load_model_and_cascades, daemon=True)
    loading_thread.start()
    
    # Start keep-alive thread
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    
    print(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)