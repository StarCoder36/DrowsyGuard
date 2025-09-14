# 🚗 DrowsyGuard: Driver Drowsiness Detection System

DrowsyGuard is a **real-time driver drowsiness detection system** built as a web application. It uses a **MobileNet-based deep learning model** to analyze a live video feed from your webcam and detect whether the driver’s eyes are open or closed. The system aims to **prevent accidents** by alerting drivers when signs of drowsiness are detected.  

---

## 🔹 Features

- **Real-time Detection:** Instantly detects the state of a driver’s eyes.  
- **Efficient Model:** Uses MobileNet CNN, a lightweight yet powerful model for fast and accurate predictions.  
- **Computer Vision:** Employs OpenCV with Haar Cascades for reliable face and eye detection.  
- **Seamless Integration:** Base64 encoding ensures smooth image transfer between frontend and backend.  
- **Modern Web Interface:** Responsive and interactive UI built with React.  
- **Robust Backend:** Flask API serves the deep learning model and handles predictions.  
- **Flexible Deployment:** Run locally or deploy to cloud platforms like Heroku, Render, or AWS.  

---

## 🛠 Tech Stack

- **Frontend:** React.js  
- **Backend:** Flask, Flask-CORS  
- **Deep Learning:** TensorFlow, MobileNet  
- **Computer Vision:** OpenCV (Haar Cascades)  
- **Data Handling:** NumPy  
- **Deployment:** Local, Heroku, Render, AWS  

---

## 📂 Folder Structure

```drowsiness-web/
├─ backend/
│ ├─ app.py # Flask API
│ ├─ my_model_tf/ # TensorFlow SavedModel (MobileNet)
│ └─ requirements.txt # Python dependencies
├─ frontend/
│ ├─ build/ # React build files (production)
│ └─ src/ # React source code
└─ .gitignore # Files/folders to ignore in Git
```
---

## ⚡ Installation

### Prerequisites

- Python 3.8+  
- Node.js & npm  

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

## 2️⃣ Backend Setup

Navigate to the backend folder and install the required Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

## 3️⃣ Frontend Setup

Navigate to the frontend folder, install dependencies, and build the project:

```bash
cd frontend
npm install
npm run build
```
## 4️⃣ Run the Application

From the backend folder, run the Flask server:

```bash
cd backend
python app.py
```
The application will be accessible at: http://127.0.0.1:5000

The frontend is automatically served via Flask from ../frontend/build.


