import React, { useEffect, useRef, useState, useCallback } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const alarmRef = useRef(null);
  const lastSentRef = useRef(0);
  const eyeClosedStartRef = useRef(null);
  const consecutiveFailuresRef = useRef(0);
  const isProcessingRef = useRef(false);

  const [status, setStatus] = useState("Initializing...");
  const [probability, setProbability] = useState(0);
  const [faceDetected, setFaceDetected] = useState(true);
  const [drowsyCount, setDrowsyCount] = useState(0);
  const [lastDrowsyTime, setLastDrowsyTime] = useState(null);
  const [probHistory, setProbHistory] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [processingTime, setProcessingTime] = useState(0);
  const [backendStatus, setBackendStatus] = useState("checking");

  const CLOSED_EYE_THRESHOLD = 0.65;
  const ALARM_DURATION = 2000;
  const BACKEND_URL = "https://drowsyguard-backend.onrender.com";
  const MAX_CONSECUTIVE_FAILURES = 3;

  // Adaptive timing based on performance
  const [requestInterval, setRequestInterval] = useState(1000); // Start with 1 second

  // Create axios instance with retry logic
  const apiClient = axios.create({
    baseURL: BACKEND_URL,
    timeout: 15000, // Increased timeout
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }
  });

  // Add retry interceptor
  apiClient.interceptors.response.use(
    (response) => response,
    async (error) => {
      const { config } = error;
      
      if (!config || !config.retry) {
        config.retry = true;
        config.retryCount = config.retryCount || 0;
        
        if (config.retryCount < 2) {
          config.retryCount += 1;
          console.log(`🔄 Retrying request (attempt ${config.retryCount})`);
          
          // Wait before retry
          await new Promise(resolve => setTimeout(resolve, 1000 * config.retryCount));
          return apiClient(config);
        }
      }
      
      return Promise.reject(error);
    }
  );

  const checkBackendHealth = useCallback(async () => {
    try {
      setBackendStatus("checking");
      const response = await apiClient.get("/health");
      
      if (response.data.model_loaded) {
        setIsConnected(true);
        setErrorMessage("");
        setBackendStatus("ready");
        consecutiveFailuresRef.current = 0;
        console.log("✅ Backend is healthy and ready");
        return true;
      } else {
        setBackendStatus("loading");
        setErrorMessage("Backend is loading model, please wait...");
        return false;
      }
    } catch (error) {
      console.error("❌ Health check failed:", error);
      setIsConnected(false);
      setBackendStatus("error");
      
      if (error.code === 'ECONNABORTED') {
        setErrorMessage("Backend timeout - server might be sleeping");
      } else if (error.message.includes('Network Error')) {
        setErrorMessage("Network error - check backend URL");
      } else {
        setErrorMessage(`Health check failed: ${error.message}`);
      }
      return false;
    }
  }, []);

  const optimizeImageForUpload = (canvas) => {
    // Reduce image size for faster upload
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 160; // Reduced from 224
    tempCanvas.height = 120; // Reduced from 224
    
    tempCtx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);
    return tempCanvas.toDataURL("image/jpeg", 0.6); // Lower quality for speed
  };

  const processFrame = useCallback(async (time) => {
    if (!videoRef.current || !canvasRef.current || isProcessingRef.current) {
      requestAnimationFrame(processFrame);
      return;
    }

    // Adaptive timing based on backend performance
    if (time - lastSentRef.current < requestInterval) {
      requestAnimationFrame(processFrame);
      return;
    }

    // Skip if backend isn't ready
    if (backendStatus !== "ready") {
      requestAnimationFrame(processFrame);
      return;
    }

    isProcessingRef.current = true;
    lastSentRef.current = time;

    try {
      const ctx = canvasRef.current.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, 224, 224);
      
      // Use optimized image
      const dataUrl = optimizeImageForUpload(canvasRef.current);
      const startTime = Date.now();

      const response = await apiClient.post("/predict", { image: dataUrl });
      const responseTime = Date.now() - startTime;
      
      setProcessingTime(responseTime);
      
      // Adaptive timing based on response time
      if (responseTime > 3000) {
        setRequestInterval(2000); // Slow down if backend is slow
      } else if (responseTime < 1000) {
        setRequestInterval(Math.max(500, requestInterval - 100)); // Speed up if fast
      }

      const probOpen = response.data.probability_open || 0;
      const faceFound = response.data.status !== "Not Detected";

      setProbability(probOpen);
      setFaceDetected(faceFound);
      setIsConnected(true);
      setErrorMessage("");
      consecutiveFailuresRef.current = 0;

      let newStatus = "Alert";
      if (!faceFound) newStatus = "Not Detected";
      else if (probOpen < CLOSED_EYE_THRESHOLD) newStatus = "Drowsy";

      setStatus(newStatus);

      const now = Date.now();

      if (faceFound && newStatus === "Drowsy") {
        if (!eyeClosedStartRef.current) eyeClosedStartRef.current = now;

        if (now - eyeClosedStartRef.current >= ALARM_DURATION) {
          if (alarmRef.current && alarmRef.current.paused) {
            alarmRef.current.play().catch(err => console.log("Audio play error:", err));
          }
          setDrowsyCount(prev => prev + 1);
          setLastDrowsyTime(new Date().toLocaleTimeString());
        }
      } else {
        eyeClosedStartRef.current = null;
        if (alarmRef.current && !alarmRef.current.paused) {
          alarmRef.current.pause();
          alarmRef.current.currentTime = 0;
        }
      }

      setProbHistory(prev => {
        const newHistory = [...prev, probOpen * 100];
        if (newHistory.length > 50) newHistory.shift(); // Reduce history size
        return newHistory;
      });

      // Visual feedback
      if (faceFound) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = newStatus === "Drowsy" ? "#ff4d4d" : "#00cc44";
        ctx.strokeRect(0, 0, 224, 224);
      }

    } catch (err) {
      console.error("❌ Prediction error:", err);
      consecutiveFailuresRef.current += 1;
      
      if (consecutiveFailuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        setIsConnected(false);
        setBackendStatus("error");
        
        if (err.code === 'ECONNABORTED') {
          setErrorMessage(`Request timeout (${requestInterval}ms) - backend is overloaded`);
          setRequestInterval(Math.min(requestInterval * 1.5, 5000)); // Slow down
        } else if (err.code === 'ERR_NETWORK') {
          setErrorMessage("Network error - backend might be down");
        } else {
          setErrorMessage(`Error: ${err.message}`);
        }
      }
    } finally {
      isProcessingRef.current = false;
    }

    requestAnimationFrame(processFrame);
  }, [backendStatus, requestInterval]);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 }, 
            height: { ideal: 480 },
            frameRate: { ideal: 15, max: 30 } // Limit frame rate
          } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera error:", err);
        setErrorMessage("Camera access denied");
      }
    };

    startCamera();
    
    // Initialize audio
    alarmRef.current = new Audio("/alarm.mp3");
    alarmRef.current.loop = true;

    // Start health checking
    checkBackendHealth();
    
    // Regular health checks
    const healthInterval = setInterval(checkBackendHealth, 30000);
    
    // Start frame processing
    requestAnimationFrame(processFrame);

    return () => {
      clearInterval(healthInterval);
      if (alarmRef.current) {
        alarmRef.current.pause();
      }
    };
  }, [processFrame, checkBackendHealth]);

  const forceWakeBackend = async () => {
    setBackendStatus("waking");
    setErrorMessage("Waking up backend, please wait...");
    
    try {
      // Make multiple requests to wake up the backend
      for (let i = 0; i < 3; i++) {
        await apiClient.get("/", { timeout: 20000 });
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      await checkBackendHealth();
    } catch (error) {
      console.error("Wake up failed:", error);
      setErrorMessage("Failed to wake up backend");
      setBackendStatus("error");
    }
  };

  const chartData = {
    labels: probHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "Eyes Open Probability",
        data: probHistory,
        borderColor: status === "Drowsy" ? "#ff4d4d" : "#00cc44",
        backgroundColor: status === "Drowsy" ? "rgba(255,77,77,0.2)" : "rgba(0,204,68,0.2)",
        tension: 0.3,
        pointRadius: 1, // Smaller points for performance
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    animation: { duration: 0 }, // Disable animations for performance
    plugins: {
      legend: { display: false },
      title: { display: true, text: "Eyes Open Probability", font: { size: 16 } }
    },
    scales: {
      y: { min: 0, max: 100, title: { display: true, text: "%" } },
      x: { display: false } // Hide x-axis for performance
    }
  };

  const getStatusColor = () => {
    switch (backendStatus) {
      case "ready": return "#00cc44";
      case "loading": return "#ffaa00";
      case "checking": return "#0099ff";
      case "waking": return "#ff6600";
      case "error": return "#ff4444";
      default: return "#666666";
    }
  };

  const getStatusText = () => {
    switch (backendStatus) {
      case "ready": return "Ready";
      case "loading": return "Loading Model...";
      case "checking": return "Checking...";
      case "waking": return "Waking Up...";
      case "error": return "Error";
      default: return "Unknown";
    }
  };

  return (
    <div style={{ fontFamily: "Arial, sans-serif", background: "linear-gradient(135deg, #efebe2ff, #f0e0c6ff)", minHeight: "100vh", padding: "20px" }}>
      
      {/* Status Banner */}
      <div style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        backgroundColor: getStatusColor(),
        color: "white",
        padding: "8px",
        textAlign: "center",
        zIndex: 1000,
        fontSize: "14px",
        fontWeight: "bold"
      }}>
        Backend Status: {getStatusText()} 
        {processingTime > 0 && ` | Response: ${processingTime}ms`}
        {backendStatus === "error" && (
          <button 
            onClick={forceWakeBackend}
            style={{
              marginLeft: "15px",
              padding: "4px 12px",
              backgroundColor: "white",
              color: getStatusColor(),
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontWeight: "bold",
              fontSize: "12px"
            }}
          >
            Wake Up Backend
          </button>
        )}
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div style={{
          margin: "50px auto 20px",
          padding: "15px",
          backgroundColor: "#ffe6e6",
          border: "1px solid #ff9999",
          borderRadius: "8px",
          textAlign: "center",
          maxWidth: "800px",
          fontSize: "14px"
        }}>
          {errorMessage}
        </div>
      )}

      {/* Main Title */}
      <h1 style={{
        textAlign: "center",
        marginBottom: "30px",
        marginTop: "50px",
        fontSize: "2.5rem",
        fontWeight: "700",
        background: "linear-gradient(90deg, #e73eb1ff, #891bbbff)",
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
        letterSpacing: "1px"
      }}>
        🚗 Driver Drowsiness Detection
      </h1>

      {/* Main Content */}
      <div style={{ display: "flex", justifyContent: "center", gap: "20px", flexWrap: "wrap" }}>
        
        {/* Video Panel */}
        <div style={{
          position: "relative",
          borderRadius: "15px",
          overflow: "hidden",
          boxShadow: "0 8px 20px rgba(0,0,0,0.2)",
          backgroundColor: "#fff"
        }}>
          <video
            ref={videoRef}
            width="640"
            height="480"
            autoPlay
            playsInline
            muted
            style={{ display: "block", borderRadius: "15px" }}
          />
          <canvas ref={canvasRef} width="224" height="224" style={{ display: "none" }} />
          <div style={{
            position: "absolute",
            bottom: "10px",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "rgba(0,0,0,0.8)",
            color: "#fff",
            padding: "8px 20px",
            borderRadius: "10px",
            fontWeight: "bold",
            fontSize: "16px"
          }}>
            {backendStatus === "ready" ? (
              faceDetected
                ? `${status} (${(probability * 100).toFixed(1)}% open)`
                : "No Face Detected"
            ) : (
              `Backend ${getStatusText()}`
            )}
          </div>
        </div>

        {/* Stats Panel */}
        <div style={{
          backgroundColor: "#fff",
          borderRadius: "15px",
          padding: "25px",
          width: "300px",
          boxShadow: "0 8px 20px rgba(0,0,0,0.2)"
        }}>
          <h3 style={{
            fontSize: "24px",
            marginBottom: "20px",
            textAlign: "center",
            background: "linear-gradient(90deg, #e73eb1ff, #891bbbff)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>Statistics</h3>
          
          <div style={{ marginBottom: "15px", fontSize: "16px" }}>
            <strong>Drowsy Alerts:</strong> {drowsyCount}
          </div>
          <div style={{ marginBottom: "15px", fontSize: "16px" }}>
            <strong>Last Alert:</strong> {lastDrowsyTime || "None"}
          </div>
          <div style={{ marginBottom: "15px", fontSize: "16px" }}>
            <strong>Face Detected:</strong> {faceDetected ? "Yes" : "No"}
          </div>
          <div style={{ marginBottom: "20px", fontSize: "16px" }}>
            <strong>Update Rate:</strong> {Math.round(1000/requestInterval)} fps
          </div>

          {/* Progress Bar */}
          <div style={{ marginTop: "20px" }}>
            <div style={{ 
              height: "25px", 
              backgroundColor: "#eee", 
              borderRadius: "12px", 
              overflow: "hidden",
              border: "2px solid #ddd"
            }}>
              <div style={{
                width: `${(probability * 100).toFixed(1)}%`,
                height: "100%",
                background: status === "Drowsy"
                  ? "linear-gradient(to right, #ff4d4d, #ff0000)"
                  : "linear-gradient(to right, #00cc44, #009900)",
                transition: "width 0.5s ease"
              }} />
            </div>
            <p style={{ fontSize: "14px", textAlign: "center", marginTop: "8px", fontWeight: "bold" }}>
              Eyes Open: {(probability * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Chart */}
      {probHistory.length > 0 && (
        <div style={{
          marginTop: "30px",
          padding: "20px",
          backgroundColor: "#fff",
          borderRadius: "15px",
          boxShadow: "0 8px 20px rgba(0,0,0,0.2)",
          maxWidth: "800px",
          margin: "30px auto"
        }}>
          <Line data={chartData} options={chartOptions} />
        </div>
      )}
    </div>
  );
}

export default App;