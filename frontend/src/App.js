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

  const [status, setStatus] = useState("Waiting...");
  const [probability, setProbability] = useState(0);
  const [faceDetected, setFaceDetected] = useState(true);
  const [drowsyCount, setDrowsyCount] = useState(0);
  const [lastDrowsyTime, setLastDrowsyTime] = useState(null);
  const [probHistory, setProbHistory] = useState([]);

  const CLOSED_EYE_THRESHOLD = 0.65;
  const ALARM_DURATION = 2000;

  const processFrame = useCallback(async (time) => {
    if (!videoRef.current || !canvasRef.current) {
      requestAnimationFrame(processFrame);
      return;
    }

    if (time - lastSentRef.current > 500) {
      lastSentRef.current = time;

      const ctx = canvasRef.current.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, 224, 224);
      const dataUrl = canvasRef.current.toDataURL("image/jpeg");

      try {
        const res = await axios.post("http://127.0.0.1:5000/predict", { image: dataUrl });
        const probOpen = res.data.probability_open;
        const faceFound = res.data.status !== "Not Detected";

        setProbability(probOpen);
        setFaceDetected(faceFound);

        let newStatus = "Alert";
        if (!faceFound) newStatus = "Not Detected";
        else if (probOpen < CLOSED_EYE_THRESHOLD) newStatus = "Drowsy";

        setStatus(newStatus);

        const now = Date.now();

        if (faceFound && newStatus === "Drowsy") {
          if (!eyeClosedStartRef.current) eyeClosedStartRef.current = now;

          if (now - eyeClosedStartRef.current >= ALARM_DURATION) {
            if (alarmRef.current.paused) alarmRef.current.play().catch(err => console.log(err));
            setDrowsyCount(prev => prev + 1);
            setLastDrowsyTime(new Date().toLocaleTimeString());
          }
        } else {
          eyeClosedStartRef.current = null;
          if (!alarmRef.current.paused) {
            alarmRef.current.pause();
            alarmRef.current.currentTime = 0;
          }
        }

        setProbHistory(prev => {
          const newHistory = [...prev, probOpen * 100];
          if (newHistory.length > 100) newHistory.shift();
          return newHistory;
        });

        if (faceFound) {
          ctx.lineWidth = 3;
          ctx.strokeStyle = newStatus === "Drowsy" ? "#ff4d4d" : "#00cc44";
          ctx.strokeRect(0, 0, 224, 224);
        }

      } catch (err) {
        console.error("Prediction error:", err);
      }
    }

    requestAnimationFrame(processFrame);
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera error:", err);
      }
    };

    startCamera();
    alarmRef.current = new Audio("/alarm.mp3");
    alarmRef.current.loop = true;
    requestAnimationFrame(processFrame);
  }, [processFrame]);

  const chartData = {
    labels: probHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: "Eyes Open Probability",
        data: probHistory,
        borderColor: status === "Drowsy" ? "#ff4d4d" : "#00cc44",
        backgroundColor: status === "Drowsy" ? "rgba(255,77,77,0.2)" : "rgba(0,204,68,0.2)",
        tension: 0.3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: true, text: "Eyes Open Probability Over Time", font: { size: 18 } }
    },
    scales: {
      y: { min: 0, max: 100, title: { display: true, text: "Probability %" } },
      x: { title: { display: true, text: "Frames" } }
    }
  };

  return (
    <div style={{ fontFamily: "Arial, sans-serif", background: "linear-gradient(135deg, #efebe2ff, #f0e0c6ff)", minHeight: "100vh", padding: "30px" }}>
      {/* Attractive Heading */}
      <h1 style={{
        textAlign: "center",
        marginBottom: "40px",
        fontSize: "3rem",
        fontWeight: "700",
        background: "linear-gradient(90deg, #e73eb1ff, #891bbbff)",
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
        textShadow: "2px 2px 10px rgba(0,0,0,0.2)",
        letterSpacing: "1px"
      }}>
        🚗 Driver Drowsiness Detection Dashboard
      </h1>

      {/* Top Panels */}
      <div style={{ display: "flex", justifyContent: "center", gap: "30px", flexWrap: "wrap" }}>
        {/* Video Panel */}
        <div style={{
          position: "relative",
          borderRadius: "20px",
          overflow: "hidden",
          boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
          backgroundColor: "#fff"
        }}>
          <video
            ref={videoRef}
            width="640"
            height="480"
            autoPlay
            playsInline
            muted
            style={{ display: "block", borderRadius: "20px" }}
          />
          <canvas ref={canvasRef} width="224" height="224" style={{ display: "none" }} />
          <div style={{
            position: "absolute",
            bottom: "10px",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "rgba(0,0,0,0.7)",
            color: "#fff",
            padding: "10px 25px",
            borderRadius: "12px",
            fontWeight: "bold",
            fontSize: "18px",
            boxShadow: "0 2px 6px rgba(0,0,0,0.3)"
          }}>
            {faceDetected
              ? `${status} (${(probability * 100).toFixed(1)}% eyes open)`
              : "Face Not Detected"}
          </div>
        </div>

        {/* Analysis Panel */}
        <div style={{
          backgroundColor: "#fff",
          borderRadius: "20px",
          padding: "30px",
          width: "380px",
          boxShadow: "0 10px 25px rgba(0,0,0,0.3)"
        }}>
          <p style={{
            fontSize: "40px", marginBottom: "65px", color: "#333", textAlign: "center", fontWeight: "bold", background: "linear-gradient(90deg, #e73eb1ff, #891bbbff)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>Analysis</p>
          <p style={{ fontSize: "18px", fontWeight: "500", marginBottom: "30px" }}><strong>Drowsy Alerts:</strong> {drowsyCount}</p>
          <p style={{ fontSize: "18px", fontWeight: "500", marginBottom: "30px" }}><strong>Last Alert Time:</strong> {lastDrowsyTime || "--:--:--"}</p>
          <p style={{ fontSize: "18px", fontWeight: "500", marginBottom: "50px" }}><strong>Face Detected:</strong> {faceDetected ? "Yes" : "No"}</p>

          <div style={{ marginTop: "20px" }}>
            <div style={{ height: "45px", backgroundColor: "#eee", borderRadius: "22px", overflow: "hidden" }}>
              <div style={{
                width: `${(probability * 100).toFixed(1)}%`,
                height: "100%",
                background: status === "Drowsy"
                  ? "linear-gradient(to right, #ff4d4d, #ff0000)"
                  : "linear-gradient(to right, #00cc44, #009900)",
                transition: "width 0.3s"
              }} />
            </div>
            <p style={{ fontSize: "18px", textAlign: "center", fontWeight: "bold", marginTop: "10px" }}>Eyes Open Probability</p>
          </div>
        </div>
      </div>

      {/* Live Chart Panel */}
      <div style={{
        marginTop: "50px",
        padding: "25px",
        backgroundColor: "#fff",
        borderRadius: "20px",
        boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
        width: "1050px",
        maxWidth: "100%",
        marginLeft: "auto",
        marginRight: "auto"
      }}>
        <Line data={chartData} options={chartOptions} />
      </div>
    </div>
  );
}

export default App;
