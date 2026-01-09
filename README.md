#  Sentry-CV: Intelligent Threat Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-magenta)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> **"Smart cameras shouldn't just record; they should react."**

**Sentry-CV** is a real-time computer vision security system that turns any standard webcam into an intelligent guardian. Unlike motion detectors that trigger on every shadow, Sentry-CV uses **YOLOv8** to semantically understand *what* it sees (Weapons vs. People) and *where* they are (Geofencing).

---

##  Key Features

* **Edge AI Inference**: Runs locally on CPU using the optimized YOLOv8 Nano model.
* **Dynamic Geofencing**: Users define a "Danger Zone". Threats outside the zone are tracked but ignored; threats *inside* trigger the alarm.
* **Weapon Recognition**: Specifically trained to detect high-risk objects (Knives, Scissors) in real-time.
* **Performance Optimized**: Implements **Skip-Frame Logic** (processing every Nth frame) to maintain high FPS on standard hardware.
* **Async Alerting**: Uses Python threading to play audio alarms without blocking the video feed.
* **Evidence Locking**: Automatically captures and timestamps a snapshot (`threat_timestamp.jpg`) the moment a breach occurs.

---

## Architecture

The system follows a non-blocking pipeline designed for low latency:

```mermaid
graph LR
    A[Webcam Feed] --> B{Frame Skipper}
    B -->|Every 3rd Frame| C[YOLOv8 Inference]
    B -->|Skipped Frame| G[Visual Output]
    C --> D[Class Filtering]
    D --> E{Point-in-Polygon Check}
    E -->|Inside Zone| F[Trigger Alert Thread]
    F --> H[Play Sound + Save Image]# Sentry-CV
