import os
import platform
import threading
import time
from typing import List, Dict, Optional

import cv2
import numpy as np

from detector import SentryDetector

# Optional audio dependency
try:
    from playsound import playsound
except ImportError:
    playsound = None

__version__ = "1.0.0"

# Configuration
DANGER_ZONE_COLOR = (0, 0, 255)  # Red
SAFE_COLOR = (0, 255, 0)      # Green
ALERT_COOLDOWN_SECONDS = 5    # Prevent alarm fatigue


class SentrySystem:
    """
    Manages the Sentry-AI security system, including video capture,
    threat detection, geofencing, and alerting mechanisms.
    """

    def __init__(self):
        self._cap = cv2.VideoCapture(0)
        self._detector = SentryDetector()
        self._danger_zone: Optional[np.ndarray] = None

        # internal state
        self._last_alert_time = 0.0
        self._frame_count = 0
        self._current_detections: List[Dict] = []
        self._system_status = "ARMED"

    def _initialize_danger_zone(self, frame_width: int, frame_height: int) -> None:
        """
        Defines a rectangular polygon covering the central 50% of the frame.
        """
        margin_x = int(frame_width * 0.25)
        margin_y = int(frame_height * 0.25)
        
        # Define vertices for a centered rectangle
        points = np.array([
            [margin_x, margin_y],
            [frame_width - margin_x, margin_y],
            [frame_width - margin_x, frame_height - margin_y],
            [margin_x, frame_height - margin_y]
        ], dtype=np.int32)
        
        self._danger_zone = points.reshape((-1, 1, 2))

    def _play_alarm(self) -> None:
        """
        Plays an alert sound using platform-specific methods in a non-blocking thread.
        Falls back to terminal bell if no audio output is available.
        """
        try:
            if platform.system() == 'Darwin':
                # macOS native fallback
                os.system('afplay /System/Library/Sounds/Glass.aiff')
            elif playsound:
                # Windows/Linux with playsound installed
                print("Audio alert triggered (Update path to a real mp3/wav file).")
                # playsound('alert.mp3') 
            else:
                # Universal fallback
                print('\a')
        except Exception as e:
            print(f"[Error] Failed to play alarm: {e}")

    def _trigger_alert(self, frame: np.ndarray, detection: Dict) -> None:
        """
        Handles threat actions: logging, evidence saving, and sound.
        """
        current_time = time.time()
        if current_time - self._last_alert_time > ALERT_COOLDOWN_SECONDS:
            print(f"!!! THREAT DETECTED: {detection['label']} !!!")
            self._last_alert_time = current_time

            timestamp = int(current_time)
            filename = f"threat_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            threading.Thread(target=self._play_alarm, daemon=True).start()

    def _is_inside_zone(self, box: List[int]) -> bool:
        """
        Determines if the center of a bounding box lies within the danger zone.
        """
        if self._danger_zone is None:
            return False

        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # pointPolygonTest returns >0 (inside), 0 (edge), <0 (outside)
        return cv2.pointPolygonTest(self._danger_zone, (center_x, center_y), False) >= 0

    def run(self) -> None:
        """
        Main event loop. Captures frames, runs detection, and updates the UI.
        """
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        # Initialize zone based on the first frame reading
        ret, frame = self._cap.read()
        if ret:
            h, w = frame.shape[:2]
            self._initialize_danger_zone(w, h)

        print(f"Sentry-AI v{__version__} ARMED. Press 'q' to quit.")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                self._frame_count += 1
                
                # Performance Optimization: Run neural net inference only every 3rd frame
                if self._frame_count % 3 == 0:
                    self._current_detections = self._detector.detect(frame)

                threat_active = False

                for det in self._current_detections:
                    box = det['box']
                    label = det['label']
                    conf = det['conf']
                    
                    in_zone = self._is_inside_zone(box)
                    
                    color = SAFE_COLOR
                    if in_zone:
                        color = DANGER_ZONE_COLOR
                        threat_active = True
                        self._trigger_alert(frame, det)
                    
                    # Draw bounding box and label
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Visualize Danger Zone
                zone_color = DANGER_ZONE_COLOR if threat_active else SAFE_COLOR
                if self._danger_zone is not None:
                    cv2.polylines(frame, [self._danger_zone], True, zone_color, 2)

                # UI Overlay
                cv2.putText(frame, f"System: {self._system_status}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                cv2.imshow('Sentry-AI Dashboard', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self._cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SentrySystem()
    app.run()
