import cv2
from ultralytics import YOLO

class SentryDetector:
    """
    Wrapper for the YOLOv8 object detection model, filtered for specific security-relevant classes.

    Attributes:
        model (YOLO): The loaded YOLOv8 model.
        allowed_classes (set): Set of COCO class IDs to report (0: person, 43: knife, 76: scissors).
        class_names (dict): Mapping of class IDs to labels.
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initializes the SentryDetector with a specific YOLO model.

        Args:
            model_path (str): Path to the YOLOv8 model weights. Defaults to 'yolov8n.pt'.
        """
        self.model = YOLO(model_path)
        # Filter for Person (0), Knife (43), Scissors (76)
        self.allowed_classes = {0, 43, 76}
        self.class_names = self.model.names

    def detect(self, frame: cv2.typing.MatLike) -> list[dict]:
        """
        Performs object detection on a single video frame.

        Args:
            frame (cv2.typing.MatLike): The input image frame from the video feed.

        Returns:
            list[dict]: A list of detection dictionaries, each containing:
                - 'box' (list[int]): [x1, y1, x2, y2] bounding box coordinates.
                - 'conf' (float): Confidence score of the detection.
                - 'class_id' (int): The COCO class ID.
                - 'label' (str): The human-readable class label.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id in self.allowed_classes and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.class_names.get(class_id, "Unknown")

                detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'class_id': class_id,
                    'label': label
                })

        return detections
