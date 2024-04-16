import cv2
from tqdm import tqdm
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

model = YOLOWorld(model_id="yolo_world/l")
classes = ["eye", "lips"]
model.set_classes(classes)

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)

cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.infer(frame, confidence=0.002)
    detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(frame, detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)

    cv2.imshow("Webcam", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
