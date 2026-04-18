import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.onnx', task='detect')

cap = cv2.VideoCapture(0)

print("Application started. Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.5)
    
    count = len(results[0].boxes)
    
    annotated_frame = results[0].plot()
    
    display_text = f"Invasive Green Crabs: {count}"
    cv2.putText(annotated_frame, display_text, (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("MATE ROV 2026 - SmartAtlantic Alliance", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed.")