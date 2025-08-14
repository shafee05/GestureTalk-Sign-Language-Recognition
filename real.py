import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Get class names from the model
class_names = model.names  # e.g., {0: 'hello', 1: 'thanks', 2: 'goodbye'}

# Optional: Hand symbol overlays
def draw_custom_symbol(frame, label):
    h, w = frame.shape[:2]
    cx = 100
    cy = h - 100

    # Draw a rectangle and label on the frame
    cv2.rectangle(frame, (cx - 20, cy - 60), (cx + 120, cy + 10), (50, 50, 50), -1)
    cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get predictions from the model
    results = model(frame, verbose=False)
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])  # Get the predicted class ID
                conf = float(box.conf[0])  # Get the confidence score
                label = class_names[cls_id]  # Get the corresponding label name

                # Filter out predictions with low confidence (you can adjust the threshold as needed)
                if conf > 0.5:  # Only display predictions with confidence > 0.5
                    # Draw the bounding box around the detected object
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Optional custom sign overlay (for visual sign)
                    draw_custom_symbol(frame, label)

                    # Print out the class and confidence for debugging
                    print(f"Predicted class ID: {cls_id}, label: {label}, confidence: {conf:.2f}")

    # Display the frame
    cv2.imshow("Sign Gesture Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
