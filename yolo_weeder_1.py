import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# -------------------- LOAD YOLO MODEL --------------------
MODEL_PATH = "/home/edwin/Documents/YOLO/best_float32.tflite"  # <-- change to your file

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

HEIGHT = input_details[0]["shape"][1]
WIDTH = input_details[0]["shape"][2]

print("YOLO model loaded.")
print(f"Input size: {WIDTH}x{HEIGHT}")

# -------------------- LABELS --------------------
class_labels = ["tomato", "greens"]   # index 0,1

# -------------------- CAMERA SETUP --------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (1.0, 1.0)
})

picam2.start()
print("Camera started. Press 'q' to quit.")


# -------------------- POST-PROCESSING --------------------
def xywh_to_xyxy(x, y, w, h, img_w, img_h):
    """Convert YOLO format (center x,y,w,h) to bounding box corners"""
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2


# -------------------- MAIN LOOP --------------------
try:
    while True:
        frame = picam2.capture_array()
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Prepare input for YOLO
        resized = cv2.resize(img, (WIDTH, HEIGHT))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, 0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # YOLO outputs
        outputs = interpreter.get_tensor(output_details[0]["index"])[0]

        # Loop through detections
        for det in outputs:
            score = det[4]
            if score < 0.5:
                continue

            class_id = int(det[5])
            label = class_labels[class_id]

            x, y, w, h = det[0], det[1], det[2], det[3]
            x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h, img.shape[1], img.shape[0])

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show final frame
        cv2.imshow("YOLO Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("ERROR:", e)

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")
