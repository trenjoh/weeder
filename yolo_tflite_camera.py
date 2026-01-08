import time
import numpy as np
import cv2
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "/home/edwin/Documents/YOLO/best_float32.tflite"
CLASSES_FILE = "classes.txt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

with open(CLASSES_FILE) as f:
    class_names = [c.strip() for c in f]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

h_in, w_in = input_details[0]["shape"][1:3]

# camera
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "XBGR8888"},
    controls={"FrameRate": 30}
)

picam2.configure(config)
picam2.start()

# LET AE/AWB SETTLE
time.sleep(2)

# NOW lock exposure & white balance
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})

meta = picam2.capture_metadata()
print("ExposureTime:", meta["ExposureTime"])
print("AnalogueGain:", meta["AnalogueGain"])
print("ColourGains:", meta["ColourGains"])

def nms(boxes, scores):
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()

print("Starting detection...")
prev_time = time.time()

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    frame = np.ascontiguousarray(frame)
    
    frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=20)
    print("Mean brightness:", frame.mean())


    resized = cv2.resize(frame, (w_in, h_in))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    raw = interpreter.get_tensor(output_details[0]["index"])[0]
    preds = raw.transpose(1, 0)  # (8400, 6)

    boxes, scores, class_ids = [], [], []

    for p in preds:
        x, y, w, h = p[:4]
        cls_id = np.argmax(p[4:])
        score = p[4 + cls_id]

        if score < CONF_THRESHOLD:
            continue

        x1 = int((x - w / 2) * frame.shape[1])
        y1 = int((y - h / 2) * frame.shape[0])
        bw = int(w * frame.shape[1])
        bh = int(h * frame.shape[0])

        boxes.append([x1, y1, bw, bh])
        scores.append(float(score))
        class_ids.append(cls_id)

    keep = nms(boxes, scores)

    for i in keep:
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    fps = 1 / max(time.time() - prev_time, 1e-6)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("YOLO TFLite Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
