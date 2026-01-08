import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import time

# --------------------- LOAD MODEL ---------------------
interpreter = tflite.Interpreter(model_path="/home/edwin/Documents/YOLO/best_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# --------------------- CAMERA SETUP ---------------------
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# --------------------- INFERENCE LOOP ---------------------
while True:
    frame = picam2.capture_array()
    img = cv2.resize(frame, (input_width, input_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])


    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
