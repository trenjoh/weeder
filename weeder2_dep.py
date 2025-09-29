import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import time

print("Starting in 10 seconds...")
time.sleep(10)

try:
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path="/home/edwin/Documents/weeder1.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height, input_width = input_details[0]['shape'][1:3]

    print(f'Model expects input: {input_details[0]["shape"]}')

    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    last_print_time = time.time()
    display_interval = 2  # seconds

    class_labels = {0: 'tomato', 1: 'greens'}

    while True:
        img = picam2.capture_array()

        # (Optional) Display camera feed
        cv2.imshow('Camera Preview', img)

        # Preprocess
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_width, input_height))
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)
        predicted_class_label = class_labels.get(predicted_class_index, "Unknown")

        # Print every 2s
        if time.time() - last_print_time >= display_interval:
            print(f'Predicted class: {predicted_class_label}')
            last_print_time = time.time()

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f'Error: {e}')

finally:
    picam2.stop()
    cv2.destroyAllWindows()