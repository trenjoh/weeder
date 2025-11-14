import numpy as np
import cv2
import time
import threading
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

# --------------------- SETUP SECTION ---------------------
print("System initializing...")
time.sleep(3)

# --- Initialize Servo ---
factory = PiGPIOFactory()
servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
print("Servo initialized.")

def smooth_move(start, end, step=0.1, delay=0.06):
    """Move the servo smoothly from start to end."""
    if start < end:
        angles = np.arange(start, end + step, step)
    else:
        angles = np.arange(start, end - step, -step)
    for angle in angles:
        servo.angle = angle
        sleep(delay)

# --- Load the TFLite model ---
print("Loading model...")
interpreter = tflite.Interpreter(model_path="/home/edwin/Documents/weeder1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Model input shape: {input_details[0]['shape']}")

# --- Initialize PiCamera2 ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

# --- Color correction (tuned for plants) ---
picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (1.4, 1.8)
})
picam2.start()
print("Camera started. Press 'q' to quit.")

# --- Label Mapping ---
class_labels = {0: 'tomato', 1: 'greens'}

# --- Confidence threshold (adjust this if needed) ---
CONFIDENCE_THRESHOLD = 0.65  # 65% minimum confidence to trust prediction

last_print_time = time.time()
display_interval = 2  # seconds between printed predictions

# --------------------- SERVO THREAD ---------------------
def servo_motion():
    """Continuously move servo while camera runs."""
    while True:
        smooth_move(0, 90, step=2, delay=0.06)
        sleep(0.5)
        smooth_move(90, 0, step=1, delay=0.06)
        sleep(0.5)

# Run servo in a separate thread
servo_thread = threading.Thread(target=servo_motion, daemon=True)
servo_thread.start()

# --------------------- MAIN CAMERA + MODEL LOOP ---------------------
try:
    while True:
        frame = picam2.capture_array()
        cv2.imshow("Camera Preview", frame)

        # Preprocess for model input
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Remove batch dim

        # Get prediction and confidence
        predicted_index = np.argmax(output_data)
        confidence = output_data[predicted_index]

        # Print prediction only every few seconds
        current_time = time.time()
        if current_time - last_print_time >= display_interval:
            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_label = class_labels.get(predicted_index, "unknown")
                print(f"Detected: {predicted_label.upper()} (confidence: {confidence:.3f})")
            else:
                unsure_class = class_labels.get(predicted_index, "unknown")
                print(f"NaN - No clear plant (leaning '{unsure_class}' @ only {confidence:.3f} confidence)")
            last_print_time = current_time

        # Save latest frame (optional)
        cv2.imwrite('/home/edwin/Documents/image_v1/captured_image.jpg', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        picam2.stop()
    except:
        pass
    cv2.destroyAllWindows()
    servo.angle = 45  # Return servo to middle on exit (optional safety)
    print("System stopped gracefully. Well done!")
