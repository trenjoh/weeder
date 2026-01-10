import time
import numpy as np
import cv2
import threading
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
from gpiozero import AngularServo, OutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

# --------------------- CONFIGURATION ---------------------
MODEL_PATH = "/home/edwin/Documents/YOLO/best_float32.tflite"
CLASSES_FILE = "classes.txt"
CONF_THRESHOLD = 0.5  # Increased for high confidence
IOU_THRESHOLD = 0.45

# GPIO Configuration
SERVO_PIN = 18
LASER_PIN = 17

# Laser Parameters
LASER_DURATION = 2.0  # seconds per weed
LASER_COOLDOWN = 0.5  # seconds between shots
MIN_TARGET_SIZE = 30  # minimum bounding box size (pixels)

# Targeting Parameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
CENTER_TOLERANCE = 40  # pixels - how close to center before firing
SERVO_STEP = 3  # degrees per adjustment
SERVO_MOVE_DELAY = 0.1  # seconds to wait after servo movement

# --------------------- SETUP ---------------------
print("Initializing Center-Targeted Laser System...")

# Load class names
with open(CLASSES_FILE) as f:
    class_names = [c.strip() for c in f]

WEED_CLASS = "greens"
CROP_CLASS = "tomato"

# Load YOLO model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
h_in, w_in = input_details[0]["shape"][1:3]
print(f"Model input shape: {h_in}x{w_in}")

# Initialize GPIO
factory = PiGPIOFactory()
servo = AngularServo(SERVO_PIN, min_pulse_width=0.0006, max_pulse_width=0.0023, 
                     pin_factory=factory)
laser = OutputDevice(LASER_PIN, pin_factory=factory)
print("Hardware initialized.")

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "XBGR8888"},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()

# Let camera settle, then lock exposure
time.sleep(2)
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})

meta = picam2.capture_metadata()
print(f"Camera locked - Exposure: {meta['ExposureTime']}, Gain: {meta['AnalogueGain']}")

# --------------------- STATE VARIABLES ---------------------
laser_lock = threading.Lock()
servo_lock = threading.Lock()
last_shot_time = 0
total_weeds_eliminated = 0
total_crops_protected = 0
current_target = None
targeting_mode = False

# --------------------- TARGETING FUNCTIONS ---------------------

def is_centered(x_center, y_center):
    """Check if target is close enough to frame center."""
    distance = np.sqrt((x_center - CENTER_X)**2 + (y_center - CENTER_Y)**2)
    return distance <= CENTER_TOLERANCE

def adjust_servo_to_target(x_center, current_angle):
    """Calculate servo adjustment to center the target."""
    offset = x_center - CENTER_X
    
    # Positive offset = target is right of center, need to move servo right
    # Negative offset = target is left of center, need to move servo left
    
    if abs(offset) < CENTER_TOLERANCE:
        return current_angle, True  # Already centered
    
    # Calculate adjustment (proportional to offset)
    adjustment = (offset / CENTER_X) * SERVO_STEP
    new_angle = current_angle + adjustment
    
    # Clamp to servo limits
    new_angle = max(0, min(90, new_angle))
    
    return new_angle, False

def fire_laser_at_center(confidence):
    """Fire laser at frame center (where it's aimed)."""
    global last_shot_time, total_weeds_eliminated
    
    current_time = time.time()
    
    with laser_lock:
        if current_time - last_shot_time < LASER_COOLDOWN:
            return False
        
        print(f"\n🎯 TARGET CENTERED - ENGAGING")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   🔴 FIRING LASER for {LASER_DURATION}s...")
        
        laser.on()
        time.sleep(LASER_DURATION)
        laser.off()
        
        last_shot_time = current_time
        total_weeds_eliminated += 1
        print(f"   ✓ ELIMINATED (Total: {total_weeds_eliminated})")
        
        return True

def nms(boxes, scores):
    """Non-maximum suppression."""
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()

# --------------------- MAIN DETECTION LOOP ---------------------
print("\n🚀 Starting center-targeted weed elimination...")
print(f"Laser aims at: ({CENTER_X}, {CENTER_Y})")
print(f"Center tolerance: ±{CENTER_TOLERANCE}px")
print(f"Minimum confidence: {CONF_THRESHOLD:.0%}")
print("Press 'q' to quit\n")

prev_time = time.time()
frame_count = 0

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        frame = np.ascontiguousarray(frame)
        
        # Enhance brightness
        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=20)
        
        # Prepare input for YOLO
        resized = cv2.resize(frame, (w_in, h_in))
        input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
        
        # Run inference
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details[0]["index"])[0]
        preds = raw.transpose(1, 0)
        
        # Parse detections
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
        
        # Apply NMS
        keep = nms(boxes, scores)
        
        # Find weeds and select target
        weed_detections = []
        
        for i in keep:
            x, y, w, h = boxes[i]
            class_name = class_names[class_ids[i]]
            confidence = scores[i]
            
            # Calculate center point
            x_center = x + w // 2
            y_center = y + h // 2
            
            # Check minimum size
            if w < MIN_TARGET_SIZE or h < MIN_TARGET_SIZE:
                continue
            
            if class_name == WEED_CLASS:
                weed_detections.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'x_center': x_center, 'y_center': y_center,
                    'confidence': confidence
                })
                
                # Draw red bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"WEED {confidence:.0%}", (x, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            elif class_name == CROP_CLASS:
                total_crops_protected += 1
                # Draw green bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"CROP {confidence:.0%}", (x, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Select highest confidence weed as target
        if weed_detections:
            # Sort by confidence
            weed_detections.sort(key=lambda d: d['confidence'], reverse=True)
            target = weed_detections[0]
            
            # Draw crosshair on target
            cv2.drawMarker(frame, (target['x_center'], target['y_center']), 
                          (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
            
            # Check if target is centered
            if is_centered(target['x_center'], target['y_center']):
                # TARGET LOCKED - FIRE!
                targeting_mode = False
                
                # Draw "LOCKED" indicator
                cv2.circle(frame, (CENTER_X, CENTER_Y), CENTER_TOLERANCE, 
                          (0, 255, 0), 3)
                cv2.putText(frame, "LOCKED", (CENTER_X - 40, CENTER_Y - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                # Fire laser
                fire_thread = threading.Thread(
                    target=fire_laser_at_center,
                    args=(target['confidence'],)
                )
                fire_thread.start()
                
            else:
                # Target not centered - adjust servo
                targeting_mode = True
                
                with servo_lock:
                    current_angle = servo.angle if servo.angle is not None else 45
                    new_angle, centered = adjust_servo_to_target(
                        target['x_center'], current_angle
                    )
                    
                    if new_angle != current_angle:
                        servo.angle = new_angle
                        time.sleep(SERVO_MOVE_DELAY)
                
                # Draw targeting indicator
                cv2.circle(frame, (CENTER_X, CENTER_Y), CENTER_TOLERANCE, 
                          (0, 255, 255), 2)
                cv2.putText(frame, "TARGETING", (CENTER_X - 60, CENTER_Y - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # Draw line from center to target
                cv2.arrowedLine(frame, (CENTER_X, CENTER_Y),
                               (target['x_center'], target['y_center']),
                               (255, 255, 0), 2)
        else:
            targeting_mode = False
        
        # Draw center crosshair (laser aim point)
        cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (0, 255, 255),
                      cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame, (CENTER_X, CENTER_Y), 5, (0, 255, 255), -1)
        
        # Calculate FPS
        fps = 1 / max(time.time() - prev_time, 1e-6)
        prev_time = time.time()
        frame_count += 1
        
        # Display stats
        status = "🎯 TARGETING" if targeting_mode else "🔍 SCANNING"
        cv2.putText(frame, f"{status} | FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Eliminated: {total_weeds_eliminated} | Protected: {total_crops_protected}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Servo: {servo.angle:.1f}deg | Conf: {CONF_THRESHOLD:.0%}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Center-Targeted Laser System", frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

except KeyboardInterrupt:
    print("\n⚠️  Shutdown requested...")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n🛑 Shutting down...")
    laser.off()
    servo.angle = 45  # Return to center
    time.sleep(0.5)
    
    picam2.stop()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Frames processed: {frame_count}")
    print(f"Weeds eliminated: {total_weeds_eliminated}")
    print(f"Crops protected: {total_crops_protected}")
    if total_weeds_eliminated > 0:
        print(f"Average confidence: High (>{CONF_THRESHOLD:.0%})")
    print("="*60)
    print("✓ System shutdown complete.")
