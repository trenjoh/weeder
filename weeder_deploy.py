# The progam is to help us deploy ,the model we trained earlier, in Raspberry (inside geany environment).
# The camera feed streams images while interpretors makes predictions and lables/classifies the images as thy are being streamed by the camera.
# a preview window pops in the screen while the predictions runs in the teminal unless interrupted.
import numpy as np
import cv2

import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2, Preview
import time  # For handling the timer

# Delay the execution for 10 seconds
print("Starting in 10 seconds...")
time.sleep(10)

try:
    # Loading the TFLite model
    interpreter = tflite.Interpreter(model_path="/home/edwin/Documents/weeder1.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f'Input shape: {input_details[0]["shape"]}')  # Print expected input shape

    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))  # Set a larger preview size
    picam2.start()

   
    last_print_time = time.time()
    display_interval = 2  # Display prediction every 2 seconds
    start_time = time.time()  

    while True:  
        # Capture image from PiCamera
        img = picam2.capture_array()

        # Display the live camera feed
        cv2.imshow('Camera Preview', img)

        # Process the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary
        img_rgb = cv2.resize(img_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))  # Resize to match model input
        img_rgb = img_rgb.astype(np.float32)  # Convert to float32
        img_rgb = np.expand_dims(img_rgb, axis=0)  # Add batch dimension

        # Normalize the image if required by the model (between 0 and 1)
        img_rgb /= 255.0

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_rgb)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)

        # Mapping class indices to labels
        class_labels = {0: 'tomato', 1: 'greens'}
        predicted_class_label = class_labels.get(predicted_class_index, "Unknown class")

        # Check if enough time has passed to display the next prediction
        current_time = time.time()
        text = predicted_class_label
        print(f'Predicted class: {text}')
    
        
        if current_time - last_print_time >= display_interval:
            print(f'Predicted class: {text}')
            last_print_time = current_time  # Reset the timer

        # Save the captured image
        cv2.imwrite('/home/edwin/Documents/captured_image5.jpg', img)  # Change the path if needed
          
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f'Error: {e}')

finally:
    picam2.stop()  
    cv2.destroyAllWindows()  # Close the OpenCV window
