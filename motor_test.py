import RPi.GPIO as GPIO
import time

# ----- GPIO pins -----
IN1 = 17
IN2 = 27
ENA = 22  # PWM pin

GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# ----- PWM setup -----
pwm = GPIO.PWM(ENA, 1000)  # 1 kHz
pwm.start(0)  # start stopped

# ----- Motor functions -----
def forward(speed):
    pwm.ChangeDutyCycle(speed)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

def reverse(speed):
    pwm.ChangeDutyCycle(speed)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)

def stop():
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

# ----- Main loop -----
speed = 20  # exact speed %

try:
    while True:
        print("Moving forward at 20% for 7 seconds")
        forward(speed)
        time.sleep(7)

        print("Stopping 1 second")
        stop()
        time.sleep(1)

        print("Reversing at 20% for 7 seconds")
        reverse(speed)
        time.sleep(7)

        print("Stopping 1 second")
        stop()
        time.sleep(1)

finally:
    stop()
    pwm.stop()
    GPIO.cleanup()
    print("Done")
