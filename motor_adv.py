import RPi.GPIO as GPIO
import sys, termios, tty
import time

# ----- GPIO setup -----
IN1 = 17
IN2 = 27
ENA = 22      # PWM pin

GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# PWM at 1 kHz
pwm = GPIO.PWM(ENA, 1000)
pwm.start(0)   # motor stopped

speed = 20  # exact speed %

def forward():
    pwm.ChangeDutyCycle(speed)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

def reverse():
    pwm.ChangeDutyCycle(speed)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)

def stop():
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

# ----- Real-time key detection -----
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("Hold keys: 'w' = forward, 's' = reverse, 'q' = quit")

try:
    while True:
        key = get_key()
        if key == 'w':
            forward()
        elif key == 's':
            reverse()
        elif key == 'q':
            break
        else:
            stop()
        # Stop if key is released
        time.sleep(0.05)
        stop()
finally:
    stop()
    pwm.stop()
    GPIO.cleanup()
    print("Clean exit")
