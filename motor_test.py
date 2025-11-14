from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
print("Starting ...")
sleep(3)
factory = PiGPIOFactory()
servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
print("Playing...")
while True:
    servo.angle =0
    sleep(2)
    servo.angle =90
    sleep(2)
