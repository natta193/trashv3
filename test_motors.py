from adafruit_servokit import ServoKit
import time

servokit = ServoKit(channels=16)

# for i in range(4):
#     servokit.servo[i].angle = 90
#     time.sleep(0.2)

servokit.servo[3].angle = 180
time.sleep(1)
servokit.servo[3].angle = 55
