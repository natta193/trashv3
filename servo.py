from adafruit_servokit import ServoKit
import time
import numpy as np
import threading
import vision

kit = ServoKit(channels=16)

class ServoController:
    def __init__(self):
        self.current_angles = [90, 90, 90]
        self.target_angles = [90, 90, 90]
        self.neutral = [90, 90, 90]
        
        self.arm_servos = [kit.servo[i] for i in range(3)]
        self.gripper = kit.servo[3]
        self.motor = kit.servo[4]
        self.steering = kit.servo[5]
        
        self.steering_trim = 10
        if self.steering_trim + 90 < 90:
            self.steering_range = [
                0, 90+self.steering_trim, 180 - self.steering_trim]  
        else:
            self.steering_range = [
                2*self.steering_trim, 90+self.steering_trim, 180]
            
        self.mode = 'drive' # or 'retrieve'
        self.lock = threading.Lock()
        
        self.motor.angle = 90 + 0.2*90
        self.steering.angle = self.steering_range[1]
        
        self.calibrating = False

    def update(self):
        if not self.calibrating:
            for i, angle in enumerate(self.target_angles):
                self.current_angles[i] += max(-5, min(angle - self.current_angles[i], 5))
                self.arm_servos[i].angle = self.current_angles[i]
    
    def calibrate(self):
        self.calibrating = True
        for i in range (90, 180, 3):
            for servo in self.arm_servos:
                servo.angle = i
            time.sleep(0.05)

        for i in range (180, 0, -3):
            for servo in self.arm_servos:
                servo.angle = i
            time.sleep(0.05)

        for i in range (0, 90, 3):
            for servo in self.arm_servos:
                servo.angle = i
            time.sleep(0.05)
        self.current_angles = self.neutral.copy()
        self.calibrating = False
            
    def switch_mode(self, mode):
        self.mode = mode
        self.target_angles = self.neutral.copy()
        if mode == 'drive':
            self.steering.angle = self.steering_range[1]
            self.motor.angle = 90+0.2*90
        elif mode == 'retrieve':
            self.motor.angle = 90
            
    def move_arm(self, rel_x, rel_y, area):
        # print("move arm", rel_x, rel_y, area)
        pass # i will code this myself
    
    def steer(self, rel_x):
        # print("steer", rel_x)
        self.steering.angle = self.steering_range[1] - rel_x * (self.steering_range[2] - self.steering_range[0]) / 2
        
    def stop_motor(self):
        self.motor.angle = 90

servo_controller = ServoController()
