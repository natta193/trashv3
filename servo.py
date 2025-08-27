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
        
        self.steering_trim = 15
        self.steering_angle = 90 + self.steering_trim
        if self.steering_trim < 90:
            self.steering_range = [
                0, 90+self.steering_trim, 180 - self.steering_trim]  
        else:
            self.steering_range = [
                2*self.steering_trim, 90+self.steering_trim, 180]
            
        self.mode = 'drive' # or 'retrieve'
        self.lock = threading.Lock()

    def update(self):
        for i, angle in enumerate(self.target_angles):
            self.current_angles[i] += max(-5, min(angle - self.current_angles[i], 5))
            self.arm_servos[i].angle = self.arm_angles[i]
    
    def calibrate(self):
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
            
    def switch_mode(self, mode):
        self.mode = mode
        self.target_angles = self.neutral.copy()
        if mode == 'drive':
            self.steering.angle = self.steering_range[1]
            self.motor.angle = 0.2*180
        elif mode == 'retrieve':
            self.motor.angle = 0
            
    def move_arm(self, rel_x, rel_y, area):
        pass # i will code this myself
    
    def steer(self, rel_x):
        pass # i will code this myself

servo_controller = ServoController()
