from adafruit_servokit import ServoKit
import time
import numpy as np
import threading
import vision

kit = ServoKit(channels=16)

class ServoController:
    def __init__(self):
        self.current_angles = [90, 90, 90]
        self.neutral = [85, 70, 150]
        self.target_angles = self.neutral.copy()
        self.ranges = [[60, 110], [45, 135], [45, 180]]
        self.gripper_range = [55, 180]
        
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
        
        # self.drive = 90 + 0.2*90
        self.drive = 90
        
        self.motor.angle = self.drive
        self.steering.angle = self.steering_range[1]
        self.gripper.angle = self.gripper_range[1]
        
        self.calibrating = False
        
        self.retrieve = False
        
        self.height_offset = 0
        self.reach_scale = 0.5
        self.len1 = 24
        self.len2 = 30

    def update(self):
        if self.mode == 'drive':
            self.target_angles = self.neutral.copy()
        
        if not self.calibrating and not self.retrieve:
            for i, angle in enumerate(self.target_angles):
                self.current_angles[i] += min(1 if self.mode == 'retrieve' else 2, max(angle - self.current_angles[i], -1 if self.mode == 'retrieve' else -2))
                self.arm_servos[i].angle = self.current_angles[i]
        
        # for i, servo in enumerate(self.arm_servos):
        #     print(i, self.target_angles[i])
        #     print(i, self.current_angles[i])
        #     print(i, servo.angle)
    
    def calibrate(self):
        self.calibrating = True
        
        for x, servo in enumerate(self.arm_servos):
            for i in range (self.neutral[x], self.ranges[x][1], 3):
                servo.angle = i
                time.sleep(0.05)

            for i in range (self.ranges[x][1], self.ranges[x][0], -3):
                servo.angle = i
                time.sleep(0.05)

            for i in range (self.ranges[x][0], self.neutral[x], 3):
                servo.angle = i
                time.sleep(0.05)
                        
        self.current_angles = self.neutral.copy()
        self.calibrating = False
            
    def switch_mode(self, mode):
        self.mode = mode
        self.target_angles = self.neutral.copy()
        if mode == 'drive':
            self.steering.angle = self.steering_range[1]
            self.motor.angle = self.drive
        elif mode == 'retrieve':
            self.motor.angle = 90
            
    def move_arm(self, rel_x, rel_y):
        if not self.retrieve:
            self.target_angles[0] = -rel_x * 45 + self.current_angles[0]
            
            # --- New code for horizontal plane movement ---

            # We define the target vertical position as a constant offset below the base.
            # This requires a new class attribute, e.g., self.height_offset = 5.0
            y_target = -self.height_offset

            # To make the move relative, we first calculate the current horizontal reach
            # using forward kinematics. This assumes angle[1] is from the horizontal plane
            # and angle[2] is relative to the first arm segment.
            theta1_current_rad = np.deg2rad(self.current_angles[1])
            theta2_current_rad = np.deg2rad(self.current_angles[2])

            # Forward Kinematics to find current horizontal reach (r_current)
            r_current = self.len1 * np.cos(theta1_current_rad) + self.len2 * np.cos(theta1_current_rad + theta2_current_rad)

            # Calculate the target horizontal reach by applying the relative input.
            # This requires a scaling factor, e.g., self.reach_scale = 2.0
            r_target = r_current + rel_y * self.reach_scale

            # Perform inverse kinematics for the target point w(r_target, y_target) in the arm's vertical plane.

            # 1. Calculate the straight-line distance from the shoulder pivot to the target.
            distance = np.sqrt(r_target**2 + y_target**2)

            # 2. Check for reachability and clip the target distance if it's outside the workspace.
            max_reach = self.len1 + self.len2
            min_reach = np.abs(self.len1 - self.len2)
            
            # Clip the distance to what is physically possible
            clipped_distance = np.clip(distance, min_reach, max_reach)
            if distance != clipped_distance:
                if distance > 0:
                    scale = clipped_distance / distance
                    r_target *= scale
                    # Recalculate the final clipped distance
                    distance = np.sqrt(r_target**2 + y_target**2)
                else:
                    # If target is at the origin, move to minimum possible reach
                    r_target = min_reach
                    distance = np.sqrt(r_target**2 + y_target**2)

            # 3. Calculate the required joint angles using the Law of Cosines, adapted from your original function.
            # Clip arguments to arccos to prevent math domain errors from floating-point inaccuracies.
            
            # Elbow angle (internal angle of the arm triangle)
            cos_theta2_arg = (self.len1**2 + self.len2**2 - distance**2) / (2 * self.len1 * self.len2)
            theta2_rad = np.arccos(np.clip(cos_theta2_arg, -1.0, 1.0))

            # Shoulder angle calculation
            alpha = np.arctan2(y_target, r_target) # Angle to the target point
            cos_beta_arg = (self.len1**2 + distance**2 - self.len2**2) / (2 * self.len1 * distance)
            beta = np.arccos(np.clip(cos_beta_arg, -1.0, 1.0)) # Internal angle at the shoulder

            # The "elbow down" configuration is best for reaching below the base.
            theta1_rad = alpha - beta

            # 4. Convert calculated angles from radians to degrees and update targets.
            self.target_angles[1] = np.rad2deg(theta1_rad)
            self.target_angles[2] = np.rad2deg(theta2_rad)
            
            for i, angle in enumerate(self.target_angles):
                self.target_angles[i] = min(self.ranges[i][1], (max(self.ranges[i][0], angle)))
        else:
            # code to move down and retrieve
            if True:
                self.activate_gripper()
                
    
    def steer(self, rel_x):
        self.steering.angle = self.steering_range[1] - rel_x * (self.steering_range[2] - self.steering_range[0]) / 2
        
    def stop_motor(self):
        self.motor.angle = 90
        
    def activate_gripper(mode='activate'):
        if mode == 'activate':
            self.gripper.angle = self.gripper_range[0]
        elif mode == 'release':
            self.gripper.angle = self.gripper_range[1]

servo_controller = ServoController()
