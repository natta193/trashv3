import cv2
import threading
import time
import numpy as np
import psutil
from gpiozero import CPUTemperature
from servo import servo_controller

# GLOBAL VARIABLES FOR VISION DATA AND THREAD SAFETY
_vision_data = {
    "objects_detected": [],
}
_vision_data_lock = threading.Lock()

# OBJECT LOCKING SYSTEM
lock_on_object = None
lock_on_start_time = None
lock_on_last_seen = None
LOCK_ON_DURATION = 3.0  # SECONDS TO LOCK ON
LOCK_ON_TIMEOUT = 10.0  # SECONDS TO WAIT BEFORE RELEASING LOCK

# TRACKING VARIABLES
object_history = []  # STORE LAST N POSITIONS FOR SMOOTHED CENTROID
confidence_counter = 0  # COUNT CONSECUTIVE FRAMES OBJECT IS SEEN
MAX_HISTORY = 5  # MAXIMUM POSITIONS TO STORE FOR SMOOTHING

# TEMPLATE UPDATE VARIABLES
template_last_update = 0.0  # TIME OF LAST TEMPLATE UPDATE
TEMPLATE_UPDATE_INTERVAL = 2.0  # SECONDS BETWEEN TEMPLATE UPDATES
TEMPLATE_UPDATE_THRESHOLD = 0.8  # MINIMUM SIMILARITY FOR TEMPLATE UPDATE

# TEMPLATE STORAGE
locked_object_template = None
locked_object_bbox = None

# CAMERA SETUP
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("ERROR: Could not open video stream. Check camera connection or index.")
    exit()

# STATS TRACKING
_stats = {
    "fps": 0.0,
    "cpu": 0.0,
    "cpu_temp": 0.0
}
_last_time = time.time()
_frame_count = 0

def detect_objects(frame, search_region=None):
    """
    OPTIMIZED ROBUST OBJECT DETECTION FOR BETTER PERFORMANCE.
    BALANCES ACCURACY WITH SPEED.
    """
    try:
        # CONVERT TO HSV AND GRAYSCALE
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. MULTI-STRATEGY DETECTION APPROACH
        detected_objects = []
        
        # STRATEGY 1: ADAPTIVE COLOR DETECTION
        # CALCULATE BACKGROUND STATISTICS FOR ADAPTIVE THRESHOLDING
        bg_mean = np.mean(hsv[:, :, 2])  # VALUE CHANNEL MEAN
        bg_std = np.std(hsv[:, :, 2])    # VALUE CHANNEL STD
        
        # ADAPTIVE WHITE/BRIGHT DETECTION
        white_thresh_low = max(0, bg_mean + 1.5 * bg_std)
        white_thresh_high = 255
        
        white_mask = cv2.inRange(hsv, np.array([0, 0, int(white_thresh_low)]), 
                                np.array([180, 50, white_thresh_high]))
        
        # ADAPTIVE BRIGHT DETECTION
        bright_thresh_low = max(0, bg_mean + 1.0 * bg_std)
        bright_mask = cv2.inRange(hsv, np.array([0, 40, int(bright_thresh_low)]), 
                                 np.array([180, 255, 255]))
        
        # STRATEGY 2: OPTIMIZED EDGE DETECTION
        # SINGLE-SCALE FOR SPEED
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 80)
        
        # STRATEGY 3: FAST MASK COMBINATION
        color_mask = cv2.bitwise_or(white_mask, bright_mask)
        
        # SIMPLE WEIGHTED COMBINATION (FASTER)
        final_mask = cv2.addWeighted(color_mask, 0.7, edges, 0.3, 0)
        final_mask = (final_mask > 127).astype(np.uint8) * 255
        
        # STRATEGY 4: FAST MORPHOLOGICAL CLEANUP
        # SINGLE KERNEL FOR BOTH OPERATIONS
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # 6. FIND CONTOURS WITH IMPROVED FILTERING
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # BASIC AREA FILTER
            area = cv2.contourArea(contour)
            if area < 150 or area > 30000:  # REASONABLE SIZE RANGE
                continue
            
            # GET BOUNDING BOX
            x, y, w, h = cv2.boundingRect(contour)
            
            # IMPROVED ASPECT RATIO FILTER
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 8.0:  # REASONABLE ASPECT RATIOS
                continue
            
            # MINIMUM SIZE FILTER
            if w < 15 or h < 15:  # MINIMUM DETECTABLE SIZE
                continue
            
            # CALCULATE CENTROID
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                center_x = x + w // 2
                center_y = y + h // 2
            
            # APPLY SEARCH REGION FILTER IF SPECIFIED
            if search_region is not None:
                start_y, end_y = search_region
                if center_y < start_y or center_y > end_y:
                    continue
            
            # OPTIMIZED STRATEGY 5: FAST CONFIDENCE CALCULATION
            confidence = 0.0
            
            # COLOR CONFIDENCE (FAST)
            color_confidence = 0.0
            if color_mask[y:y+h, x:x+w].size > 0:
                color_ratio = np.sum(color_mask[y:y+h, x:x+w] > 0) / (w * h)
                color_confidence = color_ratio
            
            # EDGE CONFIDENCE (FAST)
            edge_confidence = 0.0
            if edges[y:y+h, x:x+w].size > 0:
                edge_density = np.sum(edges[y:y+h, x:x+w]) / (w * h)
                edge_confidence = min(edge_density / 100, 1.0)
            
            # SHAPE CONFIDENCE (FAST)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # SIMPLIFIED CONFIDENCE SCORE (FASTER)
            confidence = (
                color_confidence * 0.5 +      # COLOR IS MOST IMPORTANT
                edge_confidence * 0.3 +       # EDGES HELP
                solidity * 0.2               # SHAPE MATTERS
            )
            
            # CONFIDENCE THRESHOLD
            if confidence < 0.3:  # OPTIMIZED THRESHOLD
                continue
            
            # EDGE MARGIN FILTER
            margin = 15
            if x < margin or y < margin or x + w > frame.shape[1] - margin or y + h > frame.shape[0] - margin:
                continue
            
            # CREATE OBJECT DICTIONARY
            obj = {
                'bbox': [x, y, x + w, y + h],
                'center_x': center_x,
                'center_y': center_y,
                'area': area,
                'confidence': confidence,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'color_confidence': color_confidence,
                'edge_confidence': edge_confidence
            }
            
            detected_objects.append(obj)
        
        # SORT BY CONFIDENCE (HIGHEST FIRST)
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        # RETURN TOP 5 OBJECTS FOR SPEED
        return detected_objects[:5]
        
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        return []

def get_stats():
    """RETURN SYSTEM STATS"""
    return _stats.copy()

def _update_stats():
    """UPDATE STATS EVERY SECOND"""
    global _last_time, _frame_count, _stats
    _frame_count += 1
    now = time.time()
    elapsed = now - _last_time
    if elapsed >= 1.0:
        _stats["fps"] = _frame_count / elapsed
        _stats["cpu"] = psutil.cpu_percent()
        try:
            _stats["cpu_temp"] = CPUTemperature().temperature
        except Exception:
            _stats["cpu_temp"] = 0.0
        _last_time = now
        _frame_count = 0

def update_object_tracking(matched_obj):
    """UPDATE OBJECT TRACKING WITH SMOOTHED CENTROID AND CONFIDENCE LOGIC"""
    global object_history, confidence_counter
    
    if matched_obj is not None:
        # ADD CURRENT POSITION TO HISTORY
        current_pos = (matched_obj['center_x'], matched_obj['center_y'])
        object_history.append(current_pos)
        
        # KEEP ONLY LAST N POSITIONS
        if len(object_history) > MAX_HISTORY:
            object_history.pop(0)
        
        # INCREMENT CONFIDENCE COUNTER
        confidence_counter += 1
        
        # UPDATE LOCKED OBJECT WITH SMOOTHED CENTROID
        if len(object_history) >= 3:  # NEED AT LEAST 3 POSITIONS FOR SMOOTHING
            # CALCULATE SMOOTHED CENTROID (EXPONENTIAL MOVING AVERAGE)
            smoothed_x = sum(pos[0] for pos in object_history[-3:]) / 3
            smoothed_y = sum(pos[1] for pos in object_history[-3:]) / 3
            
            # UPDATE LOCKED OBJECT WITH SMOOTHED POSITION
            lock_on_object['center_x'] = int(smoothed_x)
            lock_on_object['center_y'] = int(smoothed_y)
            
            # UPDATE BOUNDING BOX TO MATCH NEW CENTER
            w = lock_on_object['bbox'][2] - lock_on_object['bbox'][0]
            h = lock_on_object['bbox'][3] - lock_on_object['bbox'][1]
            lock_on_object['bbox'][0] = int(smoothed_x - w/2)
            lock_on_object['bbox'][1] = int(smoothed_y - h/2)
            lock_on_object['bbox'][2] = int(smoothed_x + w/2)
            lock_on_object['bbox'][3] = int(smoothed_y + h/2)
    else:
        # OBJECT NOT SEEN - DECREMENT CONFIDENCE
        confidence_counter = max(0, confidence_counter - 1)

def find_best_match_for_tracking(detected_objects, last_known_pos):
    """FIND BEST MATCHING OBJECT FOR TRACKING USING MULTIPLE CRITERIA"""
    if not detected_objects or last_known_pos is None:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for obj in detected_objects:
        # CALCULATE DISTANCE FROM LAST KNOWN POSITION
        distance = ((obj['center_x'] - last_known_pos[0])**2 + 
                   (obj['center_y'] - last_known_pos[1])**2)**0.5
        
        # CALCULATE SIZE SIMILARITY
        size_ratio = min(obj['area'], lock_on_object['area']) / max(obj['area'], lock_on_object['area'])
        
        # CALCULATE CONFIDENCE SCORE
        confidence_score = obj['confidence']
        
        # COMBINED SCORE (LOWER DISTANCE = HIGHER SCORE, HIGHER CONFIDENCE = HIGHER SCORE)
        distance_score = max(0, 1.0 - (distance / 200.0))  # NORMALIZE TO 0-1
        combined_score = (distance_score * 0.6 + size_ratio * 0.2 + confidence_score * 0.2)
        
        if combined_score > best_score and distance < 200:  # MAXIMUM DISTANCE THRESHOLD
            best_score = combined_score
            best_match = obj
    
    return best_match, best_score

def reset_tracking():
    """RESET ALL TRACKING VARIABLES BUT PRESERVE TEMPLATE AND BBOX"""
    global object_history, confidence_counter
    object_history.clear()
    confidence_counter = 0
    # NOTE: We do NOT clear locked_object_template or locked_object_bbox here
    # as they are needed for continued tracking across mode switches

def update_template_if_good_match(frame, matched_pos, similarity):
    """
    UPDATE TEMPLATE IMAGE IF SIMILARITY IS HIGH ENOUGH AND ENOUGH TIME HAS PASSED.
    """
    global locked_object_template, locked_object_bbox, template_last_update
    
    # SAFETY CHECK: ENSURE BBOX EXISTS
    if locked_object_bbox is None:
        return False
    
    if (similarity >= TEMPLATE_UPDATE_THRESHOLD and 
        time.time() - template_last_update >= TEMPLATE_UPDATE_INTERVAL):
        
        # EXTRACT NEW TEMPLATE FROM CURRENT POSITION
        w = locked_object_bbox[2] - locked_object_bbox[0]
        h = locked_object_bbox[3] - locked_object_bbox[1]
        
        x1 = max(0, matched_pos[0] - w//2)
        y1 = max(0, matched_pos[1] - h//2)
        x2 = min(frame.shape[1], matched_pos[0] + w//2)
        y2 = min(frame.shape[0], matched_pos[1] + h//2)
        
        if x2 > x1 and y2 > y1:
            new_template = frame[y1:y2, x1:x2].copy()
            if new_template.size > 0:
                locked_object_template = new_template
                locked_object_bbox = [x1, y1, x2, y2]
                template_last_update = time.time()
                # RESET SIMILARITY TO 1.0 FOR FRESH TEMPLATE
                if lock_on_object is not None:
                    lock_on_object['last_similarity'] = 1.0
                print(f"TEMPLATE UPDATED - New size: {w}x{h}")
                return True
    
    return False

def find_object_by_template(frame, template, bbox):
    """
    LIGHTWEIGHT TEMPLATE MATCHING TO FIND OBJECT IN FRAME.
    RETURNS (center_x, center_y, similarity) OR None IF NOT FOUND.
    """
    try:
        if template is None or template.size == 0:
            return None
            
        # GET LAST KNOWN POSITION FROM BBOX
        last_x = (bbox[0] + bbox[2]) // 2
        last_y = (bbox[1] + bbox[3]) // 2
        
        # DEFINE SEARCH REGION AROUND LAST POSITION
        # USE LARGE SEARCH RADIUS TO ALLOW COMPLETE SCREEN TRACKING
        search_radius_x = frame.shape[1] // 2  # HALF FRAME WIDTH
        search_radius_y = frame.shape[0] // 2  # HALF FRAME HEIGHT
        
        x1 = max(0, last_x - search_radius_x)
        y1 = max(0, last_y - search_radius_y)
        x2 = min(frame.shape[1], last_x + search_radius_x)
        y2 = min(frame.shape[0], last_y + search_radius_y)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # EXTRACT SEARCH REGION
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            return None
        
        # TEMPLATE MATCHING
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # USE CORRELATION COEFFICIENT AS SIMILARITY SCORE
        similarity = max_val
        
        if similarity > 0.3:  # THRESHOLD FOR ACCEPTABLE MATCH
            # CALCULATE ACTUAL POSITION IN FRAME
            match_x = x1 + max_loc[0] + template.shape[1] // 2
            match_y = y1 + max_loc[1] + template.shape[0] // 2
            
            return (match_x, match_y, similarity)
        
        # IF NO MATCH IN SEARCH REGION, TRY FULL FRAME SEARCH AS FALLBACK
        # This ensures we don't lose the object when it moves far from last position
        print("FALLBACK: Full frame search for lost object")
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        similarity = max_val
        
        if similarity > 0.25:  # SLIGHTLY LOWER THRESHOLD FOR FALLBACK
            # CALCULATE ACTUAL POSITION IN FRAME
            match_x = max_loc[0] + template.shape[1] // 2
            match_y = max_loc[1] + template.shape[0] // 2
            
            return (match_x, match_y, similarity)
        
        return None
        
    except Exception as e:
        print(f"Error in template matching: {e}")
        return None

def process_debug_frames(frame_width, frame_height):
    """
    OPTIMIZED DEBUG FRAME PROCESSING FUNCTION WITH PERFORMANCE IMPROVEMENTS.
    SHOWS THE DETECTION MASKS AND FILTERS THAT THE AI IS ACTUALLY USING.
    """
    # PERFORMANCE OPTIMIZATION: MAXIMUM FRAME RATE
    # REMOVED ARTIFICIAL FPS LIMITING TO USE FULL CPU POWER
    
    while True:
        now = time.time()  # GET CURRENT TIME FOR TIMING LOGIC
        success, frame = camera.read()
        if not success:
            print("ERROR: Failed to read debug frame from camera.")
            break

        # RESIZE FRAME TO SPECIFIED DIMENSIONS
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # PERFORMANCE OPTIMIZATION: REDUCE RESOLUTION FOR DEBUG VIEW
        debug_width = frame_width // 2  # LARGER FOR BETTER VISIBILITY
        debug_height = frame_height // 2
        debug_frame_small = cv2.resize(frame, (debug_width, debug_height))
        
        # CREATE DEBUG VISUALIZATION ON SMALLER FRAME
        debug_frame = create_debug_visualization(debug_frame_small)
        
        # ENCODE FRAME FOR STREAMING
        ret, buffer = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])  # VERY LOW QUALITY FOR MAXIMUM SPEED
        debug_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + debug_frame + b'\r\n')

def create_debug_visualization(frame):
    """
    SIMPLE AND FAST DEBUG VISUALIZATION SHOWING KEY DETECTION MASKS.
    """
    try:
        # CONVERT TO HSV AND GRAYSCALE
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ULTRA-FAST DEBUG VISUALIZATION - ONLY PROCESS WHAT WE NEED
        # SKIP INTERMEDIATE MASKS FOR MAXIMUM PERFORMANCE
        
        # FAST FINAL MASK GENERATION (SIMPLIFIED)
        bg_mean = np.mean(hsv[:, :, 2])
        bg_std = np.std(hsv[:, :, 2])
        
        # SIMPLE BRIGHT OBJECT DETECTION
        bright_thresh = max(0, bg_mean + 1.2 * bg_std)
        final_mask = cv2.inRange(hsv, np.array([0, 30, int(bright_thresh)]), 
                                np.array([180, 255, 255]))
        
        # MINIMAL MORPHOLOGICAL CLEANUP
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # CREATE VERTICAL STACK DEBUG LAYOUT: ORIGINAL ABOVE FINAL MASK
        # CONVERT FINAL MASK TO BGR FOR DISPLAY
        final_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        
        # CREATE VERTICAL STACK: ORIGINAL | FINAL MASK
        debug_frame = np.vstack([frame, final_mask_bgr])
        
        # ADD SIMPLE LABELS
        cv2.putText(debug_frame, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_frame, "FINAL MASK", (10, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return debug_frame
        
    except Exception as e:
        print(f"Error in debug visualization: {e}")
        return frame

def process_frames(frame_width, frame_height):
    """
    OPTIMIZED FRAME PROCESSING FUNCTION WITH PERFORMANCE IMPROVEMENTS.
    USES detect_objects ONLY WHEN SEARCHING, THEN SWITCHES TO LIGHTWEIGHT IMAGE MATCHING.
    """
    global lock_on_object, lock_on_start_time, lock_on_last_seen, locked_object_template, locked_object_bbox, template_last_update
    
    # TRACK CANDIDATE OBJECT FOR LOCK-ON
    candidate_for_lock = None
    
    # PERFORMANCE OPTIMIZATION: MAXIMUM FRAME RATE
    # REMOVED ARTIFICIAL FPS LIMITING TO USE FULL CPU POWER
    
    while True:
        now = time.time()  # GET CURRENT TIME FOR TIMING LOGIC
        success, frame = camera.read()
        if not success:
            print("ERROR: Failed to read frame from camera.")
            break

        # RESIZE FRAME TO SPECIFIED DIMENSIONS
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # IMAGE CROPPING LOGIC: CROP TO TOP 3/4 WHEN IN DRIVE MODE AND UNLOCKED
        original_frame = frame.copy()
        if servo_controller.mode == 'drive' and lock_on_object is None:
            # CROP TO TOP 3/4 OF SCREEN WHEN SEARCHING
            crop_height = int(frame_height * 0.75)
            frame = frame[:crop_height, :]
            # ADJUST FRAME HEIGHT FOR CROPPED IMAGE
            current_frame_height = crop_height
        else:
            # USE FULL FRAME FOR ALL OTHER SITUATIONS
            current_frame_height = frame_height
        
        # STATE MACHINE LOGIC
        if servo_controller.mode == 'drive':
            # DRIVE MODE: LOOK FOR OBJECTS TO LOCK ONTO
            if lock_on_object is None:
                # NOT LOCKED - USE detect_objects TO FIND CANDIDATES
                search_region = (0, int(current_frame_height * 0.67))  # TOP 2/3 OF CROPPED SCREEN
                detected_objects = detect_objects(frame, search_region)
                best_obj = detected_objects[0] if detected_objects else None
                
                if best_obj is not None:
                    if candidate_for_lock is not None:
                        # CHECK IF SAME OBJECT
                        distance = ((best_obj['center_x'] - candidate_for_lock['center_x'])**2 + 
                                   (best_obj['center_y'] - candidate_for_lock['center_y'])**2)**0.5
                        
                        if distance < 100 and abs(best_obj['confidence'] - candidate_for_lock['confidence']) < 0.3:
                            # SAME OBJECT - CHECK STABILITY TIMER
                            if now - lock_on_start_time >= LOCK_ON_DURATION:
                                lock_on_object = best_obj.copy()
                                lock_on_last_seen = now
                                print(f"LOCKED onto object with confidence {best_obj['confidence']:.3f}")
                                
                                # EXTRACT TEMPLATE IMAGE OF LOCKED OBJECT
                                x1, y1, x2, y2 = best_obj['bbox']
                                locked_object_template = frame[y1:y2, x1:x2].copy()
                                locked_object_bbox = best_obj['bbox'].copy()
                                
                                # SET INITIAL SIMILARITY SCORE
                                lock_on_object['last_similarity'] = 1.0
                                
                                candidate_for_lock = None
                                lock_on_start_time = None
                                reset_tracking()
                                print("TEMPLATE EXTRACTED - SWITCHING TO IMAGE MATCHING")
                        else:
                            # DIFFERENT OBJECT - START TRACKING NEW CANDIDATE
                            candidate_for_lock = best_obj.copy()
                            lock_on_start_time = now
                    else:
                        # NO PREVIOUS CANDIDATE - START TRACKING THIS ONE
                        candidate_for_lock = best_obj.copy()
                        lock_on_start_time = now
                else:
                    # NO OBJECTS - RESET CANDIDATE TRACKING
                    candidate_for_lock = None
                    lock_on_start_time = None
                
                # NO LOCKED OBJECT - STEER STRAIGHT
                servo_controller.steer(0.0)
                
                # ALSO CALL MOVE_ARM WITH NEUTRAL POSITION TO KEEP ARM STABLE
                servo_controller.move_arm(0.0, 0.0, 0.01)
                
            else:
                # LOCKED ON - USE LIGHTWEIGHT IMAGE MATCHING
                if locked_object_template is not None:
                    # FIND OBJECT USING TEMPLATE MATCHING
                    match_result = find_object_by_template(frame, locked_object_template, locked_object_bbox)
                    
                    if match_result is not None:
                        matched_pos = (match_result[0], match_result[1])
                        similarity = match_result[2]
                        
                        # UPDATE LOCKED OBJECT POSITION
                        lock_on_object['center_x'] = matched_pos[0]
                        lock_on_object['center_y'] = matched_pos[1]
                        
                        # STORE SIMILARITY SCORE FOR DISPLAY
                        lock_on_object['last_similarity'] = similarity
                        
                        # ONLY RESET TIMEOUT IF SIMILARITY IS GOOD ENOUGH
                        if similarity >= 0.7:  # GOOD MATCH THRESHOLD
                            lock_on_last_seen = now
                        else:
                            # LOW SIMILARITY - START TIMEOUT IF NOT ALREADY STARTED
                            if lock_on_last_seen is None:
                                lock_on_last_seen = now
                        
                        # UPDATE BOUNDING BOX (WITH SAFETY CHECK)
                        if locked_object_bbox is not None:
                            w = locked_object_bbox[2] - locked_object_bbox[0]
                            h = locked_object_bbox[3] - locked_object_bbox[1]
                            lock_on_object['bbox'] = [matched_pos[0] - w//2, matched_pos[1] - h//2,
                                                     matched_pos[0] + w//2, matched_pos[1] + h//2]
                            
                            # UPDATE TEMPLATE IF GOOD MATCH
                            update_template_if_good_match(frame, matched_pos, similarity)
                        else:
                            print("WARNING: locked_object_bbox is None during template matching")
                            # CREATE A DEFAULT BBOX BASED ON TEMPLATE SIZE
                            if locked_object_template is not None:
                                h, w = locked_object_template.shape[:2]
                                lock_on_object['bbox'] = [matched_pos[0] - w//2, matched_pos[1] - h//2,
                                                         matched_pos[0] + w//2, matched_pos[1] + h//2]
                                # RECREATE THE BBOX
                                locked_object_bbox = lock_on_object['bbox'].copy()
                                print(f"RECREATED BBOX FROM TEMPLATE SIZE: {w}x{h} at ({matched_pos[0]}, {matched_pos[1]})")
                            else:
                                print("ERROR: Cannot recreate bbox - no template available")
                        
                        # CALCULATE RELATIVE X POSITION FOR STEERING
                        rel_x = (matched_pos[0] - frame_width/2) / (frame_width/2)
                        rel_x = max(-1.0, min(1.0, rel_x))
                        
                        # CHECK IF OBJECT IS FAR ENOUGH DOWN SCREEN TO SWITCH TO RETRIEVE (3/4 DOWN)
                        if matched_pos[1] > current_frame_height * 0.75:
                            # ENSURE WE HAVE VALID BBOX BEFORE SWITCHING TO RETRIEVE
                            if locked_object_bbox is None and locked_object_template is not None:
                                # RECREATE BBOX FROM TEMPLATE SIZE
                                h, w = locked_object_template.shape[:2]
                                locked_object_bbox = [matched_pos[0] - w//2, matched_pos[1] - h//2,
                                                     matched_pos[0] + w//2, matched_pos[1] + h//2]
                                print("RECREATED BBOX BEFORE SWITCHING TO RETRIEVE MODE")
                            
                            servo_controller.switch_mode('retrieve')
                            print("SWITCHING TO RETRIEVE MODE")
                            reset_tracking()
                        else:
                            servo_controller.steer(rel_x)
                            servo_controller.move_arm(0.0, 0.0, 0.01)  # KEEP ARM STABLE
                    else:
                        # OBJECT NOT FOUND - USE LAST KNOWN POSITION
                        rel_x = (lock_on_object['center_x'] - frame_width/2) / (frame_width/2)
                        rel_x = max(-1.0, min(1.0, rel_x))
                        servo_controller.steer(rel_x)
                        servo_controller.move_arm(0.0, 0.0, 0.01)  # KEEP ARM STABLE

                if now - lock_on_last_seen > LOCK_ON_TIMEOUT/4:
                    lock_on_template = None
                    lock_on_object = None

                
        elif servo_controller.mode == 'retrieve':
            # RETRIEVE MODE: USE LIGHTWEIGHT IMAGE MATCHING FOR ARM CONTROL
            if lock_on_object is not None and locked_object_template is not None:
                # FIND OBJECT USING TEMPLATE MATCHING
                match_result = find_object_by_template(frame, locked_object_template, locked_object_bbox)
                
                if match_result is not None:
                    matched_pos = (match_result[0], match_result[1])
                    similarity = match_result[2]
                    
                    # UPDATE LOCKED OBJECT POSITION
                    lock_on_object['center_x'] = matched_pos[0]
                    lock_on_object['center_y'] = matched_pos[1]
                    
                    # STORE SIMILARITY SCORE FOR DISPLAY
                    lock_on_object['last_similarity'] = similarity
                    
                    # ONLY RESET TIMEOUT IF SIMILARITY IS GOOD ENOUGH
                    if similarity >= 0.9:  # GOOD MATCH THRESHOLD
                        lock_on_last_seen = now
                    else:
                        # LOW SIMILARITY - START TIMEOUT IF NOT ALREADY STARTED
                        if lock_on_last_seen is None:
                            lock_on_last_seen = now
                    
                    # UPDATE BOUNDING BOX (WITH SAFETY CHECK)
                    if locked_object_bbox is not None:
                        w = locked_object_bbox[2] - locked_object_bbox[0]
                        h = locked_object_bbox[3] - locked_object_bbox[1]
                        lock_on_object['bbox'] = [matched_pos[0] - w//2, matched_pos[1] - h//2,
                                                 matched_pos[0] + w//2, matched_pos[1] + h//2]
                        
                        # UPDATE TEMPLATE IF GOOD MATCH
                        update_template_if_good_match(frame, matched_pos, similarity)
                    else:
                        print("WARNING: locked_object_bbox is None during retrieve mode")
                        # CREATE A DEFAULT BBOX BASED ON TEMPLATE SIZE
                        if locked_object_template is not None:
                            h, w = locked_object_template.shape[:2]
                            lock_on_object['bbox'] = [matched_pos[0] - w//2, matched_pos[1] - h//2,
                                                     matched_pos[0] + w//2, matched_pos[1] + h//2]
                            # RECREATE THE BBOX
                            locked_object_bbox = lock_on_object['bbox'].copy()
                            print(f"RECREATED BBOX FROM TEMPLATE SIZE IN RETRIEVE MODE: {w}x{h} at ({matched_pos[0]}, {matched_pos[1]})")
                    
                    # CALCULATE RELATIVE POSITIONS AND AREA FOR ARM CONTROL
                    rel_x = (matched_pos[0] - frame_width/2) / (frame_width/2)
                    rel_y = (matched_pos[1] - frame_height/2) / (frame_height/2)
                    
                    # SAFETY CHECK FOR AREA CALCULATION
                    if locked_object_bbox is not None:
                        w = locked_object_bbox[2] - locked_object_bbox[0]
                        h = locked_object_bbox[3] - locked_object_bbox[1]
                        area_ratio = (w * h) / (frame_width * frame_height)
                    else:
                        # FALLBACK AREA RATIO
                        area_ratio = 0.01  # SMALL DEFAULT VALUE
                        print("WARNING: Using fallback area ratio due to missing bbox")
                    
                    rel_x = max(-1.0, min(1.0, rel_x))
                    rel_y = max(-1.0, min(1.0, rel_y))
                    
                    servo_controller.move_arm(rel_x, rel_y, area_ratio)
                    
                if similarity < 0.8:    
                    # OBJECT NOT FOUND - CHECK TIMEOUT
                    if lock_on_last_seen is None:
                        lock_on_last_seen = now
                    elif now - lock_on_last_seen > LOCK_ON_TIMEOUT:
                        # TIMEOUT EXCEEDED - SWITCH BACK TO DRIVE MODE
                        print(f"OBJECT LOST FOR {LOCK_ON_TIMEOUT} SECONDS - SWITCHING TO DRIVE MODE")
                        servo_controller.switch_mode('drive')
                        lock_on_object = None
                        locked_object_template = None
                        locked_object_bbox = None
                        lock_on_last_seen = None
                        candidate_for_lock = None
                        lock_on_start_time = None
                        reset_tracking()
                    elif now - lock_on_last_seen > LOCK_ON_TIMEOUT/2:
                        # AFTER HALF TIMEOUT - HOLD ARM POSITION
                        servo_controller.move_arm(0.0, 0.0, 0.01)  # SMALL AREA RATIO TO HOLD POSITION
        
        # UPDATE GLOBAL VISION DATA
        with _vision_data_lock:
            if lock_on_object is not None:
                _vision_data["objects_detected"] = [lock_on_object]
            else:
                _vision_data["objects_detected"] = []
            
        _update_stats()
        
        # UPDATE SERVO CONTROLLER EVERY ITERATION
        servo_controller.update()

        # DRAW OBJECTS ON FRAME
        if lock_on_object is not None:
            # SHOW LOCKED OBJECT IN RED
            obj = lock_on_object
            (startX, startY, endX, endY) = obj['bbox']
            
            color = (0, 0, 255)  # RED
            # WHEN USING TEMPLATE TRACKING, SHOW CURRENT SIMILARITY SCORE INSTEAD OF OLD CONFIDENCE
            if 'last_similarity' in obj:
                label = f"LOCKED: {obj['last_similarity']:.3f}"
            else:
                label = f"LOCKED: {obj.get('confidence', 0.8):.3f}"
            
            # DRAW BOUNDING BOX
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # DRAW LABEL
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            y = startY - 15 if startY - 15 > label_size[1] else startY + label_size[1] + 10
            
            cv2.rectangle(frame, (startX, y - label_size[1] - 5), 
                         (startX + label_size[0], y + 5), color, -1)
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # DRAW CENTER POINT
            cv2.circle(frame, (obj['center_x'], obj['center_y']), 5, color, -1)
            
            # SHOW TEMPLATE MATCHING STATUS AND SIMILARITY
            if locked_object_template is not None:
                cv2.putText(frame, "TEMPLATE TRACKING", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # SHOW SIMILARITY SCORE
                if 'last_similarity' in lock_on_object:
                    sim_text = f"SIMILARITY: {lock_on_object['last_similarity']:.3f}"
                    cv2.putText(frame, sim_text, (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # SHOW TEMPLATE UPDATE STATUS
                if template_last_update > 0:
                    time_since_update = time.time() - template_last_update
                    update_text = f"TEMPLATE: {time_since_update:.1f}s ago"
                    # SHOW WHEN NEXT UPDATE IS DUE
                    if time_since_update >= TEMPLATE_UPDATE_INTERVAL:
                        update_text += " (READY TO UPDATE)"
                    else:
                        time_until_update = TEMPLATE_UPDATE_INTERVAL - time_since_update
                        update_text += f" (UPDATE IN {time_until_update:.1f}s)"
                else:
                    update_text = "TEMPLATE: Never updated"
                cv2.putText(frame, update_text, (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # SHOW TIMEOUT STATUS IN RETRIEVE MODE
                if servo_controller.mode == 'retrieve' and lock_on_last_seen is not None:
                    time_since_seen = now - lock_on_last_seen
                    if time_since_seen > LOCK_ON_TIMEOUT:
                        timeout_text = f"TIMEOUT: {time_since_seen:.1f}s (SWITCHING TO DRIVE)"
                        cv2.putText(frame, timeout_text, (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif time_since_seen > LOCK_ON_TIMEOUT/2:
                        timeout_text = f"TIMEOUT: {time_since_seen:.1f}s (HOLDING ARM)"
                        cv2.putText(frame, timeout_text, (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    else:
                        timeout_text = f"TIMEOUT: {time_since_seen:.1f}s (USING LAST POS)"
                        cv2.putText(frame, timeout_text, (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            # SHOW ALL DETECTED OBJECTS IN GREEN WHEN NOT LOCKED
            if servo_controller.mode == 'drive':
                search_region = (0, int(current_frame_height * 0.67))
                detected_objects = detect_objects(frame, search_region)
                
                for obj in detected_objects:
                    (startX, startY, endX, endY) = obj['bbox']
                    confidence = obj['confidence']
                    
                    color = (0, 255, 0)  # GREEN
                    label = f"TRASH: {confidence:.3f}"
                    
                    # DRAW BOUNDING BOX
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    
                    # DRAW LABEL
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    y = startY - 15 if startY - 15 > label_size[1] else startY + label_size[1] + 10
                    
                    cv2.rectangle(frame, (startX, y - label_size[1] - 5), 
                                 (startX + label_size[0], y + 5), color, -1)
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # DRAW CENTER POINT
                    cv2.circle(frame, (obj['center_x'], obj['center_y']), 5, color, -1)

        # DRAW MODE STATUS
        mode_text = f"MODE: {servo_controller.mode.upper()}"
        cv2.putText(frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # DRAW MODE SWITCH LINE WHEN LOCKED ON IN DRIVE MODE
        if servo_controller.mode == 'drive' and lock_on_object is not None:
            switch_y = int(current_frame_height * 0.75)
            cv2.line(frame, (0, switch_y), (frame_width, switch_y), (0, 255, 255), 2)
            cv2.putText(frame, "MODE SWITCH LINE", (10, switch_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # DRAW LOCK STATUS
        if lock_on_object is not None:
            lock_text = "LOCKED ON OBJECT (TEMPLATE TRACKING)"
            cv2.putText(frame, lock_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif candidate_for_lock is not None and lock_on_start_time is not None:
            time_remaining = LOCK_ON_DURATION - (now - lock_on_start_time)
            if time_remaining > 0:
                tracking_text = f"TRACKING: {time_remaining:.1f}s"
                cv2.putText(frame, tracking_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ENCODE FRAME FOR STREAMING (OPTIMIZED)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])  # EVEN LOWER QUALITY FOR MAXIMUM SPEED
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def release_camera():
    """RELEASE CAMERA RESOURCES"""
    print("Releasing camera resources.")
    camera.release()
    cv2.destroyAllWindows()

# REGISTER CLEANUP FUNCTION
import atexit
atexit.register(release_camera)
