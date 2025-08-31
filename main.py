from adafruit_servokit import ServoKit

def startup():
    kit = ServoKit(channels=16)

    for i in range(3):
        kit.servo[i].angle = 90

    # startup
    input("Power disconnected?")
    kit.servo[4].angle = 90 
    print("Starting up motor...")
    input("Power connected?")
    
startup()

from flask import Flask, Response, render_template, jsonify
import threading
import sys
import vision
from servo import servo_controller
import logging
import time

app = Flask(__name__)
frame_width = 480
frame_height = 320
ready = False
feed_lock = threading.Lock()

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def run():
    global frame_width, frame_height
    while True:
        if ready:
            servo_controller.update()
            time.sleep(0.05)
                    
@app.route('/')
def index():
    """Video streaming home page."""
    with feed_lock:
        global ready
        ready = True
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Supplies video frames to the browser."""
    with feed_lock:
        global ready
        ready = True
    return Response(vision.process_frames(frame_width, frame_height),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/debug_feed')
def debug_feed():
    """Debug video streaming route. Shows what the AI is actually seeing."""
    with feed_lock:
        global ready
        ready = True
    return Response(vision.process_debug_frames(frame_width, frame_height),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibrate')
def calibrate():
    """Endpoint to trigger servo calibration."""
    try:
        servo_controller.calibrate()
        return jsonify({"status": "success", "message": "Calibration completed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stats')
def stats():
    import vision
    return jsonify(vision.get_stats())

@app.route('/start_browser')
def start_browser():
    import server
    server.launch_browser()
    return jsonify({"status": "success", "message": "Browser launched"})

@app.route('/stop_browser')
def stop_browser():
    import server
    server.close_browser()
    return jsonify({"status": "success", "message": "Browser closed"})

@app.route('/motor_stop')
def motor_stop():
    servo_controller.stop_motor()
    return jsonify({"status": "success", "message": "Motor stopped"})

if __name__ == '__main__':
    arm_thread = threading.Thread(target=run, daemon=True)
    arm_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
