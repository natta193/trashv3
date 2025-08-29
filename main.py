from flask import Flask, Response, render_template, jsonify
import threading
import time
import sys
import vision
from servo import servo_controller

app = Flask(__name__)
frame_width = 480
frame_height = 320
ready = False
feed_lock = threading.Lock()

def run():
    global frame_width, frame_height
    count=0
    while True:
        if ready:
            count+=1
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

if __name__ == '__main__':
    arm_thread = threading.Thread(target=run, daemon=True)
    arm_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
