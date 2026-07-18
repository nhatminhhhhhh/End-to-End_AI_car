import os
import time
import threading
import queue
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, Response

from config import (
    CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_FLIP,
    DATA_DIR, SERVO_CENTER, SERVO_MIN, SERVO_MAX,
)

try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("[WARN] Jetson.GPIO not available — using simulation")
    GPIO = None

MOTOR_PWM_PIN = 32
SERVO_PWM_PIN = 33
IN1_PIN = 11
IN2_PIN = 13

PWM_MOTOR_FREQ = 1000
PWM_SERVO_FREQ = 50
SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5
DEADZONE_PERCENT = 60
PWM_MIN = DEADZONE_PERCENT
PWM_MAX = 100

if GPIO:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(MOTOR_PWM_PIN, GPIO.OUT)
    GPIO.setup(SERVO_PWM_PIN, GPIO.OUT)
    GPIO.setup(IN1_PIN, GPIO.OUT)
    GPIO.setup(IN2_PIN, GPIO.OUT)
    motor_pwm = GPIO.PWM(MOTOR_PWM_PIN, PWM_MOTOR_FREQ)
    servo_pwm = GPIO.PWM(SERVO_PWM_PIN, PWM_SERVO_FREQ)
    motor_pwm.start(0)
    servo_pwm.start(0)
else:
    motor_pwm = servo_pwm = None

current_steering = SERVO_CENTER
current_throttle = 0
recording = False
frame_count = 0
os.makedirs(DATA_DIR, exist_ok=True)

frame_queue = queue.Queue(maxsize=2)
camera_ready = threading.Event()


def gstreamer_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
        % (CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT, CAMERA_FPS,
           CAMERA_FLIP, CAMERA_WIDTH, CAMERA_HEIGHT)
    )


def camera_loop():
    global frame_queue, camera_ready
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[Camera] Failed to open camera")
        return
    camera_ready.set()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)


def set_motor(speed):
    global current_throttle
    current_throttle = speed
    if GPIO is None:
        return
    if speed == 0:
        GPIO.output(IN1_PIN, GPIO.LOW)
        GPIO.output(IN2_PIN, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(0)
        return
    sign = 1 if speed > 0 else -1
    abs_speed = abs(speed)
    mapped = PWM_MIN + (abs_speed / 100.0) * (PWM_MAX - PWM_MIN)
    mapped = max(PWM_MIN, min(PWM_MAX, mapped))
    speed_out = sign * mapped
    if speed_out > 0:
        GPIO.output(IN1_PIN, GPIO.HIGH)
        GPIO.output(IN2_PIN, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(speed_out)
    else:
        GPIO.output(IN1_PIN, GPIO.LOW)
        GPIO.output(IN2_PIN, GPIO.HIGH)
        motor_pwm.ChangeDutyCycle(-speed_out)


def set_servo(angle):
    global current_steering
    angle = max(SERVO_MIN, min(SERVO_MAX, angle))
    current_steering = angle
    if GPIO is None:
        return
    duty = SERVO_MIN_DUTY + (angle / 180.0) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)
    servo_pwm.ChangeDutyCycle(duty)


def cleanup():
    if GPIO:
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Collect Data</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; touch-action: none; }
        body {
            background: #222;
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: sans-serif;
            color: #fff;
            min-height: 100vh;
        }
        .container {
            width: 100vw;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 10px;
        }
        #videoFeed {
            width: 100%%;
            max-width: 360px;
            background: #000;
            border-radius: 12px;
            margin-bottom: 10px;
        }
        .controls {
            display: flex;
            flex-direction: row;
            width: 100%%;
            max-width: 400px;
            flex: 1;
        }
        .half { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 10px; }
        .left { background: #1a1a2e; border-radius: 12px 0 0 12px; }
        .right { background: #16213e; border-radius: 0 12px 12px 0; }
        .slider-vertical {
            width: 80px; height: 200px;
            -webkit-appearance: slider-vertical;
            appearance: slider-vertical;
            writing-mode: bt-lr;
            background: #444; border-radius: 20px; outline: none;
            transform: rotate(180deg);
        }
        .slider-vertical::-webkit-slider-thumb {
            -webkit-appearance: none; width: 30px; height: 30px;
            border-radius: 50%%; background: #4CAF50; cursor: pointer;
        }
        .slider-horizontal {
            width: 200px; height: 80px;
            -webkit-appearance: none; appearance: none;
            background: #444; border-radius: 20px; outline: none;
        }
        .slider-horizontal::-webkit-slider-thumb {
            -webkit-appearance: none; width: 30px; height: 30px;
            border-radius: 50%%; background: #FF9800; cursor: pointer;
        }
        .label { margin: 8px 0; font-size: 16px; }
        .label .value { font-weight: bold; }
        .label .value.speed { color: #4CAF50; }
        .label .value.steer { color: #FF9800; }
        .record-bar {
            display: flex; align-items: center; gap: 12px;
            margin: 10px 0; padding: 8px 16px;
            background: #2a2a2a; border-radius: 24px;
        }
        .record-btn {
            width: 20px; height: 20px; border-radius: 50%%;
            border: 3px solid #666; cursor: pointer; transition: 0.2s;
        }
        .record-btn.idle { background: #666; }
        .record-btn.recording { background: #f00; border-color: #f00; animation: pulse 1s infinite; }
        @keyframes pulse { 0%% { opacity: 1; } 50%% { opacity: 0.5; } 100%% { opacity: 1; } }
        .status { font-size: 14px; color: #aaa; }
        .status .count { color: #fff; font-weight: bold; }
        .info { font-size: 12px; color: #666; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <img id="videoFeed" src="/video_feed">
        <div class="record-bar">
            <div class="record-btn" id="recordBtn" onclick="toggleRecord()"></div>
            <span class="status"><span class="count" id="frameCount">0</span> frames</span>
        </div>
        <div class="controls">
            <div class="half left">
                <input type="range" class="slider-vertical" id="speedSlider" min="-100" max="100" value="0" step="1">
                <div class="label">Speed: <span class="value speed" id="speedValue">0</span></div>
            </div>
            <div class="half right">
                <input type="range" class="slider-horizontal" id="servoSlider" min="40" max="140" value="95" step="1">
                <div class="label">Steer: <span class="value steer" id="servoValue">95</span>°</div>
            </div>
        </div>
        <div class="info">Deadzone PWM: %d%% | Servo center: %d°</div>
    </div>
    <script>
        const speedSlider = document.getElementById('speedSlider');
        const servoSlider = document.getElementById('servoSlider');
        const speedValue = document.getElementById('speedValue');
        const servoValue = document.getElementById('servoValue');
        const recordBtn = document.getElementById('recordBtn');
        const frameCount = document.getElementById('frameCount');

        function updateSpeed() {
            const raw = parseInt(speedSlider.value);
            const val = -raw;
            speedValue.innerText = val;
            fetch('/set_speed', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'speed=' + val });
        }
        function updateServo() {
            const val = parseInt(servoSlider.value);
            servoValue.innerText = val;
            fetch('/set_servo', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'angle=' + val });
        }
        function resetSpeed() {
            speedSlider.value = 0;
            speedValue.innerText = '0';
            fetch('/set_speed', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'speed=0' });
        }
        function resetServo() {
            servoSlider.value = 95;
            servoValue.innerText = '95';
            fetch('/set_servo', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'angle=95' });
        }
        function toggleRecord() {
            fetch('/toggle_record', { method: 'POST' }).then(r => r.json()).then(d => {
                recordBtn.className = 'record-btn ' + (d.recording ? 'recording' : 'idle');
            });
        }
        setInterval(() => {
            fetch('/status').then(r => r.json()).then(d => {
                frameCount.innerText = d.frame_count;
                recordBtn.className = 'record-btn ' + (d.recording ? 'recording' : 'idle');
            });
        }, 1000);

        speedSlider.addEventListener('input', updateSpeed);
        servoSlider.addEventListener('input', updateServo);
        speedSlider.addEventListener('mouseup', resetSpeed);
        speedSlider.addEventListener('touchend', resetSpeed);
        servoSlider.addEventListener('mouseup', resetServo);
        servoSlider.addEventListener('touchend', resetServo);

        window.onload = function() { resetSpeed(); resetServo(); };
    </script>
</body>
</html>
""" % (DEADZONE_PERCENT, SERVO_CENTER)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            global recording, frame_count
            if recording:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:21]
                filename = f"{ts}_s{current_steering}_t{current_throttle}.jpg"
                cv2.imwrite(os.path.join(DATA_DIR, filename), frame)
                frame_count += 1
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_speed', methods=['POST'])
def set_speed():
    speed = request.form.get('speed', 0, type=int)
    set_motor(speed)
    return jsonify({'status': 'OK', 'speed': speed})


@app.route('/set_servo', methods=['POST'])
def set_servo_route():
    angle = request.form.get('angle', SERVO_CENTER, type=int)
    set_servo(angle)
    return jsonify({'status': 'OK', 'angle': angle})


@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    global recording
    recording = not recording
    print(f"[Record] {'START' if recording else 'STOP'}")
    return jsonify({'recording': recording})


@app.route('/status')
def status():
    return jsonify({'recording': recording, 'frame_count': frame_count})


if __name__ == '__main__':
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    if not camera_ready.wait(timeout=5):
        print("[ERROR] Camera not ready")
        sys.exit(1)

    set_servo(SERVO_CENTER)
    print("\n" + "=" * 50)
    print("  E2E DATA COLLECTOR")
    print("=" * 50)
    print(f"  Camera: {CAMERA_CAPTURE_WIDTH}x{CAMERA_CAPTURE_HEIGHT} → {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"  Save: {DATA_DIR}")
    print(f"  Format: {{timestamp}}_s{{steering}}_t{{throttle}}.jpg")
    print(f"  Open browser at http://<jetson-ip>:5000")
    print("=" * 50)

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        set_motor(0)
        cleanup()
        print("\n[Cleanup] Done")
