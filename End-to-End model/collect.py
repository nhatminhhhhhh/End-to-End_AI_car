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


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>Collect Data</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; touch-action: none; }
        html, body {
            width: 100vw; height: 100vh; overflow: hidden;
            background: #111;
            font-family: sans-serif; color: #fff;
        }
        .container {
            width: 100vw; height: 100vh;
            display: flex; flex-direction: column;
        }
        .top-bar {
            height: 7vh; min-height: 40px;
            display: flex; align-items: center;
            padding: 0 4vw; gap: 2vw;
            background: #1a1a1a;
            flex-shrink: 0;
        }
        .record-btn {
            width: 18px; height: 18px; border-radius: 50%;
            border: 3px solid #666;
            flex-shrink: 0;
            transition: 0.2s;
        }
        .record-btn.idle { background: #555; }
        .record-btn.recording { background: #f00; border-color: #f00; animation: pulse .8s infinite; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }
        .frame-count { font-size: 14px; color: #aaa; flex: 1; }
        .frame-count b { color: #fff; }
        .main {
            flex: 1; display: flex;
            align-items: center;
            padding: 1.5vh 1.5vw;
            gap: 1.5vw;
            min-height: 0;
        }
        .left-panel {
            width: 14vw; max-width: 100px;
            height: 100%;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
        }
        .left-panel .label {
            font-size: 2.5vh; margin-top: 1vh;
            text-align: center; line-height: 1.2;
        }
        .left-panel .label .v { color: #4CAF50; font-weight: bold; }
        .center-panel {
            flex: 1; height: 100%;
            display: flex; align-items: center; justify-content: center;
        }
        #videoFeed {
            width: 100%; height: 100%;
            object-fit: contain;
            background: #000;
            border-radius: 10px;
        }
        .right-panel {
            width: 26vw; max-width: 180px;
            height: 100%;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            gap: 1.5vh;
        }
        .right-panel .label {
            font-size: 2.5vh;
            text-align: center; line-height: 1.2;
        }
        .right-panel .label .v { color: #FF9800; font-weight: bold; }
        input[type=range] {
            -webkit-appearance: none; appearance: none;
            background: transparent;
        }
        input[type=range]::-webkit-slider-runnable-track {
            background: #333; border-radius: 999px;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; appearance: none;
            border-radius: 50%; cursor: pointer;
        }
        input[type=range]::-moz-range-track {
            background: #333; border-radius: 999px; border: none;
        }
        input[type=range]::-moz-range-thumb {
            border-radius: 50%; cursor: pointer; border: none;
        }
        .speed-slider {
            width: 6vw; max-width: 44px;
            height: 70%;
            writing-mode: bt-lr;
        }
        .speed-slider::-webkit-slider-runnable-track { height: 100%; }
        .speed-slider::-webkit-slider-thumb {
            width: 5vw; height: 5vw; max-width: 36px; max-height: 36px;
            background: #4CAF50;
        }
        .speed-slider::-moz-range-track { height: 100%; }
        .speed-slider::-moz-range-thumb {
            width: 5vw; height: 5vw; max-width: 36px; max-height: 36px;
            background: #4CAF50;
        }
        .steer-slider {
            width: 80%; height: 7vw; max-height: 50px;
        }
        .steer-slider::-webkit-slider-runnable-track { height: 100%; }
        .steer-slider::-webkit-slider-thumb {
            width: 7vw; height: 7vw; max-width: 50px; max-height: 50px;
            background: #FF9800;
        }
        .steer-slider::-moz-range-track { height: 100%; }
        .steer-slider::-moz-range-thumb {
            width: 7vw; height: 7vw; max-width: 50px; max-height: 50px;
            background: #FF9800;
        }
        .steer-tick {
            width: 80%; display: flex; justify-content: space-between;
            font-size: 1.8vh; color: #666; padding: 0 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="top-bar">
            <div class="record-btn idle" id="recordBtn" onclick="toggleRecord()"></div>
            <span class="frame-count"><b id="frameCount">0</b> frames</span>
        </div>
        <div class="main">
            <div class="left-panel">
                <input type="range" class="speed-slider" id="speedSlider" min="-100" max="100" value="0" step="1">
                <div class="label"><span class="v" id="speedValue">0</span></div>
            </div>
            <div class="center-panel">
                <img id="videoFeed" src="/video_feed">
            </div>
            <div class="right-panel">
                <input type="range" class="steer-slider" id="servoSlider" min="40" max="140" value="95" step="1">
                <div class="label"><span class="v" id="servoValue">95</span>&deg;</div>
                <div class="steer-tick"><span>L</span><span>C</span><span>R</span></div>
            </div>
        </div>
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
"""

HTML_TEMPLATE = HTML_TEMPLATE.replace('DEADZONE', str(DEADZONE_PERCENT)).replace('CENTER', str(SERVO_CENTER)) % (DEADZONE_PERCENT, SERVO_CENTER)


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
