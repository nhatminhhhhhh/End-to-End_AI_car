import os
import sys
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

PCA9685_BUS = 0
PCA9685_ADDR = 0x40
SERVO_CHANNEL = 3

try:
    from pca9685 import PCA9685
    pca = PCA9685(bus=PCA9685_BUS, address=PCA9685_ADDR)
except Exception as e:
    print(f"[WARN] PCA9685 not available: {e}")
    pca = None

MOTOR_PWM_PIN = 33
IN1_PIN = 11
IN2_PIN = 13

PWM_MOTOR_FREQ = 1000
DEADZONE_PERCENT = 60
PWM_MIN = DEADZONE_PERCENT
PWM_MAX = 100

if GPIO:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(MOTOR_PWM_PIN, GPIO.OUT)
    GPIO.setup(IN1_PIN, GPIO.OUT)
    GPIO.setup(IN2_PIN, GPIO.OUT)
    motor_pwm = GPIO.PWM(MOTOR_PWM_PIN, PWM_MOTOR_FREQ)
    motor_pwm.start(0)
else:
    motor_pwm = None

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
    global frame_queue, camera_ready, recording, frame_count
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[Camera] Failed to open camera")
        return
    camera_ready.set()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if recording:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:21]
            filename = f"{ts}_s{current_steering}_t{current_throttle}.jpg"
            path = os.path.join(DATA_DIR, filename)
            try:
                cv2.imwrite(path, frame)
                frame_count += 1
                if frame_count <= 3:
                    print(f"[Save] {filename}")
            except Exception as e:
                print(f"[Save ERROR] {e}")

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
        GPIO.output(IN1_PIN, GPIO.LOW)
        GPIO.output(IN2_PIN, GPIO.HIGH)
        motor_pwm.ChangeDutyCycle(speed_out)
    else:
        GPIO.output(IN1_PIN, GPIO.HIGH)
        GPIO.output(IN2_PIN, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(-speed_out)


def set_servo(angle):
    global current_steering
    angle = max(SERVO_MIN, min(SERVO_MAX, angle))
    current_steering = angle
    if pca is None:
        print("[servo] pca is None, skip")
        return
    pulse = pca.angle_to_pulse(angle)
    print(f"[servo] angle={angle} pulse={pulse:.3f}ms")
    pca.set_servo_pulse(SERVO_CHANNEL, pulse)


def cleanup():
    if GPIO:
        motor_pwm.stop()
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
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
            font-family: sans-serif;
            color: #fff;
        }
        .container {
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: row;
        }
        .half { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .left { background: #1a1a2e; }
        .right { background: #16213e; }
        .slider-vertical {
            width: 80px;
            height: 80%;
            -webkit-appearance: slider-vertical;
            appearance: slider-vertical;
            writing-mode: bt-lr;
            background: #444;
            border-radius: 20px;
            outline: none;
            transform: rotate(180deg);
        }
        .slider-vertical::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }
        .slider-vertical::-moz-range-thumb {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }
        .slider-horizontal {
            width: 80%;
            height: 80px;
            -webkit-appearance: none;
            appearance: none;
            background: #444;
            border-radius: 20px;
            outline: none;
        }
        .slider-horizontal::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #FF9800;
            cursor: pointer;
        }
        .slider-horizontal::-moz-range-thumb {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #FF9800;
            cursor: pointer;
        }
        .label { margin-top: 20px; font-size: 20px; }
        .value { font-weight: bold; color: #4CAF50; }
        .label-right .value { color: #FF9800; }
        .bar {
            position: fixed;
            top: 12px;
            left: 0;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            font-size: 14px;
            color: #aaa;
            z-index: 10;
        }
        .rec {
            width: 16px; height: 16px; border-radius: 50%;
            border: 3px solid #666; cursor: pointer; transition: .2s;
        }
        .rec.idle { background: #555; }
        .rec.on { background: #f00; border-color: #f00; animation: blink .8s infinite; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
        .debug {
            position: fixed;
            top: 10px;
            right: 10px;
            font-size: 14px;
            color: #888;
            background: rgba(0,0,0,0.7);
            padding: 8px 12px;
            border-radius: 8px;
        }
        .status {
            position: fixed;
            bottom: 20px;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: 16px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="half left">
            <input type="range" class="slider-vertical" id="speedSlider" min="-100" max="100" value="0" step="1">
            <div class="label">Toc do: <span id="speedValue" class="value">0</span></div>
        </div>
        <div class="half right">
            <input type="range" class="slider-horizontal" id="servoSlider" min="40" max="140" value="95" step="1">
            <div class="label label-right">Goc: <span id="servoValue" class="value">95</span></div>
        </div>
    </div>
    <div class="bar">
        <div class="rec idle" id="recBtn" onclick="toggleRec()"></div>
        <span><b id="fc">0</b> frames</span>
    </div>
    <div class="debug">Deadzone PWM: 60% | Servo center: 95 deg</div>
    <div class="status">Keo slider LEN -> tien, XUONG -> lui. Tha tay -> dung. Servo trung tam 95 deg</div>
    <script>
        const speedSlider = document.getElementById('speedSlider');
        const servoSlider = document.getElementById('servoSlider');
        const speedValue = document.getElementById('speedValue');
        const servoValue = document.getElementById('servoValue');
        const recBtn = document.getElementById('recBtn');
        const fc = document.getElementById('fc');
        const SERVO_CENTER = 95;

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
            servoSlider.value = SERVO_CENTER;
            servoValue.innerText = SERVO_CENTER;
            fetch('/set_servo', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'angle=' + SERVO_CENTER });
        }
        function toggleRec() {
            fetch('/toggle_record', { method: 'POST' }).then(r => r.json()).then(d => {
                recBtn.className = 'rec ' + (d.recording ? 'on' : 'idle');
            });
        }
        setInterval(() => {
            fetch('/status').then(r => r.json()).then(d => {
                fc.innerText = d.frame_count;
                recBtn.className = 'rec ' + (d.recording ? 'on' : 'idle');
            });
        }, 1000);

        speedSlider.addEventListener('input', updateSpeed);
        servoSlider.addEventListener('input', updateServo);

        speedSlider.addEventListener('change', resetSpeed);
        servoSlider.addEventListener('change', resetServo);

        window.onload = function() {
            speedSlider.value = 0;
            speedValue.innerText = '0';
            fetch('/set_speed', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'speed=0' });
            resetServo();
        };
    </script>
</body>
</html>
"""

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
    print(f"[Record] {'START' if recording else 'STOP'}  (frame_count={frame_count})")
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
    print(f"  Camera: {CAMERA_CAPTURE_WIDTH}x{CAMERA_CAPTURE_HEIGHT} -> {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"  Save to: {os.path.abspath(DATA_DIR)}")
    print(f"  Format: {{timestamp}}_s{{steering}}_t{{throttle}}.jpg")
    print(f"  Deadzone PWM: {DEADZONE_PERCENT}%")
    print(f"  Servo center: {SERVO_CENTER} deg")
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
