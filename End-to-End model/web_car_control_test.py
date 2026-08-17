# -*- coding: utf-8 -*-
import Jetson.GPIO as GPIO
import time
import threading
from flask import Flask, request, jsonify, render_template_string

# ===== Cấu hình chân =====
MOTOR_PWM_PIN = 32
SERVO_PWM_PIN = 33
IN1_PIN = 11
IN2_PIN = 13

PWM_MOTOR_FREQ = 1000
PWM_SERVO_FREQ = 50

SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5

# ===== Cấu hình deadzone PWM =====
DEADZONE_PERCENT = 60
PWM_MIN = DEADZONE_PERCENT
PWM_MAX = 100

# ===== Giới hạn servo và trung tâm =====
SERVO_MIN_ANGLE = 40
SERVO_MAX_ANGLE = 140
SERVO_CENTER = 95   # Bù offset cơ khí

# ===== Khởi tạo GPIO =====
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

# ===== Biến trạng thái =====
current_speed = 0
current_servo_angle = SERVO_CENTER

# ===== Hàm điều khiển =====
def set_motor(speed):
    global current_speed
    current_speed = speed
    if speed == 0:
        GPIO.output(IN1_PIN, GPIO.LOW)
        GPIO.output(IN2_PIN, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(0)
        return
    sign = 1 if speed > 0 else -1
    abs_speed = abs(speed)
    mapped_pwm = PWM_MIN + (abs_speed / 100.0) * (PWM_MAX - PWM_MIN)
    mapped_pwm = max(PWM_MIN, min(PWM_MAX, mapped_pwm))
    speed_out = sign * mapped_pwm
    if speed_out > 0:
        GPIO.output(IN1_PIN, GPIO.HIGH)
        GPIO.output(IN2_PIN, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(speed_out)
    else:
        GPIO.output(IN1_PIN, GPIO.LOW)
        GPIO.output(IN2_PIN, GPIO.HIGH)
        motor_pwm.ChangeDutyCycle(-speed_out)

def set_servo(angle):
    global current_servo_angle
    angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
    current_servo_angle = angle
    duty = SERVO_MIN_DUTY + (angle / 180.0) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)
    servo_pwm.ChangeDutyCycle(duty)

def cleanup():
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()

# ===== Flask App =====
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Điều khiển xe</title>
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
        .status {
            position: fixed;
            bottom: 20px;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: 16px;
            color: #aaa;
        }
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
    </style>
</head>
<body>
    <div class="container">
        <div class="half left">
            <input type="range" class="slider-vertical" id="speedSlider" min="-100" max="100" value="0" step="1">
            <div class="label">Tốc độ: <span id="speedValue" class="value">0</span>%</div>
        </div>
        <div class="half right">
            <!-- Center = 95°, giới hạn 40°–140° -->
            <input type="range" class="slider-horizontal" id="servoSlider" min="40" max="140" value="95" step="1">
            <div class="label label-right">Góc: <span id="servoValue" class="value">95</span>°</div>
        </div>
    </div>
    <div class="debug">Deadzone PWM: 60% | Servo center: 95°</div>
    <div class="status">Kéo slider tốc độ LÊN → tiến, XUỐNG → lùi. Servo trung tâm 95° (bù offset)</div>

    <script>
        const speedSlider = document.getElementById('speedSlider');
        const servoSlider = document.getElementById('servoSlider');
        const speedValue = document.getElementById('speedValue');
        const servoValue = document.getElementById('servoValue');
        const SERVO_CENTER = 95;

        function updateSpeed() {
            const raw = parseInt(speedSlider.value);
            const val = -raw;
            speedValue.innerText = val;
            fetch('/set_speed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'speed=' + val
            });
        }

        function updateServo() {
            const val = parseInt(servoSlider.value);
            servoValue.innerText = val;
            fetch('/set_servo', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'angle=' + val
            });
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

        speedSlider.addEventListener('input', updateSpeed);
        servoSlider.addEventListener('input', updateServo);

        speedSlider.addEventListener('mouseup', resetSpeed);
        speedSlider.addEventListener('touchend', resetSpeed);
        servoSlider.addEventListener('mouseup', resetServo);
        servoSlider.addEventListener('touchend', resetServo);

        window.onload = function() {
            resetSpeed();
            resetServo();
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

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

if __name__ == '__main__':
    try:
        # Khởi tạo servo về 95°
        set_servo(SERVO_CENTER)
        print("\n" + "="*50)
        print("🚗 ROBOT WIFI CONTROL (SERVO CENTER = 95°)")
        print("="*50)
        print("🌐 Server: http://10.42.0.1:5000 (hoặc IP thực tế)")
        print("⚡ Servo trung tâm: 95° (bù offset), giới hạn 40°–140°.")
        print("   Deadzone PWM: 60%.\n")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C")
    finally:
        set_motor(0)
        cleanup()
        print("✅ Đã dọn dẹp.")
