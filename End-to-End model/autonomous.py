import os
import time
import argparse

import cv2
import numpy as np
import torch

from config import (
    CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_FLIP,
    INPUT_HEIGHT, INPUT_WIDTH,
    SERVO_CENTER, SERVO_MIN, SERVO_MAX, STEERING_NORM_FACTOR,
    THROTTLE_MAX_ABS, MODEL_DIR,
)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("[WARN] Jetson.GPIO not available — simulation mode")
    GPIO = None

PCA9685_BUS = 0
PCA9685_ADDR = 0x40
SERVO_CHANNEL = 3

try:
    from pca9685 import PCA9685
    pca = PCA9685(bus=PCA9685_BUS, address=PCA9685_ADDR)
except Exception:
    print("[WARN] PCA9685 not available — servo simulation")
    pca = None

MOTOR_PWM_PIN = 33
IN1_PIN = 11
IN2_PIN = 13
PWM_MOTOR_FREQ = 1000
PWM_MIN = 60
PWM_MAX = 100

if GPIO:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(MOTOR_PWM_PIN, GPIO.OUT)
    GPIO.setup(IN1_PIN, GPIO.OUT)
    GPIO.setup(IN2_PIN, GPIO.OUT)
    motor_pwm = GPIO.PWM(MOTOR_PWM_PIN, PWM_MOTOR_FREQ)
    motor_pwm.start(0)


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


def set_motor(speed):
    if GPIO is None:
        return
    if speed == 0:
        GPIO.output(IN1_PIN, GPIO.LOW)
        GPIO.output(IN2_PIN, GPIO.LOW)
        motor_pwm.ChangeDutyCycle(0)
        return
    sign = 1 if speed > 0 else -1
    mapped = PWM_MIN + (abs(speed) / 100.0) * (PWM_MAX - PWM_MIN)
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
    if pca is None:
        return
    angle = max(SERVO_MIN, min(SERVO_MAX, angle))
    pulse = pca.angle_to_pulse(angle)
    pca.set_servo_pulse(SERVO_CHANNEL, pulse)


def cleanup():
    if GPIO:
        motor_pwm.stop()
        GPIO.cleanup()


def preprocess(frame):
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return t


def main():
    parser = argparse.ArgumentParser(description='E2E Autonomous Driving')
    parser.add_argument('--model', type=str, default=os.path.join(MODEL_DIR, 'best_model_traced.pt'))
    parser.add_argument('--fp16', action='store_true', help='Use FP16 inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    if args.fp16 and device.type == 'cuda':
        model = model.half()

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    set_servo(SERVO_CENTER)
    time.sleep(0.5)

    print("\n" + "=" * 50)
    print("  E2E AUTONOMOUS DRIVING")
    print("=" * 50)
    print(f"  Device: {device}")
    print(f"  Model: {args.model}")
    print(f"  Camera: {CAMERA_CAPTURE_WIDTH}x{CAMERA_CAPTURE_HEIGHT} -> {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    if args.fp16 and device.type == 'cuda':
        print(f"  FP16: enabled")
    print(f"  Deadzone PWM: {PWM_MIN}% | Servo center: {SERVO_CENTER} deg")
    print("=" * 50)
    print("  Running... Press Ctrl+C to stop\n")

    fps_counter = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0

    with torch.no_grad():
        while True:
            t_frame = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                continue

            t = preprocess(frame)
            if args.fp16 and device.type == 'cuda':
                t = t.half()
            t = t.to(device)

            outputs = model(t).cpu().numpy()[0]
            steering_norm, throttle_norm = outputs[0], outputs[1]

            steering_angle = int(steering_norm * STEERING_NORM_FACTOR + SERVO_CENTER)
            steering_angle = max(SERVO_MIN, min(SERVO_MAX, steering_angle))

            throttle_val = int(throttle_norm * THROTTLE_MAX_ABS)
            throttle_val = max(-THROTTLE_MAX_ABS, min(THROTTLE_MAX_ABS, throttle_val))

            set_servo(steering_angle)
            set_motor(throttle_val)

            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                display_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now
                frame_ms = (now - t_frame) * 1000
                print(f"  FPS: {display_fps:.0f}  Frame: {frame_ms:.0f}ms  "
                      f"Steer: {steering_angle} deg ({steering_norm:+.3f})  "
                      f"Throttle: {throttle_val} ({throttle_norm:+.3f})")

    cap.release()
    cleanup()
    print("[Autonomous] Stopped")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        set_motor(0)
        cleanup()
        print("\n[Autonomous] Interrupted")
