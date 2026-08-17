import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_CAPTURE_WIDTH = 1280
CAMERA_CAPTURE_HEIGHT = 720
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 60
CAMERA_FLIP = 2

INPUT_HEIGHT = 160
INPUT_WIDTH = 120
NUM_CHANNELS = 3

SERVO_CENTER = 95
SERVO_MIN = 40
SERVO_MAX = 140
STEERING_NORM_FACTOR = 55 # góc quay tối đa của xe so với tâm 

THROTTLE_MAX_ABS = 100

DATA_DIR = os.path.join(BASE_DIR, 'collected_data')
MODEL_DIR = os.path.join(BASE_DIR, 'checkpoints')

CONV_FILTERS = [24, 32, 64, 64, 64]
CONV_KERNELS = [5, 5, 5, 3, 3]
CONV_STRIDES = [2, 2, 2, 1, 1]
DROPOUT_RATE = 0.2 # giảm overfit

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 200
PATIENCE = 10 
VAL_SPLIT = 0.2 # 20% dữ liệu dùng để test mô hình trong quá trình train
