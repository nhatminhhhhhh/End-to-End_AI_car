import os
import re
import glob

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

from config import INPUT_HEIGHT, INPUT_WIDTH, STEERING_NORM_FACTOR, THROTTLE_MAX_ABS


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class E2EDataset(Dataset):
    PATTERN = re.compile(r'_s([+-]?\d+)_t([+-]?\d+)\.jpg$')

    def __init__(self, data_dir, augment=False):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .jpg files found in {data_dir}")
        self.augment = augment
        print(f"[Dataset] {len(self.files)} samples from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return self.__getitem__((idx + 1) % len(self.files))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT)).astype(np.float32)

        label = self._parse_label(path)
        steering, throttle = label

        if self.augment and np.random.rand() > 0.5:
            frame = frame[:, ::-1, :]
            steering = -steering

        frame = frame / 255.0
        frame = (frame - _MEAN) / _STD
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        return frame, torch.tensor([steering, throttle], dtype=torch.float32)

    def _parse_label(self, path):
        basename = os.path.basename(path)
        m = self.PATTERN.search(basename)
        if m is None:
            return 0.0, 0.0
        s_raw = int(m.group(1))
        t_raw = int(m.group(2))
        steering = (s_raw - 95) / STEERING_NORM_FACTOR
        throttle = t_raw / THROTTLE_MAX_ABS
        return steering, throttle
