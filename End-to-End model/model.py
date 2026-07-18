import torch
import torch.nn as nn


class E2EModel(nn.Module):
    def __init__(self, num_channels=3, input_height=160, input_width=120, dropout=0.2):
        super().__init__()

        filters = [24, 32, 64, 64, 64]
        kernels = [5, 5, 5, 3, 3]
        strides = [2, 2, 2, 1, 1]

        conv_layers = []
        in_c = num_channels
        h, w = input_height, input_width
        for i in range(len(filters)):
            conv_layers.append(nn.Conv2d(in_c, filters[i], kernels[i], strides[i]))
            conv_layers.append(nn.BatchNorm2d(filters[i]))
            conv_layers.append(nn.ReLU(inplace=True))
            in_c = filters[i]
            h = (h - kernels[i]) // strides[i] + 1
            w = (w - kernels[i]) // strides[i] + 1

        self.conv = nn.Sequential(*conv_layers)

        flattened = h * w * filters[-1]
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flattened, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(50, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def compute_input_dims(height, width, kernels, strides):
    h, w = height, width
    for k, s in zip(kernels, strides):
        h = (h - k) // s + 1
        w = (w - k) // s + 1
    return h, w
