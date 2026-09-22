"""
Deep Detailed Network (DDN) for rain streak removal — CVPR 2017.

Device-safe guided filter (upstream hard-coded CUDA in box_filter).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


def _diff_x(input: torch.Tensor, r: int) -> torch.Tensor:
    left = input[:, :, r : 2 * r + 1]
    middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
    return torch.cat((left, middle, right), dim=2)


def _diff_y(input: torch.Tensor, r: int) -> torch.Tensor:
    left = input[:, :, :, r : 2 * r + 1]
    middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
    return torch.cat((left, middle, right), dim=3)


def _box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    return _diff_y(torch.cumsum(_diff_x(torch.cumsum(x, dim=2), r), dim=3), r)


def guided_filter(x: torch.Tensor, y: torch.Tensor, r: int, eps: float = 1e-8) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    n = _box_filter(torch.ones((1, 1, x.shape[2], x.shape[3]), device=device, dtype=dtype), r)
    mean_x = _box_filter(x, r) / n
    mean_y = _box_filter(y, r) / n
    cov_xy = _box_filter(x * y, r) / n - mean_x * mean_y
    var_x = _box_filter(x * x, r) / n - mean_x * mean_x
    a = cov_xy / (var_x + eps)
    b = mean_y - a * mean_x
    return _box_filter(a, r) / n * x + _box_filter(b, r) / n


class DeRain(nn.Module):
    """DDN rain streak removal (Rain100L/H pretrained weights)."""

    def __init__(
        self,
        n_features: int = 16,
        n_channels: int = 3,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(n_channels, n_features, kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(n_features, eps=0.001, momentum=0.99),
            nn.ReLU(),
        ]
        self.b1 = nn.Sequential(*layers)

        layers = []
        for _ in range(12):
            layers.extend(
                [
                    nn.Conv2d(n_features, n_features, kernel_size, padding=padding, bias=True),
                    nn.BatchNorm2d(n_features, eps=0.001, momentum=0.99),
                    nn.ReLU(),
                    nn.Conv2d(n_features, n_features, kernel_size, padding=padding, bias=True),
                    nn.BatchNorm2d(n_features, eps=0.001, momentum=0.99),
                    nn.ReLU(),
                ]
            )
        self.b2 = nn.Sequential(*layers)

        self.b3 = nn.Sequential(
            nn.Conv2d(n_features, n_channels, kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(n_channels, eps=0.001, momentum=0.99),
        )
        self._initialize_weights()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        base = guided_filter(images, images, 15, 1)
        detail = images - base
        output_shortcut = self.b1(detail)
        output_shortcut = self.b2(output_shortcut) + output_shortcut
        neg_residual = self.b3(output_shortcut)
        return images + neg_residual

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
