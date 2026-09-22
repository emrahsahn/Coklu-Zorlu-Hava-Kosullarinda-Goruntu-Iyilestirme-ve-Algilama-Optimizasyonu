
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import imageio.v2 as imageio

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from PyTorch_GuidedFilter import guided_filter


# =========================
# NETWORK
# =========================
class DeRain(nn.Module):

    def __init__(
        self,
        n_features=16,
        n_channels=3,
        kernel_size=3,
        padding=1
    ):
        super(DeRain, self).__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                n_channels,
                n_features,
                kernel_size,
                padding=padding
            )
        )

        layers.append(
            nn.BatchNorm2d(
                n_features,
                eps=0.001,
                momentum=0.99
            )
        )

        layers.append(nn.ReLU())

        self.b1 = nn.Sequential(*layers)

        layers = []

        for _ in range(12):

            layers.append(
                nn.Conv2d(
                    n_features,
                    n_features,
                    kernel_size,
                    padding=padding
                )
            )

            layers.append(
                nn.BatchNorm2d(
                    n_features,
                    eps=0.001,
                    momentum=0.99
                )
            )

            layers.append(nn.ReLU())

            layers.append(
                nn.Conv2d(
                    n_features,
                    n_features,
                    kernel_size,
                    padding=padding
                )
            )

            layers.append(
                nn.BatchNorm2d(
                    n_features,
                    eps=0.001,
                    momentum=0.99
                )
            )

            layers.append(nn.ReLU())

        self.b2 = nn.Sequential(*layers)

        layers = []

        layers.append(
            nn.Conv2d(
                n_features,
                n_channels,
                kernel_size,
                padding=padding
            )
        )

        layers.append(
            nn.BatchNorm2d(
                n_channels,
                eps=0.001,
                momentum=0.99
            )
        )

        self.b3 = nn.Sequential(*layers)

    def forward(self, images):

        base = guided_filter(
            images,
            images,
            15,
            1,
            nhwc=True
        )

        detail = images - base

        out = self.b1(detail)

        out = self.b2(out) + out

        neg_residual = self.b3(out)

        return images + neg_residual


# =========================
# PATHS
# =========================

RAIN_DIR = "./Rain1400_Test/rainy_image"
GT_DIR = "./Rain1400_Test/ground_truth"

MODEL_PATH = "./model/rain1400_finetune/model_best.pth"

OUTPUT_DIR = "./output_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

if hasattr(checkpoint, "module"):
    model = checkpoint.module
else:
    model = checkpoint

model = model.cuda()
model.eval()


# =========================
# IMAGE LIST
# =========================

image_list = sorted([
    x for x in os.listdir(RAIN_DIR)
    if x.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

psnr_list = []
ssim_list = []

metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.txt"
)

metrics_file = open(
    metrics_path,
    "w",
    encoding="utf-8"
)

# =========================
# LOOP
# =========================

with torch.no_grad():

    for idx, name in enumerate(image_list):

        print(
            f"[{idx+1}/{len(image_list)}] {name}"
        )

        rain_path = os.path.join(
            RAIN_DIR,
            name
        )

        gt_path = os.path.join(
            GT_DIR,
            name
        )

        rain = imageio.imread(rain_path)

        gt = imageio.imread(gt_path)

        if len(rain.shape) == 2:
            rain = np.stack(
                [rain, rain, rain],
                axis=2
            )

        if len(gt.shape) == 2:
            gt = np.stack(
                [gt, gt, gt],
                axis=2
            )

        rain = rain.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0

        inp = torch.from_numpy(
            rain.transpose(2, 0, 1)
        ).unsqueeze(0).float().cuda()

        pred = model(inp)

        pred = (
            pred.squeeze(0)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        pred = np.clip(pred, 0, 1)

        psnr = peak_signal_noise_ratio(
            gt,
            pred,
            data_range=1.0
        )

        ssim = structural_similarity(
            gt,
            pred,
            channel_axis=2,
            data_range=1.0
        )

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        out_name = os.path.splitext(name)[0]
        out_name += "_output.png"

        imageio.imwrite(
            os.path.join(
                OUTPUT_DIR,
                out_name
            ),
            (pred * 255).astype(np.uint8)
        )

        metrics_file.write(
            f"{name} | "
            f"PSNR={psnr:.4f} | "
            f"SSIM={ssim:.4f}\n"
        )

# =========================
# FINAL
# =========================

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)

metrics_file.write("\n")
metrics_file.write(
    f"Average PSNR = {avg_psnr:.4f}\n"
)
metrics_file.write(
    f"Average SSIM = {avg_ssim:.4f}\n"
)

metrics_file.close()

print()
print("================================")
print(f"Average PSNR : {avg_psnr:.4f}")
print(f"Average SSIM : {avg_ssim:.4f}")
print("================================")
print()
print("Metrics saved:", metrics_path)

