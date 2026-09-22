
import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader

from PyTorch_dataset_Rain100 import Rain100_Test
from PyTorch_GuidedFilter import guided_filter

import skimage.io as io

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

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

        # Block 1
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_features,
                kernel_size=kernel_size,
                padding=padding,
                bias=True
            )
        )
        layers.append(nn.BatchNorm2d(n_features, eps=0.001, momentum=0.99))
        layers.append(nn.ReLU())

        self.b1 = nn.Sequential(*layers)

        # Block 2
        layers = []

        for i in range(12):

            layers.append(
                nn.Conv2d(
                    in_channels=n_features,
                    out_channels=n_features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True
                )
            )
            layers.append(nn.BatchNorm2d(n_features, eps=0.001, momentum=0.99))
            layers.append(nn.ReLU())

            layers.append(
                nn.Conv2d(
                    in_channels=n_features,
                    out_channels=n_features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True
                )
            )
            layers.append(nn.BatchNorm2d(n_features, eps=0.001, momentum=0.99))
            layers.append(nn.ReLU())

        self.b2 = nn.Sequential(*layers)

        # Block 3
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=n_features,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True
            )
        )

        layers.append(nn.BatchNorm2d(n_channels, eps=0.001, momentum=0.99))

        self.b3 = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, images):

        base = guided_filter(images, images, 15, 1, nhwc=True)

        detail = images - base

        output_shortcut = self.b1(detail)

        output_shortcut = self.b2(output_shortcut) + output_shortcut

        neg_residual = self.b3(output_shortcut)

        final_out = images + neg_residual

        return final_out

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# =========================
# ARGS
# =========================
parser = argparse.ArgumentParser(description='PyTorch DeRain')

parser.add_argument(
    '--input_path_test',
    default="./input"
)

parser.add_argument(
    '--model_dir',
    type=str,
    default="./model/rain1400_finetune_new"
)

parser.add_argument(
    '--output_dir',
    type=str,
    default="./output"
)

args = parser.parse_args()


# =========================
# CUDA
# =========================
cuda = torch.cuda.is_available()

if not cuda:
    raise Exception("CUDA GPU bulunamadı!")


# =========================
# OUTPUT DIR
# =========================
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# =========================
# MAIN
# =========================
if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    print('===> Building model')

    checkpoint = torch.load(
    os.path.join(args.model_dir, 'model_best.pth'),
    map_location='cpu',
    weights_only=False
    )

    if hasattr(checkpoint, 'module'):
        model = checkpoint.module
    else:
        model = checkpoint

    model = model.cuda()
    model.eval()

    # Dataset
    DDataset_test = Rain100_Test(args)

    DLoader_test = DataLoader(
        dataset=DDataset_test,
        num_workers=0,
        drop_last=False,
        batch_size=1,
        shuffle=False
    )

    with torch.no_grad():

        i = 0

        for _, batch_test in enumerate(DLoader_test):

            print("Processing:", i)

            i += 1

            # INPUT
            img_input = batch_test[0].cuda()

            # MODEL
            img_dn = model(img_input)

            # tensor -> numpy
            img_dn = img_dn.squeeze(0)

            img_dn = img_dn.cpu().numpy().astype(np.float32)

            img_dn = np.transpose(img_dn, (1, 2, 0))

            # SAVE NAME
            save_file = batch_test[2][0] + '_output.png'

            # float -> uint8
            img_save = (
                np.clip(img_dn, 0, 1) * 255
            ).astype(np.uint8)

            # SAVE
            io.imsave(
                os.path.join(output_dir, save_file),
                img_save
            )

    print("Finished!")




