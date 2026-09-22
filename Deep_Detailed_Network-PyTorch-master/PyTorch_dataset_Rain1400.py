
import os
import random
import numpy as np
import imageio.v2 as imageio

import torch
import torch.utils.data as data


def is_img(x):
    return x.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp')
    )


def np2tensor(img):

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = np.ascontiguousarray(
        img.transpose((2, 0, 1))
    )

    return torch.from_numpy(img).float()


class Rain1400(data.Dataset):

    def __init__(self, args, isTrain=True):

        self.args = args
        self.isTrain = isTrain
        self.patch_size = args.patch_size

        self.rain_dir = args.input_path
        self.clean_dir = args.gt_path

        self.rain_images, self.clean_images = self.scan()

    def scan(self):

        rain_images = sorted([
            os.path.join(self.rain_dir, x)
            for x in os.listdir(self.rain_dir)
            if is_img(x)
        ])

        clean_images = []

        for rain_path in rain_images:

            name = os.path.basename(rain_path)

            clean_path = os.path.join(
                self.clean_dir,
                name
            )

            if not os.path.exists(clean_path):
                raise FileNotFoundError(
                    f"Clean image bulunamadı: {clean_path}"
                )

            clean_images.append(clean_path)

        return rain_images, clean_images

    def __len__(self):
        return len(self.rain_images)

    def __getitem__(self, idx):

        rain_path = self.rain_images[idx]
        clean_path = self.clean_images[idx]

        rain = imageio.imread(rain_path)
        clean = imageio.imread(clean_path)

        rain = rain.astype(np.float32) / 255.0
        clean = clean.astype(np.float32) / 255.0

        if self.isTrain:

            h, w = rain.shape[:2]

            if h > self.patch_size and w > self.patch_size:

                x = random.randint(
                    0,
                    h - self.patch_size
                )

                y = random.randint(
                    0,
                    w - self.patch_size
                )

                rain = rain[
                    x:x+self.patch_size,
                    y:y+self.patch_size
                ]

                clean = clean[
                    x:x+self.patch_size,
                    y:y+self.patch_size
                ]

        rain = np2tensor(rain)
        clean = np2tensor(clean)

        return rain, clean
