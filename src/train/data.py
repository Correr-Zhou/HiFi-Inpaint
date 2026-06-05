import torch
import os
import json
import random
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler
import torchvision.transforms as T

random.seed(2222)

HEIGHT_WIDTH_RESOLUTIONS = [
    (512, 2048), (512, 1984), (512, 1920), (512, 1856),
    (576, 1792), (576, 1728), (576, 1664),
    (640, 1600), (640, 1536), (768, 768),
    (704, 1472), (704, 1408), (704, 1344),
    (768, 1344), (768, 1280),
    (832, 1216), (832, 1152),
    (896, 1152), (896, 1088),
    (960, 1088), (960, 1024),
    (1024, 1024), (1024, 960), (1024, 576),
    (1088, 960), (1088, 896),
    (1152, 896), (1152, 832),
    (1216, 832), (1280, 768),
    (1344, 768), (1408, 704),
    (1472, 704), (1536, 640),
    (1600, 640), (1664, 576),
    (1728, 576), (1792, 576),
    (1856, 512), (1920, 512),
    (1984, 512), (2048, 512)
]
RATIO_RESOLUTION_PAIRS = [(round(w / h, 3), (h, w)) for h, w in HEIGHT_WIDTH_RESOLUTIONS]


class DatasetGenAI(TorchDataset):
    """
    Training dataset for HiFi-Inpaint.

    Expected JSON format (HP-Image-40K compatible):
        {
            "ref_image_path": "path/to/reference_image.png",
            "gt_image_path": "path/to/ground_truth_image.png",
            "condition_image_path": "path/to/masked_condition_image.png",
            "mask_path": "path/to/binary_mask.png",
            "caption": "text description of the image"
        }
    """
    def __init__(self, meta_path_str, resolution, use_caption=False, random_mask=False,
                 sample_num=1000000, pad_color=(255, 255, 255), bucket_sampler=False):
        self.samples = []
        self.return_pil_image = False
        self.condition_type = "subject"
        self.to_tensor = T.ToTensor()

        if isinstance(meta_path_str, list):
            for path in meta_path_str:
                print(f"Loading dataset: {path}")
                with open(path, 'r') as f:
                    self.samples.extend(json.load(f))
        else:
            print(f"Loading dataset: {meta_path_str}")
            with open(meta_path_str, 'r') as f:
                self.samples.extend(json.load(f))

        self.pad_color = pad_color
        self.bucket_sampler = bucket_sampler
        self.resolution = resolution
        self.use_caption = use_caption
        self.random_mask = random_mask

        if sample_num < len(self.samples):
            self.samples = random.sample(self.samples, sample_num)

    def __len__(self):
        return len(self.samples)

    def get_resize_and_crop_transforms(self, resolution):
        return transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
        ])

    def apply_high_pass_filter(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        rows, cols = image.shape
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols, 2), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= 30**2
        mask[mask_area] = 0
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back = np.uint8((img_back / np.max(img_back)) * 255)
        return Image.fromarray(img_back)

    def extract_mask(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_rect = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > max_area:
                max_area = area
                max_rect = (x, y, w, h)
        mask = np.zeros_like(image)
        if max_rect:
            x, y, w, h = max_rect
            mask[y:y+h, x:x+w] = 255
        return Image.fromarray(mask)

    def find_best_fit_resolution(self, image_width, image_height):
        target_ratio = image_width / image_height
        best_resolution = None
        min_diff = float('inf')
        for ratio, resolution in RATIO_RESOLUTION_PAIRS:
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                best_resolution = resolution
        return min_diff, best_resolution

    def enhance_mask(self, cond_image: Image.Image, mask_area: Image.Image) -> Image.Image:
        mask_np = np.array(mask_area)
        ys, xs = np.where(mask_np == 255)
        if len(xs) == 0 or len(ys) == 0:
            return cond_image

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        strategy = random.choice([1, 2, 3, 4])
        new_width = width
        new_height = height

        if strategy == 2:
            new_height = int(height * 1.2)
        elif strategy == 3:
            new_width = int(width * 1.2)
        elif strategy == 4:
            new_width = int(width * 1.1)
            new_height = int(height * 1.1)

        center_x = x_min + width // 2
        center_y = y_min + height // 2
        new_x_min = center_x - new_width // 2
        new_y_min = center_y - new_height // 2
        new_x_max = new_x_min + new_width
        new_y_max = new_y_min + new_height

        img_width, img_height = cond_image.size
        new_x_min = max(new_x_min, 0)
        new_y_min = max(new_y_min, 0)
        new_x_max = min(new_x_max, img_width)
        new_y_max = min(new_y_max, img_height)

        cond_np = np.array(cond_image)
        cond_np[new_y_min:new_y_max, new_x_min:new_x_max] = 0
        return Image.fromarray(cond_np)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return idx
        d = self.samples[idx]
        while not os.path.exists(d['gt_image_path']):
            rand_idx = random.choice(list(range(len(self.samples))))
            d = self.samples[rand_idx]

        ref_image = Image.open(d['ref_image_path']).convert("RGB")
        cond_image = Image.open(d['condition_image_path']).convert("RGB")
        gt_image = Image.open(d['gt_image_path']).convert("RGB")
        texture = self.apply_high_pass_filter(d['ref_image_path'])
        mask_area = Image.open(d['mask_path'])
        if self.random_mask:
            cond_image = self.enhance_mask(cond_image, mask_area)

        if self.bucket_sampler:
            nearest_ratio, nearest_res = self.find_best_fit_resolution(gt_image.width, gt_image.height)
        else:
            nearest_res = self.resolution

        if self.use_caption:
            text = d['caption']
        else:
            text = ' '

        resize_and_crop = self.get_resize_and_crop_transforms(nearest_res)
        condition_size = nearest_res[1]

        return {
            "image": self.to_tensor(resize_and_crop(gt_image)),
            "mask": self.to_tensor(resize_and_crop(cond_image)),
            "mask_area": self.to_tensor(resize_and_crop(mask_area)),
            "texture": self.to_tensor(resize_and_crop(texture.convert('RGB'))),
            "condition": self.to_tensor(resize_and_crop(ref_image)),
            "condition_type": self.condition_type,
            "description": text,
            "position_delta": np.array([0, -condition_size // 16]),
        }


class BucketSampler(Sampler):
    def __init__(self, data_source: DatasetGenAI, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = False) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.buckets = {resolution: [] for ratio, resolution in RATIO_RESOLUTION_PAIRS}
        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last:
            return sum((len(bucket) // self.batch_size) for bucket in self.buckets.values())
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            img_metadata = data["image_metadata"]
            h, w = img_metadata["height"], img_metadata["width"]

            self.buckets[(h, w)].append(data)
            if len(self.buckets[(h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(h, w)])
                yield self.buckets[(h, w)]
                del self.buckets[(h, w)]
                self.buckets[(h, w)] = []

        if self.drop_last:
            return

        for hw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[hw]
                self.buckets[hw] = []
