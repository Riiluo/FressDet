# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional
from ultralytics.utils.patches import imread
import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from .utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS


class BaseDataset(Dataset):


    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = False  # Rectangular batches disabled (using LetterBox for fixed size)
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if isinstance(cache, str):
            cache = cache.lower()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images
        if cache == "ram" and not self.check_cache_ram():
            cache = False
        self.cache = cache  # 存储cache作为实例变量
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            # im_files = im_files[: round(len(im_files) * self.fraction)]
            num_elements_to_select = round(len(im_files) * self.fraction)
            im_files = random.sample(im_files, num_elements_to_select)
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            # 检查文件是否为 .npy 格式
            if Path(f).suffix.lower() == '.npy':  # load npy
                # 加载npy文件
                im = np.load(f, allow_pickle=False)
                # 如果需要转置，可以在这里进行
                if im.ndim == 3 and im.shape[2] > im.shape[0]:  # 如果第三个维度大于第一个维度，可能需要转置
                    im = np.transpose(im, (2, 1, 0))  # 转置为 (H, W, C)
                    # im = np.transpose(im, (1, 2, 0))  # 转置为 (H, W, C)
            else:
                # 优先尝试加载对应的npy文件（如果存在）
                if fn.exists() and fn.suffix == '.npy':  # load npy
                    im = np.load(fn, allow_pickle=False)
                else:
                    # 加载标准图像文件
                    im = imread(f)
                    if im is None:
                        raise FileNotFoundError(f"Image not found: {f}")
            
            h0, w0 = im.shape[:2]  # orig hw
            # 只在需要缓存到 RAM 或训练增强阶段时保留图像在 self.ims 中，
            # 对于 cache='disk' 且 augment=False 的验证/测试集，不在内存里长期保存整图。
            if self.cache == "ram" or self.augment:
                self.ims[i] = im
            else:
                self.ims[i] = None
            self.im_hw0[i], self.im_hw[i] = (h0, w0), im.shape[:2]  # hw_original, hw_resized

            if self.augment:  # 千万不能删，否则会爆内存（训练集通过 buffer 限制在内存中的样本数）
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {cache})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            # 使用load_image方法来确保正确的图像加载和转换
            try:
                # 注意：load_image 会将图像缓存在 self.ims[i] 中，这里只用于生成 .npy，
                # 所以在保存到磁盘后要显式释放以避免占用大量 RAM。
                im, _, _ = self.load_image(i)
                if im is not None:
                    np.save(f.as_posix(), im, allow_pickle=False)
            except Exception:
                # 如果load_image失败，尝试直接读取
                im = imread(self.im_files[i])
                if im is not None:
                    np.save(f.as_posix(), im, allow_pickle=False)

        # 无论是否新建 .npy，都不要因为 cache='disk' 把整套图像常驻在内存里
        self.ims[i], self.im_hw0[i], self.im_hw[i] = None, None, None

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            try:
                # 使用更安全的图像加载方式
                if Path(im_file).suffix.lower() == '.npy':
                    im = np.load(im_file, allow_pickle=False)
                    # 处理npy文件的维度转换
                    if im.ndim == 3 and im.shape[2] > im.shape[0]:
                        im = np.transpose(im, (2, 1, 0))
                else:
                    im = imread(im_file)
                
                if im is None:
                    continue  # 跳过无法加载的图像
                ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
                b += im.nbytes * ratio**2
            except Exception:
                continue  # 跳过有问题的图像
        
        if b == 0:  # 如果没有成功加载任何图像，禁用缓存
            return False
            
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
