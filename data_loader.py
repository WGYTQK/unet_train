import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import itertools
import random
from scipy import ndimage


def color_jitter(image, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.0):
    """Apply color jitter to BGR image (OpenCV format)"""
    # Brightness
    if brightness != 1.0:
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    # Contrast
    if contrast != 1.0:
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    # Convert to HSV for saturation and hue
    if saturation != 1.0 or hue != 0.0:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Saturation
        if saturation != 1.0:
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)

        # Hue (shift by hue radians)
        if hue != 0.0:
            hsv[..., 0] = (hsv[..., 0] + hue * 180) % 180

        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image


def random_flip(image):
    """Random horizontal or vertical flip (50% chance each)"""
    if random.random() < 0.5:
        image = cv2.flip(image, 1)  # Horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 0)  # Vertical flip
    return image


def random_grayscale(image):
    """Convert to grayscale with probability"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def random_autocontrast(image):
    """Autocontrast using CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def random_equalize(image):
    """Histogram equalization"""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_random_augmentation(image):
    """Randomly select and apply ONE augmentation"""
    augmentations = [
        lambda img: color_jitter(img, brightness=random.uniform(0.8, 1.2)),
        lambda img: color_jitter(img, contrast=random.uniform(0.8, 1.2)),
        lambda img: color_jitter(img, saturation=random.uniform(0.8, 1.2), hue=random.uniform(-0.2, 0.2)),
        # random_flip,
        random_grayscale,
        random_autocontrast,
        # random_equalize,
    ]

    # Randomly select one augmentation
    augmentation = random.choice(augmentations)
    return augmentation(image)
def random_gamma_correction(image, gamma_range=(0.8, 1.5)):
    """
    随机应用Gamma校正以调整图像亮度。
    """
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def random_gaussian_noise(image, mean=0, sigma_range=(0, 0.05)):
    """
    随机添加高斯噪声。
    """
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def random_color_space_augmentation(image):
    """
    随机在LAB颜色空间中增强图像。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 对L通道（亮度）进行随机调整
    l = cv2.convertScaleAbs(l, alpha=random.uniform(0.9, 1.1), beta=random.randint(-10, 10))

    # 对A和B通道（颜色）进行随机调整
    a = cv2.convertScaleAbs(a, alpha=random.uniform(0.9, 1.1), beta=random.randint(-10, 10))
    b = cv2.convertScaleAbs(b, alpha=random.uniform(0.9, 1.1), beta=random.randint(-10, 10))

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_random_brightness_color_augmentation(image):
    """
    结合多种亮度和颜色增强策略。
    """
    augmentations = [
        lambda img: random_gamma_correction(img),
        lambda img: random_gaussian_noise(img),
        lambda img: random_color_space_augmentation(img),
        lambda img: color_jitter(img, brightness=random.uniform(0.8, 1.2)),
        lambda img: color_jitter(img, contrast=random.uniform(0.8, 1.2)),
        lambda img: color_jitter(img, saturation=random.uniform(0.8, 1.2), hue=random.uniform(-0.2, 0.2)),
    ]

    # 随机选择并应用多个增强策略
    num_augs = random.randint(1, 3)  # 每次随机应用1到3个增强
    selected_augs = random.sample(augmentations, num_augs)
    for aug in selected_augs:
        image = aug(image)
    return image

# 修改旋转函数以支持out_image
def random_rotate(image, mask, out_image, degrees=[0, 90, 180, 270]):
    rotation_degree = random.choice(degrees)
    rows, cols = image.shape[:2]

    # 生成旋转矩阵（对图像中心旋转）
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_degree, 1)

    # 旋转三通道图像（使用默认双线性插值）
    rotated_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

    # 旋转单通道 GT 和 out_image 时强制使用最近邻插值
    rotated_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
    rotated_out = cv2.warpAffine(out_image, M, (cols, rows), flags=cv2.INTER_NEAREST)

    return rotated_image, rotated_mask, rotated_out


class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        # self.images = sorted(glob.glob(root_dir+"/*.png"))
        self.images = sorted(glob.glob(root_dir + "/*.jpg") + glob.glob(root_dir + "/*.png"))

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        image = cv2.imread(image_path)
        image_ori = cv2.imread(image_path)

        beilv = 256
        image = cv2.resize(image, (beilv, beilv))

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        auggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('gray_img.jpg', auggray*255.0)
        auggray = auggray[:, :, None]

        image = np.transpose(image, (2, 0, 1))
        auggray = np.transpose(auggray, (2, 0, 1))

        return image, image_ori, auggray

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]

        filename = os.path.basename(img_path)
        # 分割文件名和扩展名，取第一个部分作为图片的名字
        image_name = os.path.splitext(filename)[0]

        aug_list, image_ori, auggray = self.transform_image(img_path)
        sample = {'image': aug_list, 'image_name': image_name, 'image_ori': image_ori, "auggray": auggray}
        return sample


def get_random_image_path_fast(root_dir="D:/datasets/dtd/images"):
    """使用 glob 快速匹配所有图片路径"""
    all_images = glob.glob(
        os.path.join(root_dir, "**", "*.[pPjJ][nNpP][gG]"),  # 匹配 .png, .PNG, .jpg, .JPG 等
        recursive=True
    )

    if not all_images:
        raise FileNotFoundError(f"No images found in {root_dir}")

    return random.choice(all_images)


class MVTecDRAEMTrainDataset(Dataset):
    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        增强版MVTec数据集加载器

        参数:
            root_dir (str): 正常图像目录路径
            anomaly_source_path (str): 异常样本目录路径
            resize_shape (tuple): 图像目标尺寸 (height, width)
        """
        self.root_dir = root_dir
        self.mask_dir = anomaly_source_path
        self.resize_shape = resize_shape

        # 增强功能开关配置
        self.augmentation_flags = {
            "rotate": True,  # 随机旋转
            "brightness_contrast": True,  # 亮度对比度调整
            "hue_saturation": True,  # 色调饱和度调整
            "color_augmentation": True,  # 颜色增强
            "scale_augmentation": True,  # 多尺度缩放
            "random_crop": True,  # 随机裁剪（读取后立即执行）
            "patch_overlay": True  # 随机干扰块（仅影响原图）
        }

        self.beilv = resize_shape[0]  # 缩放基准尺寸
        self.image_paths = sorted(glob.glob(root_dir + "/*.jpg") + glob.glob(root_dir + "/*.png"))
        self.mask_paths = sorted(glob.glob(anomaly_source_path + "/*.jpg") + glob.glob(anomaly_source_path + "/*.png"))

    def __len__(self):
        return len(self.image_paths)

    def random_crop_resize(self, image, mask=None, crop_ratio=(0.8, 0.95)):
        """
        随机裁剪图像部分区域并resize回原尺寸
        同时处理原图和mask，保持空间变换同步

        参数:
            image: 原图 (H,W,C)
            mask: 对应的mask (H,W) 或 None
            crop_ratio: 裁剪比例范围

        返回:
            处理后的image和mask (或仅image当mask为None时)
        """
        if not self.augmentation_flags["random_crop"]:
            return (image, mask) if mask is not None else image

        h, w = image.shape[:2]
        ratio = random.uniform(*crop_ratio)
        crop_h, crop_w = int(h * ratio), int(w * ratio)

        # 随机确定裁剪起始点 (确保原图和mask使用相同的裁剪位置)
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)

        # 执行裁剪并恢复原始尺寸
        cropped_img = image[y:y + crop_h, x:x + crop_w]
        image = cv2.resize(cropped_img, (w, h), interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            cropped_mask = mask[y:y + crop_h, x:x + crop_w]
            mask = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return image, mask

        return image

    def apply_random_patch_overlay(self, image, patch_size_range=(10, 50), num_patches_range=(1, 3)):
        """
        在图像上随机放置干扰块
        （只影响原图，不修改mask）
        """
        if not self.augmentation_flags["patch_overlay"]:
            return image

        h, w = image.shape[:2]
        num_patches = random.randint(*num_patches_range)

        for _ in range(num_patches):
            patch_size = random.randint(*patch_size_range)
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)

            # 生成随机干扰块
            if random.random() > 0.5:  # 随机颜色块
                patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
            else:  # 从异常源随机取纹理块
                if len(self.mask_paths) > 0:
                    patch_img = cv2.imread(random.choice(self.mask_paths))
                    ph, pw = patch_img.shape[:2]
                    px = random.randint(0, pw - patch_size)
                    py = random.randint(0, ph - patch_size)
                    patch = patch_img[py:py + patch_size, px:px + patch_size]
                else:
                    patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)

            # 应用干扰块（不修改mask）
            image[y:y + patch_size, x:x + patch_size] = patch

        return image


    def random_rotate(self, image, mask, out_image, angle_range=(-30, 30)):
        """随机角度旋转增强"""
        if not self.augmentation_flags["rotate"]:
            return image, mask, out_image

        angle = random.uniform(angle_range[0], angle_range[1])
        rows, cols = image.shape[:2]

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)
        rotated_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
        rotated_out = cv2.warpAffine(out_image, M, (cols, rows), flags=cv2.INTER_NEAREST)

        return rotated_image, rotated_mask, rotated_out

    def random_adjust_brightness_contrast(self, image, brightness_range=(0.95, 1.2), contrast_range=(0.9, 1.2)):
        """随机调整亮度和对比度"""
        if not self.augmentation_flags["brightness_contrast"]:
            return image

        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        contrast_factor = random.uniform(contrast_range[0], contrast_range[1])

        image = image.astype(np.float32)
        image = image * contrast_factor + (brightness_factor - 1) * 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        return image

    def random_adjust_hue_saturation(self, image, hue_range=(-5, 5), saturation_range=(0.9, 1.1)):
        """随机调整色调和饱和度"""
        if not self.augmentation_flags["hue_saturation"]:
            return image

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_shift = random.randint(hue_range[0], hue_range[1])
        saturation_factor = random.uniform(saturation_range[0], saturation_range[1])

        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image

    def transform_image(self, image_path):
        beilv = self.beilv

        # 1. 读取原始图像和对应的mask/out_image
        # image = cv2.imread(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # 分离通道
        y, cr, cb = cv2.split(img_yuv)
        # 只对亮度通道进行均衡化
        y_eq = cv2.equalizeHist(y)
        # 合并通道
        img_yuv_eq = cv2.merge((y_eq, cr, cb))
        # 转换回BGR颜色空间
        image = cv2.cvtColor(img_yuv_eq, cv2.COLOR_YCrCb2BGR)
        # cv2.imwrite("./test.jpg",image)
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")


        mask_path = image_path.replace("image", "mask").replace(".jpg", ".png").replace(".JPG", ".png")

        out_path = image_path.replace("image", "out")
        # cv2.imwrite("./test.jpg", image)
        # 读取或生成默认的mask和out_image
        mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else np.zeros((image.shape[0], image.shape[1]),
                                                                                   dtype=np.uint8)
        if not os.path.exists(mask_path):
            print(" aaaaa"+str(mask_path))
        out_image = cv2.imread(out_path, 0) if os.path.exists(out_path) else np.ones((image.shape[0], image.shape[1]),
                                                                                     dtype=np.uint8)

        # 2. 初始二值化处理
        out_image = np.where(out_image > 100, 255, 0).astype(np.uint8)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        # 3. 随机裁剪增强（同步处理image和mask）
        if self.augmentation_flags["random_crop"]:
            image, mask = self.random_crop_resize(image, mask)
            out_image = cv2.resize(out_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 4. 调整到目标尺寸（保持同步）
        orig_size = image.shape[:2]
        image = cv2.resize(image, (beilv, beilv), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (beilv, beilv), interpolation=cv2.INTER_NEAREST)
        out_image = cv2.resize(out_image, (beilv, beilv), interpolation=cv2.INTER_NEAREST)

        # 5. 应用随机干扰块（仅影响原图）
        if self.augmentation_flags["patch_overlay"]:
            image = self.apply_random_patch_overlay(image)

        # 6. 颜色增强（仅影响原图）
        if self.augmentation_flags["brightness_contrast"]:
            image = self.random_adjust_brightness_contrast(image)
        if self.augmentation_flags["hue_saturation"]:
            image = self.random_adjust_hue_saturation(image)
        if self.augmentation_flags["color_augmentation"]:
            image = apply_random_brightness_color_augmentation(image)  # 假设这个方法已实现

        # 7. 多尺度缩放增强（同步处理所有）
        if self.augmentation_flags["scale_augmentation"]:
            scale_factor = random.uniform(0.8, 1.2)
            new_size = (int(beilv * scale_factor), int(beilv * scale_factor))

            image = cv2.resize(cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR),
                               (beilv, beilv), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST),
                              (beilv, beilv), interpolation=cv2.INTER_NEAREST)
            out_image = cv2.resize(cv2.resize(out_image, new_size, interpolation=cv2.INTER_NEAREST),
                                   (beilv, beilv), interpolation=cv2.INTER_NEAREST)

        # 8. 随机旋转（同步处理所有）
        image, mask, out_image = self.random_rotate(image, mask, out_image)

        # 9. 最终处理
        gt = mask[:, :, None] if mask.ndim == 2 else mask
        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32) / 255.0
        mask = np.array(gt).reshape((gt.shape[0], gt.shape[1], 1)).astype(np.float32) / 255.0
        auggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("./test.jpg",auggray*255.0)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        auggray = auggray[:, :, None]
        auggray = np.transpose(auggray, (2, 0, 1))


        return image, mask, auggray, image

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        augment_image, mask, auggray, image = self.transform_image(self.image_paths[idx])
        sample = {'augment_image': augment_image, "mask": mask, "auggray": auggray, "image": image}
        return sample
