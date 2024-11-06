import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import itertools
import random
from scipy import ndimage
cv2.ocl.setUseOpenCL(False)
class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, args,resize_shape=None):
        
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.images = sorted(glob.glob(root_dir+"/*.jpg"))
        self.args = args

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):

        crop_h, crop_w = self.resize_shape[0], self.resize_shape[1]
        augment_image = cv2.imread(image_path)
        augment_image = cv2.resize(augment_image,(self.args.width,self.args.height))
        h, w, _ = augment_image.shape
        # directory, filename = os.path.split(image_path)
        # gt_path = directory +'_gt' + '/' + filename
        # gt = cv2.imread(gt_path)
        # gt = cv2.resize(gt, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # gt = gt[:, :, :1]
        # anomaly_mask = np.array(gt).reshape((gt.shape[0], gt.shape[1], gt.shape[2])).astype(np.float32) / 255.0
        aug_list = []
        auggray_list = []
        for m, n in itertools.product(range(0, h, crop_h), range(0, w, crop_w)):
            split2 = augment_image[m:m+crop_h, n:n+crop_w]
            split2 = cv2.resize(split2,(self.args.resize,self.args.resize))
            augmented_image=np.array(split2).reshape((split2.shape[0], split2.shape[1], split2.shape[2])).astype(np.float32)/ 255.0
            auggray = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)
            auggray=auggray[:,:,None]
            augmented_image = np.transpose(augmented_image, (2, 0, 1))
            auggray = np.transpose(auggray, (2, 0, 1))
            aug_list.append(augmented_image)
            auggray_list.append(auggray)
        # anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return aug_list, aug_list,auggray_list

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        aug_list, aug_list,auggray_list = self.transform_image(img_path)
        sample = {'image': aug_list, 'mask': aug_list, 'img_name': os.path.splitext(os.path.basename(img_path))[0],'imagegray':auggray_list}
        return sample

class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, args, condition,resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.args = args
        self.len = (self.args.height // self.args.crop) * (self.args.width//self.args.crop)
        self.condition=condition
        all_image_paths = sorted(glob.glob(root_dir + "/*.jpg"))
        all_mask_paths = []
        anomaly_image_paths =[]
        mask_image_path = self.args.mask_path
        labels = self.args.labels.split(',')
        for label in labels:
            rot_dir = mask_image_path+label+"/"
            all_mask_paths += sorted(glob.glob(rot_dir + "/*.png"))
        filenames = [os.path.splitext(os.path.split(path)[1])[0] for path in all_mask_paths]
        for filename in filenames:
            new_path = os.path.join(root_dir,filename+".jpg")
            anomaly_image_paths.append(new_path)
        anomaly_image_paths = list(set(anomaly_image_paths))
        cutoff = int(0.8 * len(all_image_paths))
        cutoff1 = int(0.2 * len(anomaly_image_paths))
        if condition == 'train':
            self.image_paths = all_image_paths[:cutoff]
        else:
            self.image_paths = list(set(all_image_paths[cutoff:]+anomaly_image_paths[:cutoff1]))
        self.condition = condition

    def __len__(self):
        return len(self.image_paths)*self.len


    def transform_image(self, image_path,idx):

        weight1,height1 = self.resize_shape[0],self.resize_shape[1]

        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.args.width, self.args.height))
        fname = os.path.basename(image_path)
        fname, _ = os.path.splitext(fname)
        labels = self.args.labels.split(',')
        masks = []
        mask = np.ones(image.shape, dtype=np.uint8)*255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masks.append(mask)
        for label in labels:
            path = os.path.join(self.args.mask_path, label, fname + '.png')
            path1 = os.path.join(self.args.mask_path, label, fname + '.jpg')
            if os.path.exists(path):
                mask = cv2.imread(path)
                mask = cv2.resize(mask, (self.args.width, self.args.height))
            elif os.path.exists(path1):
                mask = cv2.imread(path1)
                mask = cv2.resize(mask, (self.args.width, self.args.height))
            else:
                mask = np.zeros(image.shape, dtype=np.uint8)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            mask[mask > 0] = 255

            for j in range(len(masks)):
                masks[j][mask > 0] = 0
            masks.append(mask)
            # for j in range(len()-1)
            # masks[0][mask > 0] = 0

        gt = np.dstack(masks)
        image = cv2.resize(image, (self.args.width, self.args.height))
        gt = cv2.resize(gt, (self.args.width, self.args.height))

        rows_per_image = image.shape[0] // weight1
        cols_per_image = image.shape[1] // height1

        # 获取索引号对应的行数和列数
        row_index = idx // cols_per_image
        col_index = idx % cols_per_image

        # 计算区域的起始和结束的行列号
        row_start = row_index * weight1
        row_end = row_start + weight1

        col_start = col_index * weight1
        col_end = col_start + weight1

        # 切片获取图片的区域内容
        image = np.copy(image[row_start:row_end, col_start:col_end, :])
        gt = np.copy(gt[row_start:row_end, col_start:col_end, :])


        beilv = self.args.resize
        image = cv2.resize(image, (beilv, beilv))

        gts = cv2.resize(gt, (beilv, beilv))

        image = np.array(image).astype(np.float32) / 255.0
        anomaly_mask = gts.astype(np.float32) / 255.0
        anomaly_mask = np.where(anomaly_mask>0.5,1.0,0.0)

        auggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        auggray=auggray[:,:,None]

        image = np.transpose(image, (2, 0, 1))

        auggray = np.transpose(auggray, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        return image, anomaly_mask, auggray

    def __getitem__(self, idx):

        idx = torch.randint(0, len(self.image_paths)*(self.len), (1,)).item()
        image, anomaly_mask, auggray = self.transform_image(self.image_paths[idx // (self.len)], idx % (self.len))

        sample = {'image': image, "anomaly_mask": anomaly_mask,'auggray':auggray}

        return sample
