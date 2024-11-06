import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
import glob
import cv2
import itertools
def generate_image(ndarr):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255],
                       [255, 255, 255], [255, 255, 0], [0, 255, 255],
                       [255, 0, 255]], dtype=np.uint8)  # 颜色数组
    f, h, w = ndarr.shape
    used_colors = np.array([colors[i % len(colors)] for i in range(f)])  # 确保颜色数量与图像数量相同
    img_mask = np.where(ndarr > 0.5,1,0).astype(np.uint8)
    color_img = np.zeros((h,w,3),np.uint8)
    for i in range(1,f):
        # 将单通道图像转换为三通道
        img_3_channel = cv2.cvtColor(img_mask[i], cv2.COLOR_GRAY2BGR)
        # 将图像的每个通道与颜色数组中的相应值相乘，这将为图像提供颜色
        colored = img_3_channel * used_colors[i-1]
        color_img = cv2.add(color_img, colored)
    return color_img

def reshape_tensors(tensors, rows, cols):
    assert len(tensors) == rows * cols, 'Number of tensors does not fit the shape'
    lists = [[tensors[i * cols + j] for j in range(cols)] for i in range(rows)]
    row_tensors = [np.concatenate(lists[i], axis=-1) for i in range(rows)]
    image_tensor = np.concatenate(row_tensors, axis=-2)
    return image_tensor
def caculate_metrics(target,pred):
    pred1 = pred
    pred1 = pred1.flatten()
    pred = np.where(pred>0.5,1,0).astype(int)
    target = np.where(target > 0.5, 1, 0).astype(int)
    pred = pred.flatten()
    target = target.flatten()
    auroc = roc_auc_score(target, pred1)
    if np.sum(pred)==0:
        precision=0
    else:
        precision = precision_score(target, pred,zero_division=1)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)

    return precision, recall, f1, auroc

def vail(model,args):
    precisions, recalls, f1s, aurocs = [], [], [], []
    all_image_paths = sorted(glob.glob(args.image_path + "/*.jpg"))
    all_mask_paths = []
    anomaly_image_paths = []
    mask_image_path = args.mask_path
    labels = args.labels.split(',')
    for label in labels:
        rot_dir = mask_image_path + label + "/"
        all_mask_paths += sorted(glob.glob(rot_dir + "/*.png"))
    filenames = [os.path.splitext(os.path.split(path)[1])[0] for path in all_mask_paths]
    for filename in filenames:
        new_path = os.path.join(args.image_path, filename + ".jpg")
        anomaly_image_paths.append(new_path)
    anomaly_image_paths = list(set(anomaly_image_paths))
    cutoff = int(0.95 * len(all_image_paths))
    cutoff1 = int(0.3 * len(anomaly_image_paths))
    image_paths = list(set(all_image_paths[cutoff:] + anomaly_image_paths[:cutoff1]))

    for file_path in image_paths:
        image = cv2.imread(file_path)
        image = cv2.resize(image, (args.width, args.height))
        fname = os.path.basename(file_path)
        fname, _ = os.path.splitext(fname)
        labels = args.labels.split(',')
        masks = []
        mask = np.ones(image.shape, dtype=np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masks.append(mask)
        for label in labels:
            path = os.path.join(args.mask_path, label, fname + '.png')
            if os.path.exists(path):
                mask = cv2.imread(path)
                mask = cv2.resize(mask, (args.width, args.height))
            else:
                mask = np.zeros(image.shape, dtype=np.uint8)
            mask[mask > 0] = 255
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask)
            masks[0][mask > 0] = 0
        gt = np.dstack(masks)

        gt = cv2.resize(gt, (args.width, args.height))
        out_masks = []
        gray_recs = []
        anomaly_masks = []
        crop_h, crop_w = args.crop, args.crop
        h, w, _ = image.shape
        for m, n in itertools.product(range(0, h, crop_h), range(0, w, crop_w)):
            image_seg = image[m:m + crop_h, n:n + crop_w]
            anomaly_mask_seg = gt[m:m + crop_h, n:n + crop_w]
            image_seg = cv2.resize(image_seg, (args.resize, args.resize))
            image_seg = np.array(image_seg).reshape((image_seg.shape[0], image_seg.shape[1], image_seg.shape[2])).astype(np.float32) / 255.0
            anomaly_mask_seg = cv2.resize(anomaly_mask_seg, (args.resize, args.resize))
            anomaly_mask_seg = np.array(anomaly_mask_seg).astype(np.float32) / 255.0
            auggray = cv2.cvtColor(image_seg, cv2.COLOR_BGR2GRAY)
            auggray = auggray[:, :, None]
            image_seg = np.transpose(image_seg, (2, 0, 1))
            auggray = np.transpose(auggray, (2, 0, 1))
            anomaly_mask_seg = np.transpose(anomaly_mask_seg, (2, 0, 1))
            gray_rec, out_mask = model(torch.from_numpy(np.expand_dims(image_seg,axis=0)).cuda())
            out_masks.append(out_mask.detach().cpu().numpy()[0, :, :, :])
            gray_recs.append(gray_rec.detach().cpu().numpy()[0, :, :, :])
            anomaly_masks.append(anomaly_mask_seg)

        true_mask_cv = reshape_tensors(anomaly_masks, h // crop_h, w // crop_w)
        out_mask = reshape_tensors(out_masks,h//crop_h,w//crop_w)

        out_mask_sm = torch.softmax(torch.tensor(out_mask), dim=0)
        out_mask_cv = out_mask_sm.detach().cpu().numpy()


        precision, recall, f1, auroc = caculate_metrics(true_mask_cv, out_mask_cv)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aurocs.append(auroc)
        out_mask_cv = generate_image(out_mask_cv)
        cv2.imwrite("./test.jpg",out_mask_cv)

    precision = np.average(precisions)
    recall = np.average(recalls)
    f1 = np.average(f1s)
    auroc = np.average(aurocs)
    print(f"precision:{precision} recall:{recall} f1:{f1} auroc{auroc}")
    return precision, recall, f1,auroc



