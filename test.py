import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unetskip import CombinedNetwork
from loss import SSIM,FocalLoss,MultiClassFocalLoss
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from vail import vail
def generate_image(ndarr):
    # 预设的颜色
    colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]])
    # 选择与ndarr中f数量相同的颜色，如果颜色数少于f，通过循环扩充颜色数
    used_colors = colors[:, :ndarr.shape[0] % colors.shape[1]]
    for _ in range(ndarr.shape[0] // colors.shape[1]):
        used_colors = np.concatenate((used_colors, colors), axis=1)

    img_mask = ndarr > 0.5  # 创建一个大小相同的mask，大于0.5的位置为True
    img = np.tensordot(img_mask, used_colors, axes=1)  # 利用tensor乘法将mask对应部分换成相应颜色
    img = np.sum(img, axis=0, dtype=np.uint8) # 某一点多种颜色同时存在，取和

    return img
def caculate_metrics(target,pred):
    pred = np.where(pred>0.5,1,0)
    pred = pred.flatten()
    target = target.flatten()
    try:
        auroc = roc_auc_score(target, pred)
    except:
        auroc = 0
    precision = precision_score(target, pred,zero_division=1)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)

    return precision, recall, f1, auroc
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(args):
    dataset = MVTecDRAEMTrainDataset(args.image_path + "/", args,condition="train",resize_shape=[args.crop,args.crop])
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            shuffle=False, num_workers=0)

    labels = args.labels.split(',')
    model = CombinedNetwork(in_channels=3, mid_channels = 1, out_channels=len(labels)+1)
    model.cuda()
    model.apply(weights_init)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr}, ], weight_decay=0)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)
    n_iter = 0
    best_auroc = 0
    criterion = MultiClassFocalLoss(gamma=2, alpha=[0.2]+[1]*len(labels), size_average=True)
    for epoch in range(args.epochs):

        Loss = 0
        model.train()
        print("Epoch: "+str(epoch))
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
        for i_batch, sample_batched in progress_bar:
            gray_batch = sample_batched["image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()
            gray_grayimage=sample_batched["auggray"].cuda()

            gray_rec, out_mask = model(gray_batch, gray_grayimage)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            gray_rec = torch.softmax(gray_rec, dim=1)

            segment_loss1 = criterion(out_mask_sm, anomaly_mask[:, 1:, :, :])*5+F.smooth_l1_loss(out_mask_sm[:, 1:, :, :], anomaly_mask[:, 1:, :, :])
            segment_loss2 = criterion(gray_rec, anomaly_mask[:, 1:, :, :]) * 5 + F.smooth_l1_loss(gray_rec[:, 1:, :, :],
                                                                                        anomaly_mask[:, 1:, :, :])
            loss = segment_loss1 + segment_loss2
            Loss = Loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter +=1
        print(Loss)
        scheduler.step()
        if epoch%5==0:
            model.eval()
            precisions, recalls, f1s, aurocs = vail(model,args)
            if aurocs>best_auroc:
                best_auroc = aurocs
                dummy_input1 = torch.randn(1, 1, args.resize, args.resize).to('cuda')
                dummy_input2 = torch.randn(1, 3, args.resize, args.resize).to('cuda')
                torch.onnx.export(model, (dummy_input1, dummy_input2),
                                  os.path.join(args.model_savepath, str(epoch) + '-' + "model.onnx"),
                                  opset_version=11)
                torch.save(model.state_dict(),os.path.join(args.model_savepath, str(epoch) + '-' + "model.pckl"))

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Python script for image processing")

    parser.add_argument("--image_path", type=str, default="../mask_test_image/image/", help="Component input port 1.")
    parser.add_argument("--output_path", type=str, default="../mask_test_image/mask/", help="Component input port 2.")
    parser.add_argument("--model_path", type=str, default="../mask_test_image/model/",
                        help="Component output port 1.")
    parser.add_argument("--output_image_savepath", type=str, default="/Mura_MMBR/code/output/output_image_savepath/",
                        help="Component output port 2.")

    parser.add_argument("--labels", type=str, default="label_1,label_2,label_3",
                        help="Component output port 2.")

    parser.add_argument("--height", type=int, default=1024, help="The height of the expected image size.")
    parser.add_argument("--width", type=int, default=5120, help="The width of the expected image size.")
    parser.add_argument("--crop", type=int, default=1024, help="The size of cropped image.")
    parser.add_argument("--resize", type=int, default=512, help="The size of resized image.")

    args = parser.parse_args()

    if args.height % args.crop == 0 or args.height % args.crop == 0:
        with torch.cuda.device(args.gpu_id):
            train_on_device(args)
    else:
        print("The length and width must be divided by the crop S!!")


