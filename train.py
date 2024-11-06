import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unetskip import CombinedNetwork
from loss import SSIM,FocalLoss,MultiClassFocalLoss
import torch.nn.functional as F
import os,json,random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from vail import vail
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2
def width_height(path,crop):
    image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
    # ���ͼƬ��������50�������ѡȡ50��ͼƬ������ѡ������ͼƬ
    if len(image_files) > 50:
        image_files = random.sample(image_files, 50)
    # ��ʼ�趨�ܿ�Ⱥ��ܳ���Ϊ0��ͼƬ����Ϊ0
    total_width = 0
    total_height = 0
    num_images = 0
    # ��������ѡ�е�ͼƬ�ļ�
    for image_file in image_files:
        # ��ͼƬ
        with Image.open(path + image_file) as img:
            # ��ȡͼƬ��С
            width, height = img.size
            # ���ӵ��ܿ�Ⱥ��ܳ���
            total_width += width
            total_height += height
            # ����ͼƬ����
            num_images += 1

    # ����ƽ����Ⱥ�ƽ������
    average_width = total_width / num_images
    average_height = total_height / num_images

    crop_width = round(average_width /crop )*crop
    crop_height = round(average_height / crop) * crop
    return crop_width,crop_height

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
def compute_sub_blocks(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    new_height = round(height / 1024) * 1024
    new_width = round(width / 1024) * 1024
    return (new_width//1024) * (new_height//1024)
def train_on_device(args):
    dataset = MVTecDRAEMTrainDataset(args.image_path + "/", args,condition="train",resize_shape=[args.crop,args.crop])
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            shuffle=False, num_workers=0)

    labels = args.labels.split(',')
    model = CombinedNetwork(in_channels=3, mid_channels = 3, out_channels=len(labels)+1)
    model.cuda()
    model.apply(weights_init)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr}, ], weight_decay=0)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)
    n_iter = 0
    best_auroc = 0
    pre_loss = 1000000
    criterion =MultiClassFocalLoss(gamma=2, alpha=[1]+[1]*len(labels), size_average=True)#[0.1]+[1]*len(labels)
    for epoch in range(args.epochs):

        Loss = 0
        model.train()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
        for i_batch, sample_batched in progress_bar:
            gray_batch = sample_batched["image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()

            gray_rec, out_mask = model(gray_batch)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            gray_rec = torch.softmax(gray_rec, dim=1)

            # if ak % 3 == 0:
            #     to_pil(gray_batch[0]).save('./image' + '.jpg')
            #     to_pil(out_mask_sm[0][1]).save('./ano'+'.jpg')
            #     to_pil(anomaly_mask[0][1]).save('./mask'+'.jpg')
            #
            # ak += 1

            segment_loss1 = criterion(out_mask_sm, anomaly_mask)*5+F.smooth_l1_loss(out_mask_sm[:, 1:, :, :], anomaly_mask[:, 1:, :, :])
            segment_loss2 = criterion(gray_rec, anomaly_mask) * 5 + F.smooth_l1_loss(gray_rec[:, 1:, :, :],
                                                                                        anomaly_mask[:, 1:, :, :])
            loss = segment_loss1 + segment_loss2
            Loss = Loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: " + str(epoch)+"   Loss: "+str(Loss.item()))


        scheduler.step()
        model.eval()
        dummy_input1 = torch.randn(1, 3, args.resize, args.resize).to('cuda')

        if args.complete_training == "yes":
            if Loss.item()<pre_loss:
                torch.onnx.export(model, dummy_input1,
                                  os.path.join(args.model_savepath, "model.onnx"),
                                  opset_version=11)
                torch.save(model.state_dict(), os.path.join(args.model_savepath,  "model.pckl"))
                n_iter = 0
                precisions, recalls, f1s, aurocs = vail(model, args)
                preci = precisions
                recall = recalls
                f1 = f1s
                lun = epoch
                best_auroc = aurocs
                pre_loss = Loss.item()
            else:
                n_iter=n_iter+1
            if (epoch<100 and n_iter>=10) or (epoch>=100 and n_iter>=5) or (epoch == args.epochs-1):
                break
        else:
            torch.onnx.export(model, dummy_input1,
                              os.path.join(args.model_savepath, "model.onnx"),
                              opset_version=11)
            torch.save(model.state_dict(), os.path.join(args.model_savepath,  "model.pckl"))
            if epoch%3==0 and epoch>=0:
                precisions, recalls, f1s, aurocs = vail(model,args)
                preci = precisions
                recall = recalls
                f1 = f1s
                lun = epoch
                best_auroc = aurocs
        
    params = {
        "precision":preci,
        "recall": recall,
        "f1":f1,
        "best_result_epoch":lun,
        "auroc":best_auroc,
        "width":args.width,
        "height":args.height,
        "need_cut":args.need_cut
    }
    with open(os.path.join(args.model_savepath,"params.json"),'w') as f:
        json.dump(params,f)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Python script for image processing")
    parser.add_argument('--bs', action='store', type=int, default=4, required=False, help="Batch size for training")
    parser.add_argument('--lr', action='store', type=float, default=0.0001, required=False, help="Learning rate")
    parser.add_argument('--epochs', action='store', type=int, default=300, required=False,
                        help="Number of epochs for training")
    parser.add_argument("--input1", type=str, default="../image/image/", help="Component input port 1.")
    parser.add_argument("--image_path", type=str, default="../image/image/", help="Component input port 1.")
    parser.add_argument("--mask_path", type=str, default="../image/mask/", help="Component input port 2.")
    parser.add_argument("--model_savepath", type=str, default="../Mod/",
                        help="Component output port 1.")


    parser.add_argument("--labels", type=str, default="ano"#,label_2,label_3",
                                    ,help="Component output port 2.")

    parser.add_argument("--height", type=int, default=4096, help="The height of the expected image size.")
    parser.add_argument("--width", type=int, default=3072, help="The width of the expected image size.")
    parser.add_argument("--crop", type=int, default=1024, help="The size of cropped image.")
    parser.add_argument("--resize", type=int, default=256, help="The size of resized image.")
    parser.add_argument("--need_cut", type=str, default="yes", help="yes/no")
    parser.add_argument("--complete_training", type=str, default="yes", help="yes/no")

    parser.add_argument("--output1", type=str, default="/Mura_MMBR/code/output/output1/",
                        help="Component output port 1.")
    parser.add_argument("--output2", type=str, default="/Mura_MMBR/code/output/output2/",
                        help="Component output port 2.")
    parser.add_argument("--output3", type=str, default=None, help="Component output port 3.")
    parser.add_argument("--output4", type=str, default=None, help="Component output port 4.")

    args = parser.parse_args()
    
    args.image_path = args.input1+'image/'
    args.mask_path = args.input1+'mask/'
    
    args.model_savepath = os.path.dirname(args.output1.rstrip("/"))+"/"
    if args.need_cut == "yes":
        args.width,args.height =width_height(args.image_path,args.crop)
    else:
        args.width, args.height, args.crop = args.resize

    if args.height % args.crop == 0 or args.height % args.crop == 0:
            train_on_device(args)
    else:
        print("The length and width must be divided by the crop S!!")

