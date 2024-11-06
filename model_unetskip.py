import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedNetwork(nn.Module):
    def __init__(self,in_channels=3, mid_channels = 3,out_channels=3):
        super(CombinedNetwork, self).__init__()
        self.ReconstructiveSubNetwork = ReconstructiveSubNetwork(in_channels=in_channels, out_channels=out_channels, base_width=128)
        self.DiscriminativeSubNetwork = DiscriminativeSubNetwork(in_channels=out_channels+mid_channels, out_channels=out_channels, base_channels=64, out_features=False)

    def forward(self, gray_rec):
        rec_output = self.ReconstructiveSubNetwork(gray_rec)
        joined_in = torch.cat((rec_output, gray_rec), dim=1)
        output_segment = self.DiscriminativeSubNetwork(joined_in)
        return rec_output, output_segment

class ReconstructiveSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, x):
        b5,b4,b3,b2,b1 = self.encoder(x)
        output = self.decoder(b5,b4,b3,b2,b1)
        return output

class DiscriminativeSubNetwork(nn.Module):
    def __init__(self,in_channels=6, out_channels=2, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        #self.segment_act = torch.nn.Sigmoid()
        self.out_features = out_features
    def forward(self, x):
        b1,b2,b3,b4,b5,b6 = self.encoder_segment(x)
        output_segment = self.decoder_segment(b1,b2,b3,b4,b5,b6)
        return output_segment


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=False))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=False))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1,b2,b3,b4,b5,b6

class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(
            nn.ConvTranspose2d(base_width * 8, base_width * 8, kernel_size=2, stride=2)
        )
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=False)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_width * 8, base_width * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=False)
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=False)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_width * 4, base_width * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=False)
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=False)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_width * 2, base_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False)
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_width, base_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False)
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)

        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)

        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)

        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)

        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)

        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out



class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False))
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=False))
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=False))
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False))
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=False))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b5,b4,b3,b2,b1



class DecoderReconstructive(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_width * 8, base_width * 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=False)
        )

        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=False)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_width * 4, base_width * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=False)
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=False)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_width * 2, base_width * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=False)
        )

        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width * 2, base_width * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 1),
            nn.ReLU(inplace=False)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_width, base_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False)
        )

        self.db4 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False)
        )

        self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5,b4,b3,b2,b1):
        up1 = self.up1(b5)
        db1 = self.db1(up1+b4)

        up2 = self.up2(db1)
        db2 = self.db2(up2+b3)

        up3 = self.up3(db2)
        db3 = self.db3(up3+b2)

        up4 = self.up4(db3)
        db4 = self.db4(up4+b1)

        out = self.fin_out(db4)
        return out


    # self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
