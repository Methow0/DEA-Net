
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.transforms import Resize, Compose, RandomCrop, ToTensor, ToPILImage, Normalize
import cv2
from DefEDNetmain.DefEDNet import SeparableConv2d
from FullNet import DoubleConv
from MyUnet import DoubleConvn
from RFBmodel import BasicRFB_a
from backbones.resnet.resnet_factory import get_resnet_backbone

from functools import partial

from backbones.scale_attention_layer import scale_atten_convblock, conv3x3, conv1x1
from models import DeepLab
from network import deeplabv3plus_resnet101
from smatunetmodels.layers import DepthwiseSeparableConv, CBAM, ChannelAttention, SpatialAttention
from smatunetmodels.unet_parts_depthwise_separable import DoubleConvDS

nonlinearity = partial(F.elu, inplace=True)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(F_g,F_int,1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.w_g(g)
        # 上采样的 l 卷积
        x1 = self.w_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi




class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out


class DACblock_with_inception(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class DACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv1 = DepthwiseSeparableConv(in_channels,in_channels//4,1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        # self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.conv3 = DepthwiseSeparableConv(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class my_up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=2):
        super(my_up,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        self.conv1 = nn.Conv2d(in_channels,in_channels//2,1,padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x1_ = self.conv1(x1)
        x3 = x2*x1_
        x = torch.cat([x3, x1_], dim=1)
        return self.conv(x)+self.conv(x1)
class my_up1(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=2):
        super(my_up1,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x3 = x2*x1
        x = torch.cat([x3, x1], dim=1)
        return self.conv(x)+x1


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class CE_Net_(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(CE_Net_, self).__init__()
        print("构造CE_Net_")
        filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        # self.corp = Compose([
        #     Resize(26),
        #     ToTensor(),
        #     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])


        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])



        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(128, filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
		
    def forward(self, x,x1=None):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder


        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out1 = self.finalrelu2(out)
        out = self.finalconv3(out)

        if x1==None:
            return out
        else:
            x1 = self.firstconv(x1)
            x1 = self.firstbn(x1)
            x1 = self.firstrelu(x1)
            x1 = self.firstmaxpool(x1)
            de1 = self.encoder1(x1)
            de2 = self.encoder2(de1)
            de3 = self.encoder3(de2)
            de4 = self.encoder4(de3)

            # Center
            de4 = self.dblock(de4)
            de4 = self.spp(de4)

            # Decoder

            d4 = self.decoder4(de4) + de3
            d3 = self.decoder3(d4) + de2
            d2 = self.decoder2(d3) + de1
            d1 = self.decoder1(d2)

            out_s = self.finaldeconv1(d1)
            out_s = self.finalrelu1(out_s)
            out_s = self.finalconv2(out_s)
            out_s = self.finalrelu2(out_s)
            out_s = self.finalconv3(out_s)

            return out, out_s

class Our_CE_Net_(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(Our_CE_Net_, self).__init__()
        print("Semi My_version5.0")
        filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        self.deeplabv3 = DeepLab()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4

        # self.conv1 = conv1x1(112,64,1)
        self.rfb1 = BasicRFB_a(112, 48)
        self.rfb2 = BasicRFB_a(128, 48)
        self.rfb3 = BasicRFB_a(512, 48)
        self.blockconv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True)
        )

        self.blockconv0 = nn.Conv2d(48, 48, 3, padding=1, stride=1)
        self.blockconv_ = nn.Conv2d(96, 96, 3, padding=1, stride=1)
        self.blockconv1 = nn.Conv2d(48, 48, kernel_size=(7, 1), padding=(3, 0), stride=1)
        self.blockconv2 = nn.Conv2d(48, 48, kernel_size=(1, 7), padding=(0, 3), stride=1)
        self.blockconv3 = nn.Conv2d(96, 48, 1, padding=0, stride=1)

        self.assp1 = MY_ASPP(112)
        self.assp2 = MY_ASPP(512)
        self.bat1 = SAD(96)
        self.bat2 = SAD(144)
        # self.bat4 = SAD(512)

        # self.decoder4 = my_up(256, 64)
        # self.decoder3 = my_up1(128, 64)
        # self.decoder2 = my_up1(64, 32)
        # self.decoder2 = DecoderBlock(filters[0], filters[0])
        self.dsv1 = UnetDsv3(96, out_size=3, scale_factor=(416, 416))
        self.dsv2 = UnetDsv3(144, out_size=3, scale_factor=(416, 416))
        # self.dsv3 = UnetDsv3(64, out_size=3, scale_factor=(416, 416))

        self.finaldeconv1 = nn.ConvTranspose2d(144, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv4 = nn.Conv2d(32, 1, 3, padding=1)
        self.tanh = nn.Tanh();
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):

        high_level_features, low_level_features = self.deeplabv3(x)

        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        xe1 = self.encoder1(x1)
        xe2 = self.encoder2(xe1)
        xe3 = self.encoder3(xe2)

        xe1 = torch.cat([xe1, low_level_features], dim=1)
        xe1 = self.assp1(xe1)

        xe3 = torch.cat([xe3, high_level_features], dim=1)
        xe3 = self.assp2(xe3)


        xe1 = self.rfb1(xe1)
        xe2 = self.rfb2(xe2)
        xe3 = self.rfb3(xe3)

        xde3 = self.blockconv0(xe3)
        xde3 = F.interpolate(xde3, size=xe2.size()[2:], mode='bilinear', align_corners=True)
        xde3_ = self.blockconv3(torch.cat([self.blockconv1(xde3), self.blockconv2(xde3)], dim=1))
        xde3x = torch.sigmoid(xde3_) * xe2
        xde3 = self.bat1(torch.cat([xde3x, xde3], dim=1))
        out2 = self.dsv1(xde3)

        xde2 = self.blockconv0(xe2)
        xde2 = F.interpolate(xde2, size=xe1.size()[2:], mode='bilinear', align_corners=True)
        xde2_ = self.blockconv3(torch.cat([self.blockconv1(xde2), self.blockconv2(xde2)], dim=1))
        xde2x = torch.sigmoid(xde2_) * xe1
        xde3 = F.interpolate(xde3, size=xe1.size()[2:], mode='bilinear', align_corners=True)
        xde2 = self.bat2(torch.cat([xde3, xde2x], dim=1))
        out1 = self.dsv2(xde2)

        xout = self.finaldeconv1(xde2)
        xout = self.finalrelu1(xout)
        xout = self.finalconv2(xout)
        xout_s = self.finalrelu2(xout)


        out_s = self.finalconv4(xout_s)
        out_s = F.interpolate(out_s,size=(416,416),mode='bilinear',align_corners=True)
        out_s = self.tanh(out_s)

        xout = self.finalconv3(xout_s)
        xout = F.interpolate(xout, size=(416, 416), mode='bilinear', align_corners=True)

        return out2, out1, xout, out_s
class CE_Net_WithAttion(nn.Module):
    def __init__(self, num_classes=3, out_size=(416,416)):
        super(CE_Net_WithAttion, self).__init__()
        filters = [64, 128, 256, 512]
        self.out_size =out_size
        self.conv1 = nn.Conv2d(6,3,3,padding=1)
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.att1 = Attention_block(filters[2],filters[2],filters[1])

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.att2 = Attention_block(filters[1],filters[1],filters[0])

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.att3 = Attention_block(filters[0],filters[0],32)

        self.decoder1 = DecoderBlock(filters[0], filters[0])




        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        # x = torch.cat([x,self.out1(x)],dim=1)
        # x = self.conv1(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.att1(g=self.decoder4(e4),x=e3)

        d3 = self.decoder3(d4) + self.att2(g=self.decoder3(d4),x=e2)
        d2 = self.decoder2(d3) + self.att3(g=self.decoder2(d3),x=e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
class CE_Net_CBAM(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(CE_Net_CBAM, self).__init__()
        print("构造CE_Net_")
        filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.ca = ChannelAttention(filters[2])
        self.sa = SpatialAttention()

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.ca1 = ChannelAttention(filters[1])
        self.sa1 = SpatialAttention()

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.ca2 = ChannelAttention(filters[0])
        self.sa2 = SpatialAttention()

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.sa(self.ca(e3)*e3)*(self.sa(e3)*e3)
        d3 = self.decoder3(d4) + self.sa1(self.ca1(e2)*e2)*(self.sa1(e2)*e2)
        d2 = self.decoder2(d3) + self.sa2(self.ca2(e1)*e1)*(self.sa2(e1)*e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
class CE_Net_WithAttion_scale_atten_convblock(nn.Module):
    def __init__(self, num_classes=3, out_size=(416,416)):
        super(CE_Net_WithAttion_scale_atten_convblock, self).__init__()
        print("CE_Net_WithAttion_scale_atten_convblock")
        filters = [64, 128, 256, 512]
        self.out_size =out_size
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.att1 = Attention_block(filters[2],filters[2],filters[1])

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.att2 = Attention_block(filters[1],filters[1],filters[0])

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.att3 = Attention_block(filters[0],filters[0],32)

        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[0], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Sequential(
            nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1),
            nn.Upsample(size=(416,416), mode='bilinear'))

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        self.final = nn.Sequential(nn.Conv2d(4, 3, kernel_size=1))

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.att1(g=self.decoder4(e4),x=e3)
        # Deep Supervision
        # print(d4.shape)
        dsv4 = self.dsv4(d4)
        d3 = self.decoder3(d4) + self.att2(g=self.decoder3(d4),x=e2)
        dsv3 = self.dsv3(d3)
        d2 = self.decoder2(d3) + self.att3(g=self.decoder2(d3),x=e1)
        dsv2 = self.dsv2(d2)
        d1 = self.decoder1(d2)
        dsv1 = self.dsv1(d1)
        # print(dsv4.shape)
        # print(dsv3.shape)
        # print(dsv2.shape)
        # print(dsv1.shape)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)
        out = self.final(out)
        # out = self.finaldeconv1(out)
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        # out = self.finalconv3(out)

        return out




class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = self.relu(x)
        return x

class CE_Net_WithAttion_mutilpath(nn.Module):
    def __init__(self, num_classes=3, out_size=(416,416)):
        super(CE_Net_WithAttion_mutilpath, self).__init__()
        print("CE_Net_WithAttion_mutilpath")
        filters = [64, 128, 256, 512]
        self.out_size =out_size

        self.out = deeplabv3plus_resnet101()


        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.att1 = Attention_block(filters[2],filters[2],filters[1])

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.att2 = Attention_block(filters[1],filters[1],filters[0])

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.att3 = Attention_block(filters[0],filters[0],32)

        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        # self.dsv3 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        # self.dsv2 = UnetDsv3(in_size=filters[0], out_size=4, scale_factor=self.out_size)
        # self.dsv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1),
        #     nn.Upsample(size=(416,416), mode='bilinear'))
        # self.gp = nn.Sequential(
        #     nn.Conv2d(64,16,kernel_size=3,padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        # )
        # self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        #
        # self.final = nn.Sequential(nn.Conv2d(4, 3, kernel_size=1))


        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(112, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        s = self.out(x)
        # print(s.shape)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.att1(g=self.decoder4(e4),x=e3)
        # Deep Supervision
        # print(d4.shape)
        # dsv4 = self.dsv4(d4)
        # 256
        d3 = self.decoder3(d4) + self.att2(g=self.decoder3(d4),x=e2)
        # dsv3 = self.dsv3(d3)
        d2 = self.decoder2(d3) + self.att3(g=self.decoder2(d3),x=e1)
        # dsv2 = self.dsv2(d2)
        d1 = self.decoder1(d2)
        # dsv1 = self.dsv1(d1)
        # print(dsv4.shape)
        # print(dsv3.shape)
        # print(dsv2.shape)
        # print(dsv1.shape)
        # dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4,s], dim=1)
        # dsv_cat = self.gp(dsv_cat)
        # out = self.scale_att(dsv_cat)
        # out = self.final(out)
        out = F.interpolate(d1,size=(416,416),mode='bilinear', align_corners=True)
        out = torch.cat([out,s],dim=1)
        # out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


class Our_Net_V1(nn.Module):
    def __init__(self, num_classes=3):
        super(Our_Net_V1, self).__init__()
        print("Construct Our_Net_V2")
        filters = [64, 128, 256, 512]
        resnet = get_resnet_backbone('resnet34')(pretrain=True)

        self.deeplabv3 = DeepLab()
        # self.mynet = My_CE_Net_()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.block1 = MY_ASPP(64)
        self.block2 = MY_ASPP(256)
        self.block3 = MY_ASPP(512)

        self.cbam1 = CBAM(48, reduction_ratio=16)
        self.cbam2 = CBAM(256)

        self.ca1 = ChannelAttention(48)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(48)
        self.sa2 = SpatialAttention()
        # self.cbam3 = CBAM(96)
        # self.cbam4 = CBAM(144)
        self.dsv1 = UnetDsv3(96, out_size=3, scale_factor=(416,416))
        self.dsv2 = UnetDsv3(144, out_size=3, scale_factor=(416, 416))
        self.dsv3 = UnetDsv3(32, out_size=3, scale_factor=(416, 416))
        self.scale = my_scale_atten_convblock(in_size=9,out_size=4)

        # self.avg_pol = nn.AdaptiveAvgPool2d((1, 1))
        # self.sgm = nn.Sigmoid()
        self.blockconv = nn.Sequential(
            nn.Conv2d(256,48,1,bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True)
        )
        # self.conv1 = conv1x1(256*3,256)
        # self.conv2 = conv1x1(48 * 3, 48)

        self.decoderconv = nn.Sequential(
            nn.Conv2d(48,48,3,padding=1,stride=1),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True))


        # self.finaldeconv1 = nn.ConvTranspose2d(144, 32, 4, 2, 1)
        self.finaldeconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(144,32,3,padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(inplace=True)
        )
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = conv1x1(4,3,stride=1)
        # self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):

        # Encoder
        high_level_features, low_level_features = self.deeplabv3(x)
        # print("low_level_features:",low_level_features.shape)
        # print("high_level_features:", high_level_features.shape)

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        d1 = self.block1(e1)
        d2 = self.block2(e3)
        d3 = self.block3(e4)


        # g1 = self.avg_pol(d1)
        # g1 = self.blockconv(g1)
        d1 = self.blockconv(d1)
        # o1 = (low_level_features+d1)*self.sgm(g1)
        d1 = low_level_features + d1
        d1 = d1 + self.cbam1(d1)


        d2 = self.blockconv(d2)

        d3 = F.interpolate(d3, size=high_level_features.size()[2:],mode='bilinear', align_corners=True)
        # g3 = self.avg_pol(d3)
        # g3 = self.sgm(g3)
        # o3 = g3*(high_level_features+d3)
        d3 = high_level_features + d3
        d3 = d3 + self.cbam2(d3)
        d3 = self.blockconv(d3)



        p1 = self.decoderconv(d3)
        p2 = F.interpolate(d3, size=d2.size()[2:], mode='bilinear', align_corners=True)
        p3 = self.ca1(p2)*self.sa1(d2);
        p3 = F.interpolate(p3, size=d1.size()[2:],mode='bilinear', align_corners=True)

        p4 = F.interpolate(p1,size=d1.size()[2:],mode='bilinear', align_corners=True)


        p5 = torch.cat([p3,p4],dim=1)
        out3 = self.dsv1(p5)


        s = self.decoderconv(d2)
        s1 =  F.interpolate(s,size=d1.size()[2:],mode='bilinear', align_corners=True)

        s2 = self.ca2(s1)*self.sa2(d1)
        s3 = torch.cat([s2,p5],dim=1)
        out2 = self.dsv2(s3)
        # Decoder

        out = self.finaldeconv1(s3)
        out1 = self.dsv3(out)
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = torch.cat([out1,out2,out3],dim=1)
        out = self.scale(out)
        out = self.finalconv3(out)
        # out = F.interpolate(out, size=(416,416), mode='bilinear', align_corners=True)
        return [out,out1,out2,out3]

class Our_Net_V3_(nn.Module):
    def __init__(self, num_classes=3):
        super(Our_Net_V3_, self).__init__()
        print("Construct Our_Net_V3_ ")
        filters = [64, 128, 256, 512]
        resnet = get_resnet_backbone('resnet34')(pretrain=True)

        self.deeplabv3 = DeepLab()
        # self.mynet = My_CE_Net_()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.block1 = ASPP(64,atrous_rates=[1,12, 24, 36])
        # self.block2 = ASPP(256,atrous_rates=[1,12, 24, 36])
        # self.block3 = ASPP(512,atrous_rates=[1,12, 24, 36])
        self.block1 = ASPP(64)
        self.block2 = ASPP(256)
        self.block3 = ASPP(512)

        self.cbam1 = CBAM(48, reduction_ratio=16)

        self.cbam2 = CBAM(256)

        self.cbam3 = CBAM(96)
        self.cbam4 = CBAM(144)
        self.dsv1 = UnetDsv3(96, out_size=3, scale_factor=(416,416))
        self.dsv2 = UnetDsv3(144, out_size=3, scale_factor=(416, 416))
        self.scale = my_scale_atten_convblock(in_size=6, out_size=4)
        self.conv1 = conv1x1(4,3,1)

        self.blockconv = nn.Sequential(
            nn.Conv2d(256,48,1,bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True)
        )

        self.decoderconv = nn.Sequential(
            nn.Conv2d(48,48,3,padding=1,stride=1),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True))


        # self.finaldeconv1 = nn.ConvTranspose2d(144, 32, 4, 2, 1)
        self.finaldeconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(144,32,3,padding=1)
        )
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):

        # Encoder
        high_level_features, low_level_features = self.deeplabv3(x)
        # print("low_level_features:",low_level_features.shape)
        # print("high_level_features:", high_level_features.shape)

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        d1 = self.block1(e1)
        d2 = self.block2(e3)
        d3 = self.block3(e4)

        d1 = self.blockconv(d1)
        d1 = low_level_features+d1
        d1 = d1 + self.cbam1(d1)


        d2 = self.blockconv(d2)
        d3 = F.interpolate(d3, size=high_level_features.size()[2:],mode='bilinear', align_corners=True)
        d3 = high_level_features+d3
        d3 = d3 + self.cbam2(d3)
        d3 = self.blockconv(d3)



        p1 = self.decoderconv(d3)
        p2 = F.interpolate(d3, size=d2.size()[2:], mode='bilinear', align_corners=True)
        p3 = p2*d2;
        p3 = F.interpolate(p3, size=d1.size()[2:],mode='bilinear', align_corners=True)

        p4 = F.interpolate(p1,size=d1.size()[2:],mode='bilinear', align_corners=True)

        p5 = torch.cat([p3,p4],dim=1)
        p5 = p5 + self.cbam3(p5)

        out1 = self.dsv1(p5)


        s = self.decoderconv(d2)
        s1 =  F.interpolate(s,size=d1.size()[2:],mode='bilinear', align_corners=True)

        s2 = s1*d1
        s3 = torch.cat([s2,p5],dim=1)
        s3 = s3 + self.cbam4(s3)
        out2 = self.dsv2(s3)
        # Decoder
        out_Muti = torch.cat([out1,out2],dim=1)
        out_Muti = self.scale(out_Muti)
        out_Muti = self.conv1(out_Muti)
        out = self.finaldeconv1(s3)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = F.interpolate(out, size=(416,416), mode='bilinear', align_corners=True)
        out = out_Muti + out
        return [out,out1,out2]


class Our_Net_V5_(nn.Module):
    def __init__(self,in_ch,out_ch):
        print("Construct MY_NET_version3.0 ...")
        super(Our_Net_V5_, self).__init__()

        self.deeplabv3 = DeepLab()

        self.conv1 = DoubleConvn(in_ch, 32,0.3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConvn(32, 64,0.5)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConvn(64,128,0.5)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConvn(128, 256,0.6)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConvn(256, 512,0.6)

        self.block3 = MY_ASPP(512)

        self.BAt1 = SAD(32)

        self.BAt2 = SAD(64)

        self.BAt3 = SAD(128)
        self.BAt4 = SAD(256)
        self.BAt5 = SAD(512)

        self.blockconv = nn.Sequential(
            nn.Conv2d(176, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True)
        )
        self.blockconv1 = nn.Sequential(
            nn.Conv2d(768, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True)
        )

        self.decoder4 = my_up(512, 256//2)
        self.decoder3 = my_up1(256, 128//2)
        self.decoder2 = my_up1(128, 64//2)
        self.decoder1 = my_up1(64, 32)
        self.conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self,x):
        high_level_features, low_level_features = self.deeplabv3(x)

        c1=self.conv1(x)
        att1=self.BAt1(c1)

        p1=self.pool1(c1)
        c2=self.conv2(p1)
        att2=self.BAt2(c2)

        p2=self.pool2(c2)
        c3=self.conv3(p2)

        p3=self.pool3(c3)
        c4=self.conv4(p3)
        att4 = self.BAt4(c4)

        p4=self.pool4(c4)
        c5=self.conv5(p4)


        c3 = torch.cat([low_level_features,c3],dim=1) #176
        c3 = self.blockconv(c3)
        att3 = self.BAt3(c3)


        # print(high_level_features.shape)
        # print(c5.shape)
        c5 = torch.cat([high_level_features,c5],dim=1)
        c5 = self.blockconv1(c5)
        att5 = self.BAt5(c5)

        d4 = self.decoder4(att5, att4)
        # out1 = self.dsv1(d4)
        d3 = self.decoder3(d4, att3)
        # out2 = self.dsv2(d3)
        d2 = self.decoder2(d3, att2)
        # out3 = self.dsv3(d2)
        d1 = self.decoder1(d2,att1)
        out = self.conv(d1)



        return out

class My_CE_Net_(nn.Module):
    def __init__(self, num_classes=3):
        super(My_CE_Net_, self).__init__()
        print("Construct My_CE_Net_+ Mutil loss(0.6 ,0.3,0.1 ")
        filters = [64, 128, 256, 512]
        resnet = get_resnet_backbone('resnet34')(pretrain=True)

        self.deeplabv3 = DeepLab()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.block1 = ASPP(64,atrous_rates=[1,12, 24, 36])
        self.block2 = ASPP(256,atrous_rates=[1,12, 24, 36])
        self.block3 = ASPP(512,atrous_rates=[1,12, 24, 36])

        self.cbam1 = CBAM(48, reduction_ratio=16)

        self.cbam2 = CBAM(256)

        self.cbam3 = CBAM(96)
        self.cbam4 = CBAM(144)
        self.dsv1 = UnetDsv3(96, out_size=3, scale_factor=(416,416))
        self.dsv2 = UnetDsv3(144, out_size=3, scale_factor=(416, 416))


        self.blockconv = nn.Sequential(
            nn.Conv2d(256,48,1,bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True)
        )

        self.decoderconv = nn.Sequential(
            nn.Conv2d(48,48,3,padding=1,stride=1),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True))


        # self.finaldeconv1 = nn.ConvTranspose2d(144, 32, 4, 2, 1)
        self.finaldeconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(144,32,3,padding=1)
        )
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):

        # Encoder
        high_level_features, low_level_features = self.deeplabv3(x)
        # print("low_level_features:",low_level_features.shape)
        # print("high_level_features:", high_level_features.shape)

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        d1 = self.block1(e1)
        d2 = self.block2(e3)
        d3 = self.block3(e4)

        d1 = self.blockconv(d1)
        d1 = low_level_features+d1
        d1 = d1 + self.cbam1(d1)


        d2 = self.blockconv(d2)
        d3 = F.interpolate(d3, size=high_level_features.size()[2:],mode='bilinear', align_corners=True)
        d3 = high_level_features+d3
        d3 = d3 + self.cbam2(d3)
        d3 = self.blockconv(d3)



        p1 = self.decoderconv(d3)
        p2 = F.interpolate(d3, size=d2.size()[2:], mode='bilinear', align_corners=True)
        p3 = p2*d2;
        p3 = F.interpolate(p3, size=d1.size()[2:],mode='bilinear', align_corners=True)

        p4 = F.interpolate(p1,size=d1.size()[2:],mode='bilinear', align_corners=True)

        p5 = torch.cat([p3,p4],dim=1)
        p5 = p5 + self.cbam3(p5)

        out1 = self.dsv1(p5)


        s = self.decoderconv(d2)
        s1 =  F.interpolate(s,size=d1.size()[2:],mode='bilinear', align_corners=True)

        s2 = s1*d1
        s3 = torch.cat([s2,p5],dim=1)
        s3 = s3 + self.cbam4(s3)
        out2 = self.dsv2(s3)
        # Decoder

        out = self.finaldeconv1(s3)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = F.interpolate(out, size=(416,416), mode='bilinear', align_corners=True)
        return out

class Our_Net_V4_(nn.Module):
    def __init__(self, num_classes=3):
        super(Our_Net_V4_, self).__init__()
        print("Construct My_CE_Net_+ Mutil loss(0.6 ,0.3,0.1) + Boundary_Attention ")
        filters = [64, 128, 256, 512]
        resnet = get_resnet_backbone('resnet34')(pretrain=True)

        # self.deeplabv3 = DeepLab()
        # self.mynet = My_CE_Net_()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.block3 = MY_ASPP(512)

        self.bat1 = Boundary_Attention(64)
        self.bat2 = Boundary_Attention(128)
        self.bat3 = Boundary_Attention(256)
        self.bat4 = Boundary_Attention(512)



        self.up1 = up(768, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.decoder = DecoderBlock(64,64)
        # self.up4 = up(128, 32)

        self.dsv1 = UnetDsv3(256, out_size=3, scale_factor=(416, 416))
        self.dsv2 = UnetDsv3(128, out_size=3, scale_factor=(416, 416))
        self.dsv3 = UnetDsv3(64, out_size=3, scale_factor=(416, 416))
        self.dsv4 = UnetDsv3(64, out_size=3, scale_factor=(416, 416))
        self.scale = my_scale_atten_convblock(in_size=12, out_size=3)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.convlast = conv1x1(6, 3, 1)

    def forward(self, x):

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e1att = self.bat1(e1)
        e2 = self.encoder2(e1)
        e2att = self.bat2(e2)
        e3 = self.encoder3(e2)
        e3att = self.bat3(e3)

        e4 = self.encoder4(e3)
        e4 = self.block3(e4)
        e4att = self.bat4(e4)

        out = self.up1(e4att,e3att)
        out1 = self.dsv1(out)
        out = self.up2(out,e2att)
        out2 = self.dsv2(out)
        out = self.up3(out,e1att)
        out3 = self.dsv3(out)
        out = self.decoder(out)
        out4 = self.dsv4(out)



        # Decoder



        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out_Muti = torch.cat([out1, out2, out3,out4], dim=1)
        out_Muti = self.scale(out_Muti)
        out = self.convlast(torch.cat([out, out_Muti], dim=1))
        return [out,out1,out2,out3,out4]



class my_scale_atten_convblock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, drop_out=False):
        super(my_scale_atten_convblock, self).__init__()
        # if stride != 1 or in_size != out_size:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(in_size, out_size,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out
        # self.cbam = CBAM(4)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ELU(inplace=True)

        self.conv1 = conv1x1(in_size,out_size)
        self.conv3 = conv3x3(in_size, out_size)

        self.bn3 = nn.BatchNorm2d(out_size)

        # self.conv_gpb = SeparableConv2d(in_size, 256, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(out_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        # residual = self.conv1(x)
        x0 = self.max_pool(x)
        x0 = self.conv1(x0)
        x0 = self.bn_gpb(x0)
        x1 = self.avg_pool(x)
        x1 = self.conv1(x1)
        x1 = self.bn_gpb(x1)
        x2 = self.relu(self.conv1(x) * self.sg(x1) + self.conv1(x) * self.sg(x0))

        # out = self.relu(x)
        # s = self.sa(x)
        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        # print(out.shape)
        # print(self.sa(out).shape)
        # out =  out*self.ca(out)*s + residual
        out = out + x2

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


class MY_ASPP(nn.Module):
    def __init__(self, channel):
        super(MY_ASPP, self).__init__()
        self.dilate1 = SeparableConv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = SeparableConv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = SeparableConv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate4 = SeparableConv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.bn = nn.BatchNorm2d(channel)
        self.drop = nn.Dropout2d(0.5)
        self.sg = nn.Sigmoid()

        self.cbam = CBAM(channel)
        self.finalchannel = channel

        self.conv1x1_1 = SeparableConv2d(channel * 5, channel, kernel_size=1, dilation=1, padding=0)
        # self.conv1x1_2 = SeparableConv2d(channel * 3, channel * 2, kernel_size=1, dilation=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        # Master branch
        self.conv_master = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channel)
        self.conv1x1 = SeparableConv2d(channel,channel,kernel_size=1)
        # self.conv1x2 = SeparableConv2d(256, channel, kernel_size=1)
        # Global pooling branch
        self.conv_gpb = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_gpb = self.avg_pool(x)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)
        x_gpb = self.sg(x_gpb)


        x1 = self.conv1x1(x)
        x_se = x_gpb * x1

        # first block rate1
        d1 = self.dilate1(x)
        d1 = self.bn(d1)
        d1 = self.relu(d1)


        # second block rate3
        # d2 = torch.cat([d1, x], 1)
        # print("d1:",d1.shape)
        # print("x:",x1.shape)
        # d2 = self.conv1x2(d1) + x
        d2 = self.dilate2(x)
        d2 = self.bn(d2)
        d2 = self.relu(d2)
        d2_ = d2 + d1

        # third block rate5
        # d3 = torch.cat([d1, d2, x], 1)
        # d3 = self.conv1x2(d2) + x
        d3 = self.dilate3(x)
        d3 = self.bn(d3)
        d3 = self.relu(d3)
        d3_ = d3+d2

        # last block rate7
        # d4 = torch.cat([d1, d2, d3, x], 1)
        # d4 = self.conv1x2(d3) + x
        d4 = self.dilate4(x)
        d4 = self.bn(d4)
        d4 = self.relu(d4)
        d4_ = d3 + d4

        out = torch.cat([d1, d2_, d3_, d4_, x_se], 1)
        # out = out*self.ca(out)
        out = self.drop(out)
        out = self.conv1x1_1(out)
        out,_ = self.cbam(out)
        out = self.bn1(out)
        out = self.dropout(self.relu(out) + x1)
        return out

class Boundary_Attention(nn.Module):
    def __init__(self,channels):
        super(Boundary_Attention, self).__init__()
        self.cbam = CBAM(channels)
        self.conv3 = nn.Conv2d(2, 1, 1, padding=0)
        self.conv4 = nn.Conv2d(2*channels, channels, 1, padding=0)
        self.bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x1,x2 = self.cbam(x)
        x2 = torch.sigmoid(x2)
        threshold = 0.5
        p = x2.clone()
        p[p<threshold]=0
        p[p>=threshold]=1


        x3 = torch.cat([p,x2],dim=1)

        x4 = self.conv3(x3)
        x5 = self.bn(x4)
        x5 = torch.sigmoid(x5)
        mb = x5*x1
        f = self.drop(self.conv4(torch.cat([x1,mb],dim=1)) + x)

        return f

class SAD(nn.Module):
    def __init__(self, channels):
        super(SAD, self).__init__()
        self.conv2 = nn.Conv2d(channels,channels,3,padding=1)
        self.conv3 = nn.Conv2d(channels,channels,kernel_size=(7,1),padding=(3,0))
        self.conv4 = nn.Conv2d(channels,channels,kernel_size=(1,7),padding=(0,3))
        self.cbam = CBAM(channels)
        self.conv5 = nn.Conv2d(2, 1, 1, padding=0)
        self.conv6 = nn.Conv2d(2 * channels, channels, 1, padding=0)
        self.bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x0 = self.conv2(x)
        x1 = self.conv6(torch.cat([self.conv3(self.conv4(x0)),self.conv4(self.conv3(x0))],dim=1))
        x1 = torch.sigmoid(x1)
        x_ = x*x1
        x6, x2 = self.cbam(x_)
        x2 = torch.sigmoid(x2)
        threshold = 0.5
        p = x2.clone()
        p[p < threshold] = 0
        p[p >= threshold] = 1

        x3 = torch.cat([p, x2], dim=1)

        x4 = self.conv5(x3)
        x5 = self.bn(x4)
        x5 = torch.sigmoid(x5)
        mb = x5 * x1
        f = self.drop(self.conv6(torch.cat([x6, mb], dim=1)) + x_)

        return f













