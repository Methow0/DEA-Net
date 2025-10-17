"""
This script defines the structure of MyNet

Author: YU
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from CE_Net import *
from DeepLabv3_plus import DeepLabv3_plus
from DefEDNetmain.DefEDNet import Ladder_ASPP, SeparableConv2d
from DefEDNetmain.defconv import DefC
from backbones.scale_attention_layer import scale_atten_convblock
from models import DeepLab
from smatunetmodels.layers import DepthwiseSeparableConv, CBAM
from smatunetmodels.unet_parts_depthwise_separable import UpDS, DoubleConvDS
from torchvision.ops import DeformConv2d
nonlinearity = partial(F.elu, inplace=True)
"""
    U-Net卷积模块
"""
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
"""
    Attention-UNet中的attention模块
"""
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            SeparableConv2d(F_g,F_int,1,stride=1,padding=0,bias=True),
            # nn.Conv2d(F_g,F_int,1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.w_x = nn.Sequential(
            SeparableConv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            SeparableConv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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
"""
    CE-Net中的空洞卷积模块
"""
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = SeparableConv2d(channel, channel, kernel_size=3, dilation=1, padding=1,bias=True)
        self.dilate2 = SeparableConv2d(channel, channel, kernel_size=3, dilation=3, padding=3,bias=True)
        self.dilate3 = SeparableConv2d(channel, channel, kernel_size=3, dilation=5, padding=5,bias=True)
        self.conv1x1 = SeparableConv2d(channel, channel, kernel_size=1, dilation=1, padding=0,bias=True)
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
"""
    CE-Net中的金字塔池化模块
"""
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConvDSM(in_channels, out_channels,kernels_per_layer=2)
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out), out

class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(DConv, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
        #                        stride=stride, padding=padding, bias=bias)
        self.conv1 = SeparableConv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out

class DoubleConvDSM(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleDefConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleDefConv, self).__init__()
        self.conv = nn.Sequential(
            DConv(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DConv(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Single_level_densenet(nn.Module):
    def __init__(self, filters,num_conv=4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        filter1 = int(filters*0.5)
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filter1,filters,3,padding=1))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            # nn.Conv2d(
            #     input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            # )
            SeparableConv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            # nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            SeparableConv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        self.conv_skip = nn.Sequential(
            SeparableConv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            # nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        return self.down_sample(self.conv_block(x)+self.conv_skip(x)),self.conv_block(x) + self.conv_skip(x)

"""
    Unet_Ladder_ASPP 四层Unet加上空间空洞池化
"""
class Unet_Ladder_ASPP(nn.Module):
    def __init__(self,in_ch,out_ch):
        print("Constructing UNetwithDesConv_Ladder_ASPP model...")
        super(Unet_Ladder_ASPP, self).__init__()


        self.conv1 = DoubleDefConv(in_ch,64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.5)

        self.conv2 = DoubleDefConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)


        self.conv3 = DoubleDefConv(128,256)
        self.pool3 = nn.MaxPool2d(2)


        # self.conv4 = DoubleConvDS(256, 512, kernels_per_layer=2)
        self.conv4 = DoubleDefConv(256, 512)

        self.dblock = Ladder_ASPP(512)

        self.up6 = nn.ConvTranspose2d(1024, 512, 1,stride=1)
        self.att6 = Attention_block(512,512,256)
        self.conv6 = DoubleDefConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = Attention_block(256,256,128)
        self.conv7 = DoubleDefConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = Attention_block(128,128,64)
        self.conv8 = DoubleDefConv(256, 128)



        self.up9 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.att9= Attention_block(64,64,32)
        self.conv9 = DoubleDefConv(128,64)

        self.conv10 = DConv(64,out_ch,kernel_size=1,padding=0)
        # self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self,x):
        c1=self.conv1(x)
        c1=self.drop(c1)

        p1=self.pool1(c1)
        c2=self.conv2(p1)
        c2=self.drop(c2)

        p2=self.pool2(c2)
        c3=self.conv3(p2)
        c3=self.drop(c3)

        p3=self.pool3(c3)
        c4=self.conv4(p3)
        c4=self.drop(c4)  # 512 52 52

        c5=self.dblock(c4)  #1024 52 52



        up_6= self.up6(c5)  # 512 104  104
        c4 = self.att6(g=up_6,x=c4)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        c6=self.drop(c6)

        up_7=self.up7(c6)
        c3 = self.att7(g=up_7,x=c3)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        c7=self.drop(c7)

        up_8=self.up8(c7)
        c2 = self.att8(g=up_8,x=c2)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        c8=self.drop(c8)

        up_9 = self.up9(c8)
        c1 = self.att9(g=up_9, x=c1)
        merge8 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge8)
        c9=self.drop(c9)

        c10=self.conv10(c9)


        return c10
"""
    Unet_DesConv_Attetion_DAC 四层Unet加上空洞卷积DAC、Attetion、深度可分离卷积
"""
class Unet_DesConv_Attetion_DAC(nn.Module):
    def __init__(self,in_ch,out_ch,kernels_per_layer=2,out_size=(416,416)):
        print("Consturct Unet_DesConv_Attetion_DAC ...")
        super(Unet_DesConv_Attetion_DAC, self).__init__()
        self.kernels_per_layer = kernels_per_layer
        self.out_size = out_size
        self.conv1 = DoubleConvDS(in_ch, 64,kernels_per_layer=kernels_per_layer)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConvDS(64, 128,kernels_per_layer=kernels_per_layer)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConvDS(128, 256,kernels_per_layer=kernels_per_layer)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConvDS(256, 512,kernels_per_layer=kernels_per_layer)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConvDS(512, 1024,kernels_per_layer=kernels_per_layer)
        self.dblock = DACblock(1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att6 = Attention_block(512,512,256)
        self.conv6 = DoubleConvDS(1024, 512,kernels_per_layer=kernels_per_layer)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = Attention_block(256,256,128)
        self.conv7 = DoubleConvDS(512, 256,kernels_per_layer=kernels_per_layer)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = Attention_block(128,128,64)
        self.conv8 = DoubleConvDS(256, 128,kernels_per_layer=kernels_per_layer)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att9 = Attention_block(64,64,32)
        self.conv9 = DoubleConvDS(128, 64,kernels_per_layer=kernels_per_layer)

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=512, out_size=4, scale_factor=self.out_size)
        # self.dsv3 = UnetDsv3(in_size=256, out_size=4, scale_factor=self.out_size)
        # self.dsv2 = UnetDsv3(in_size=128, out_size=4, scale_factor=self.out_size)
        # self.dsv1 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1)

        # self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # self.final = nn.Sequential(nn.Conv2d(4, 3, kernel_size=1))
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        c5=self.dblock(c5)

        up_6= self.up6(c5)
        c4 = self.att6(g=up_6,x=c4)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        # dsv4 = self.dsv4(c6)

        up_7=self.up7(c6)
        c3 = self.att7(g=up_7,x=c3)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        # dsv3 = self.dsv3(c7)

        up_8=self.up8(c7)
        c2 = self.att8(g=up_8,x=c2)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        # dsv2 = self.dsv2(c8)

        up_9=self.up9(c8)
        c1 = self.att9(g=up_9,x=c1)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        # dsv1 = self.dsv1(c9)

        # dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        # out = self.scale_att(dsv_cat)
        # out = self.final(out)
        c10=self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        return c10
"""
    Unet_DesConv_Attetion_DAC 四层Unet加上空洞卷积DAC、Attetion、深度可分离卷积、残差块
"""
class Unet_ResBlock_DesConv_Attetion(nn.Module):
    def __init__(self,in_ch,out_ch,kernels_per_layer=2):
        print("Constructing Unet_ResBlock_DesConv_Attetion...")
        super(Unet_ResBlock_DesConv_Attetion, self).__init__()
        self.kernels_per_layer = kernels_per_layer

        self.resblock1 = ResBlock(in_ch, 64)
        self.resblock2 = ResBlock(64, 128)
        self.resblock3 = ResBlock(128, 256)
        self.resblock4 = ResBlock(256, 512)
        self.resblock5 = ResBlock(512,1024)



        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att6 = Attention_block(512,512,256)
        self.conv6 = DoubleConvDS(1024, 512,kernels_per_layer=kernels_per_layer)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = Attention_block(256,256,128)
        self.conv7 = DoubleConvDS(512, 256,kernels_per_layer=kernels_per_layer)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = Attention_block(128,128,64)
        self.conv8 = DoubleConvDS(256, 128,kernels_per_layer=kernels_per_layer)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att9 = Attention_block(64,64,32)
        self.conv9 = DoubleConvDS(128, 64,kernels_per_layer=kernels_per_layer)

        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        x, x0_0 = self.resblock1(x)
        x, x1_0 = self.resblock2(x)
        x, x2_0 = self.resblock3(x)
        x, x3_0 = self.resblock4(x)
        _, x4_0 = self.resblock5(x)

        up_6= self.up6(x4_0)
        c4 = self.att6(g=up_6,x=x3_0)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)

        up_7=self.up7(c6)
        c3 = self.att7(g=up_7,x=x2_0)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)

        up_8=self.up8(c7)
        c2 = self.att8(g=up_8,x=x1_0)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)

        up_9=self.up9(c8)
        c1 = self.att9(g=up_9,x=x0_0)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        return c10

class Unet_ResBlock_DesConv_Attetion_DAC(nn.Module):
    def __init__(self,in_ch,out_ch,kernels_per_layer=2):
        print("Constructing Unet_ResBlock_DesConv_Attetion_DAC model...")
        super(Unet_ResBlock_DesConv_Attetion_DAC, self).__init__()

        self.resblock1 = ResBlock(in_ch, 64)
        self.resblock2 = ResBlock(64, 128)
        self.resblock3 = ResBlock(128, 256)
        self.resblock4 = ResBlock(256, 512)
        self.resblock5 = ResBlock(512, 1024)

        self.dblock = DACblock(1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att6 = Attention_block(512,512,256)
        self.conv6 = DoubleConvDS(1024, 512,kernels_per_layer=kernels_per_layer)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = Attention_block(256,256,128)
        self.conv7 = DoubleConvDS(512, 256,kernels_per_layer=kernels_per_layer)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = Attention_block(128,128,64)
        self.conv8 = DoubleConvDS(256, 128,kernels_per_layer=kernels_per_layer)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att9 = Attention_block(64,64,32)
        self.conv9 = DoubleConvDS(128, 64,kernels_per_layer=kernels_per_layer)

        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        x, x0_0 = self.resblock1(x)
        x, x1_0 = self.resblock2(x)
        x, x2_0 = self.resblock3(x)
        x, x3_0 = self.resblock4(x)
        _, x4_0 = self.resblock5(x)
        c5=self.dblock(x4_0)

        up_6= self.up6(c5)
        c4 = self.att6(g=up_6,x=x3_0)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)

        up_7=self.up7(c6)
        c3 = self.att7(g=up_7,x=x2_0)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)

        up_8=self.up8(c7)
        c2 = self.att8(g=up_8,x=x1_0)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)

        up_9=self.up9(c8)
        c1 = self.att9(g=up_9,x=x0_0)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        return c10
"""
    Unet_DesConv_Attetion_CBAM_DAC 四层Unet加上空洞卷积DAC、Attetion、深度可分离卷积、CBAM
"""
class Unet_DesConv_Attetion_CBAM_DAC(nn.Module):
    def __init__(self, in_ch, out_ch, kernels_per_layer=2):
        print("Constructing Unet_DesConv_Attetion_CBAM_DAC model...")
        super(Unet_DesConv_Attetion_CBAM_DAC, self).__init__()
        self.kernels_per_layer = kernels_per_layer

        self.conv1 = DoubleConvDS(in_ch, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConvDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=16)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConvDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=16)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConvDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=16)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConvDS(512, 1024, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024, reduction_ratio=16)
        self.dblock = DACblock(1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att6 = Attention_block(512, 512, 256)
        self.conv6 = DoubleConvDS(1024, 512, kernels_per_layer=kernels_per_layer)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = Attention_block(256, 256, 128)
        self.conv7 = DoubleConvDS(512, 256, kernels_per_layer=kernels_per_layer)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = Attention_block(128, 128, 64)
        self.conv8 = DoubleConvDS(256, 128, kernels_per_layer=kernels_per_layer)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att9 = Attention_block(64, 64, 32)
        self.conv9 = DoubleConvDS(128, 64, kernels_per_layer=kernels_per_layer)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        x1Att = self.cbam1(c1)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        x2Att = self.cbam2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        x3Att = self.cbam3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        x4Att = self.cbam4(c4)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        c5 = self.dblock(c5)
        x5Att = self.cbam5(c5)

        up_6 = self.up6(x5Att)
        c4 = self.att6(g=up_6, x=x4Att)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c3 = self.att7(g=up_7, x=x3Att)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c2 = self.att8(g=up_8, x=x2Att)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c1 = self.att9(g=up_9, x=x1Att)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        return c10

class Unet_ResidualConv_DesConv_Attetion_DAC(nn.Module):
    def __init__(self,in_ch,out_ch,kernels_per_layer=2):
        print("Constructing Unet_ResidualConv_DesConv_Attetion_DAC + Deeplabv3Ecoder model...")
        super(Unet_ResidualConv_DesConv_Attetion_DAC, self).__init__()

        self.deeplabv3 = DeepLab()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SeparableConv2d(32,32,kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.residual_conv_1 = ResidualConv(32, 64, 1, 1)
        self.residual_conv_2 = ResidualConv(64,128, 1, 1)
        self.residual_conv_3 = ResidualConv(128,256, 1, 1)

        self.residual_conv_4 = ResidualConv(256, 512,1,1)

        self.block1 = ASPP(128, atrous_rates=[1, 12, 24, 36])
        self.block2 = ASPP(256, atrous_rates=[1, 12, 24, 36])
        self.block3 = ASPP(512, atrous_rates=[1, 12, 24, 36])

        self.blockconv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoderconv = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))

        self.finaldeconv1 = nn.ConvTranspose2d(144, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self,x):
        high_level_features,low_level_features=self.deeplabv3(x)
        d = self.input_layer(x) + self.input_skip(x)
        x = self.pool1(d)
        x, x0_0 = self.residual_conv_1(x)
        x, x1_0 = self.residual_conv_2(x)
        x, x2_0 = self.residual_conv_3(x)
        x, x3_0 = self.residual_conv_4(x)

        x1_0 = self.block1(x1_0)
        x1_0 = self.blockconv(x1_0)
        x2_0 = self.block2(x2_0)
        x3_0 = self.block3(x3_0)

        x1_0 = x1_0 + low_level_features
        x3_0 = x3_0 + high_level_features

        x2_0 = self.blockconv(x2_0)
        x3_0 = self.blockconv(x3_0)

        d3 = self.decoderconv(x3_0)
        d2 = F.interpolate(d3, size=x2_0.size()[2:], mode='bilinear', align_corners=True)
        p1 = x2_0*d2;
        p2 = F.interpolate(d3, size=x1_0.size()[2:], mode='bilinear', align_corners=True)

        p4 = F.interpolate(p1, size=x1_0.size()[2:], mode='bilinear', align_corners=True)

        p5 = torch.cat([p2, p4], dim=1)

        f2 = self.decoderconv(x2_0)
        s1 = F.interpolate(f2, size=x1_0.size()[2:], mode='bilinear', align_corners=True)

        s2 = s1 * x1_0
        s3 = torch.cat([s2, p5], dim=1)
        out = self.finaldeconv1(s3)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = F.interpolate(out, size=(416, 416), mode='bilinear', align_corners=True)
        return out


class Unet_DesConv_Attetion_DAC_pool3(nn.Module):
    def __init__(self,in_ch,out_ch,kernels_per_layer=2):
        print("Consturct Unet_DesConv_Attetion_DAC  + deeplabv3+ ...")
        super(Unet_DesConv_Attetion_DAC_pool3, self).__init__()
        self.kernels_per_layer = kernels_per_layer
        # self.out = DeepLabv3_plus()
        self.conv1 = ACRLM(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ACRLM(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ACRLM(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ACRLM(256, 512)
        self.dblock = DACblock(512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att6 = Attention_block(256,256,128)
        self.conv6 = SeparableConv2d(512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att7 = Attention_block(128,128,64)
        self.conv7 = SeparableConv2d(256, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att8 = Attention_block(64,64,32)
        self.conv8 = SeparableConv2d(128, 64)


        self.conv9 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):

        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)

        c4=self.conv4(p3)
        c5=self.dblock(c4)

        up_6= self.up6(c5)
        c3 = self.att6(g=up_6,x=c3)
        merge6 = torch.cat([up_6,c3],dim=1)
        c6=self.conv6(merge6)
        # dsv4 = self.dsv4(c6)

        up_7=self.up7(c6)
        c2 = self.att7(g=up_7,x=c2)
        merge7 = torch.cat([up_7, c2], dim=1)
        c7=self.conv7(merge7)
        # dsv3 = self.dsv3(c7)

        up_8=self.up8(c7)
        c1 = self.att8(g=up_8,x=c1)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        # dsv2 = self.dsv2(c8)
        c9=self.conv9(c8)
        # out = nn.Sigmoid()(c10)
        return c9



class ACRLM(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ACRLM, self).__init__()
        self.down = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.dilate1 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True))
        self.dilate2 = nn.Sequential(
            SeparableConv2d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.dilate3 = nn.Sequential(
            SeparableConv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.dilate4 = nn.Sequential(
            SeparableConv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True))
        self.conv1x1 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=1, dilation=1, padding=0,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True))

    def forward(self, x):
        out1 = self.conv1x1(x)
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        out =  dilate1_out+dilate2_out+dilate3_out+dilate4_out
        return self.down(out+out1)




def MyNet():
    mode1 = Unet_ResidualConv_DesConv_Attetion_DAC(3,3)
    return mode1

