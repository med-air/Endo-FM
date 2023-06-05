import os
import math
import torch
from torch import nn
from stft_core.layers import ModulatedDeformConvPack as DCN



def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)




class CenterHead(torch.nn.Module):
    def __init__(self, cfg, dcn_in_channels):
        super(CenterHead, self).__init__()
        self.heads = {'hm': cfg.MODEL.CENTERNET.CHANNEL_HM,
                 'wh': cfg.MODEL.CENTERNET.CHANNEL_WH,
                 'reg': cfg.MODEL.CENTERNET.CHANNEL_OFFSET,
                 'mask': cfg.MODEL.CENTERNET.CHANNEL_HM}
        self.head_conv = cfg.MODEL.CENTERNET.HEAD_CONV
        self.dcn_in_channels = 2048
        self.dcn_out_channels = 64
        self.dcn_kernel = 4
        self.num_heads = cfg.MODEL.CENTERNET.NUM_HEADS

        dcn_block = "dcn_layers"
        num_up_layers = 3
        num_dcn_filters = [256, 128, 64]
        num_up_kernels = [4, 4, 4]
        dcn_block_module = self._make_deconv_layer(
            num_up_layers,
            num_dcn_filters,
            num_up_kernels, 
        )
        self.__setattr__(dcn_block, dcn_block_module)

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                    nn.Conv2d(64, self.head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.head_conv, classes, 
                            kernel_size=1, stride=1, 
                            padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            head_block = "head_{}".format(head)
            self.__setattr__(head_block, fc)


    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        self.inplanes = self.dcn_in_channels

        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i] 
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        assert self.num_heads == len(x)

        output = []

        for idx in range(self.num_heads):
            #dcn
            dcn_block = "dcn_layers"
            x_dcn = self.__getattr__(dcn_block)(x[idx])

            #head
            ret = {}
            for head in self.heads:
                head_block = "head_{}".format(head)
                ret[head] = self.__getattr__(head_block)(x_dcn)
            output.append(ret)

        return output



class MultiSharedCenterHead(torch.nn.Module):
    def __init__(self, cfg, dcn_in_channels):
        super(MultiSharedCenterHead, self).__init__()
        self.heads = {'hm': cfg.MODEL.CENTERNET.CHANNEL_HM,
                 'wh': cfg.MODEL.CENTERNET.CHANNEL_WH,
                 'reg': cfg.MODEL.CENTERNET.CHANNEL_OFFSET,
                 'mask': cfg.MODEL.CENTERNET.CHANNEL_HM}
        self.head_conv = cfg.MODEL.CENTERNET.HEAD_CONV
        self.dcn_in_channels = dcn_in_channels #256
        self.dcn_filters = [256, 128, 64, 64]
        self.up_kernel = 4
        self.num_heads = cfg.MODEL.CENTERNET.NUM_HEADS #4

        inplanes = self.dcn_in_channels
        for idx in range(len(self.dcn_filters)):
            dcn_block = "dcn_{}_layers".format(idx)
            planes = self.dcn_filters[idx]
            dcn_layer = DCN(inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            dcn_block_module = nn.Sequential(
                    dcn_layer,
                    nn.BatchNorm2d(planes, momentum=0.1),
                    nn.ReLU(inplace=True))
            self.__setattr__(dcn_block, dcn_block_module)
            inplanes = planes


        up_kernel, up_padding, up_output_padding = self._get_up_cfg(self.up_kernel)
        for idx in range(len(self.dcn_filters)):
            up_block = "up_{}_layers".format(idx)
            planes = self.dcn_filters[idx]
            up_layer = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=up_kernel,
                    stride=2,
                    padding=up_padding,
                    output_padding=up_output_padding,
                    bias=False)
            fill_up_weights(up_layer)
            up_block_module = nn.Sequential(
                up_layer,
                nn.BatchNorm2d(planes, momentum=0.1),
                nn.ReLU(inplace=True)
                )
            self.__setattr__(up_block, up_block_module)


        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                    nn.Conv2d(64, self.head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.head_conv, classes, 
                            kernel_size=1, stride=1, 
                            padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            head_block = "head_{}".format(head)
            self.__setattr__(head_block, fc)


    def _get_up_cfg(self, up_kernel):
        if up_kernel == 4:
            padding = 1
            output_padding = 0
        elif up_kernel == 3:
            padding = 1
            output_padding = 1
        elif up_kernel == 2:
            padding = 0
            output_padding = 0
        return up_kernel, padding, output_padding




    def forward(self, x):
        assert self.num_heads <= len(x)

        output = []
        for idx in range(self.num_heads):

            x_dcn = self.__getattr__("dcn_0_layers")(x[idx])
            if idx >= 3:
                x_dcn = self.__getattr__("up_0_layers")(x_dcn)

            x_dcn = self.__getattr__("dcn_1_layers")(x_dcn)
            if idx >= 2:
                x_dcn = self.__getattr__("up_1_layers")(x_dcn)

            x_dcn = self.__getattr__("dcn_2_layers")(x_dcn)
            if idx >= 1:
                x_dcn = self.__getattr__("up_2_layers")(x_dcn)

            x_dcn = self.__getattr__("dcn_3_layers")(x_dcn)
            x_dcn = self.__getattr__("up_3_layers")(x_dcn)

            #head
            ret = {}
            for head in self.heads:
                head_block = "head_{}".format(head)
                ret[head] = self.__getattr__(head_block)(x_dcn)

            output.append(ret)

        return output