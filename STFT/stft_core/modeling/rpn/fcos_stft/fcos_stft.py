import os
import math
import copy
import torch
import torch.nn.functional as F
from torch import nn

from stft_core.layers import Scale
from stft_core.layers import DFConv2d
from stft_core.layers import DeformConv
from .inference import make_fcos_stft_postprocessor
from .loss import make_fcos_stft_loss_evaluator


class SFTBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, deformable_groups=4):
        super(SFTBranch, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        offset_input_channel = 4
        self.conv_offset = nn.Conv2d(offset_input_channel, deformable_groups * offset_channels, 1, bias=False)
        # self.conv_adaption = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        # self.conv_adaption = DFConv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=kernel_size,
        #     # padding=(kernel_size - 1) // 2,
        #     deformable_groups=deformable_groups
        # )
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, offset_weight_std=0.01):
        torch.nn.init.normal_(self.conv_offset.weight, mean=0, std=offset_weight_std)
        if hasattr(self.conv_offset, 'bias') and self.conv_offset.bias is not None:
            torch.nn.init.constant_(self.conv_offset.bias, 0)
        torch.nn.init.normal_(self.conv_adaption.weight, mean=0, std=0.01)
        if hasattr(self.conv_adaption, 'bias') and self.conv_adaption.bias is not None:
            torch.nn.init.constant_(self.conv_adaption.bias, 0)

    def forward(self, feature, pred_shape):
        pred_shape = pred_shape.permute(0, 2, 1).reshape(feature.shape[0], -1, feature.shape[2], feature.shape[3])
        with torch.no_grad():  #
            offset = self.conv_offset(pred_shape)
        # offset = self.conv_offset(pred_shape.detach())
        feature = self.relu(self.conv_adaption(feature.contiguous(), offset.contiguous()))
        return feature


class STFTFCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(STFTFCOSHead, self).__init__()
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.add_module("dcn_cls_subnet", SFTBranch(in_channels, in_channels))
        self.add_module("dcn_bbox_subnet", SFTBranch(in_channels, in_channels))

        self.dcn_cls_score = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.dcn_bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                if isinstance(l, nn.GroupNorm):
                    torch.nn.init.constant_(l.weight, 1)
                    torch.nn.init.constant_(l.bias, 0)

        self.dcn_cls_subnet.init_weights(offset_weight_std=cfg.MODEL.STFT.OFFSET_WEIGHT_STD)
        self.dcn_bbox_subnet.init_weights(offset_weight_std=cfg.MODEL.STFT.OFFSET_WEIGHT_STD)

        for modules in [self.dcn_cls_score, self.dcn_bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.dcn_cls_score.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])


    def forward(self, x, shifts):
        logits = []
        bbox_reg = []
        centerness = []
        stft_logits = []
        stft_bbox_reg = []
        pre_bbox = []

        shifts = [
            torch.cat([shi.unsqueeze(0) for shi in shift], dim=0) 
            for shift in list(zip(*shifts))
        ]

        for l, (feature, shifts_i) in enumerate(zip(x, shifts)):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            # only record target frame
            cls_logits = self.cls_logits(cls_tower)
            logits.append(cls_logits[0].unsqueeze(0))

            if self.centerness_on_reg:
                centerness_logits = self.centerness(box_tower)
            else:
                centerness_logits = self.centerness(cls_tower)
            centerness.append(centerness_logits[0].unsqueeze(0))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred) * self.fpn_strides[l]
            else:
                bbox_pred = torch.exp(bbox_pred) * self.fpn_strides[l]
            bbox_reg.append(bbox_pred[0].unsqueeze(0))

            ###### STFT
            N, C, H, W = feature.shape
            pre_off = bbox_pred.clone().detach()
            with torch.no_grad():
                pre_off = pre_off.permute(0, 2, 3, 1).reshape(N, -1, 4) #l,t,r,b
                pre_boxes = self.compute_bbox(shifts_i, pre_off) #x1,y1,x2,y2
                pre_bbox.append(pre_boxes[0].unsqueeze(0))

                #align on feature map scale
                align_boxes, wh = self.compute_border(pre_boxes, l, H, W) #x1,y1,x2,y2,  w,h
                align_ltrb_off = self.compute_ltrb_off(align_boxes, shifts_i, l, H, W) #l,t,r,b

            align_boxes[1:] = align_boxes[0] - align_boxes[1:]
            wh[1:] = wh[0] - wh[1:]
            align_ltrb_off[1:] = align_ltrb_off[0] - align_ltrb_off[1:]

            # STFT -- classification
            stft_cls_feats = self.dcn_cls_subnet(cls_tower, align_ltrb_off)
            # channel-aware
            target_stft_cls_feats = stft_cls_feats[0].unsqueeze(0).permute(1, 0, 2, 3)
            target_stft_cls_feats = target_stft_cls_feats.reshape(stft_cls_feats.shape[1], 1, -1)
            support_stft_cls_feats = stft_cls_feats[1:].permute(1, 0, 2, 3)
            support_stft_cls_feats = support_stft_cls_feats.reshape(stft_cls_feats.shape[1], -1, stft_cls_feats.shape[2]*stft_cls_feats.shape[3])
            sim_stft_cls = torch.bmm(target_stft_cls_feats, support_stft_cls_feats.transpose(1, 2)) 
            sim_stft_cls = (1.0 / math.sqrt(float(support_stft_cls_feats.shape[2]))) * sim_stft_cls
            sim_stft_cls = F.softmax(sim_stft_cls, dim=2)
            att_stft_cls = torch.bmm(sim_stft_cls, support_stft_cls_feats)

            target_stft_cls_feats = target_stft_cls_feats + att_stft_cls
            target_stft_cls_feats = target_stft_cls_feats.reshape(stft_cls_feats.shape[1], stft_cls_feats.shape[2], stft_cls_feats.shape[3])
            target_stft_cls_feats = target_stft_cls_feats.unsqueeze(0)
            
            stft_cls_logits = self.dcn_cls_score(target_stft_cls_feats)
            stft_logits.append(stft_cls_logits)

            # STFT -- regression
            stft_reg_feats = self.dcn_bbox_subnet(box_tower, align_ltrb_off)
            # channel-aware
            target_stft_reg_feats = stft_reg_feats[0].unsqueeze(0).permute(1, 0, 2, 3)
            target_stft_reg_feats = target_stft_reg_feats.reshape(stft_reg_feats.shape[1], 1, -1)
            support_stft_reg_feats = stft_reg_feats[1:].permute(1, 0, 2, 3)
            support_stft_reg_feats = support_stft_reg_feats.reshape(stft_reg_feats.shape[1], -1, stft_reg_feats.shape[2]*stft_reg_feats.shape[3])
            sim_stft_reg = torch.bmm(target_stft_reg_feats, support_stft_reg_feats.transpose(1, 2)) 
            sim_stft_reg = (1.0 / math.sqrt(float(support_stft_reg_feats.shape[2]))) * sim_stft_reg
            sim_stft_reg = F.softmax(sim_stft_reg, dim=2)
            att_stft_reg = torch.bmm(sim_stft_reg, support_stft_reg_feats)

            target_stft_reg_feats = target_stft_reg_feats + att_stft_reg
            target_stft_reg_feats = target_stft_reg_feats.reshape(stft_reg_feats.shape[1], stft_reg_feats.shape[2], stft_reg_feats.shape[3])
            target_stft_reg_feats = target_stft_reg_feats.unsqueeze(0)

            stft_bbox_reg_pred = self.dcn_bbox_pred(target_stft_reg_feats)
            stft_bbox_reg.append(stft_bbox_reg_pred)

        if self.training:
            pre_bbox = torch.cat(pre_bbox, dim=1)
        return logits, bbox_reg, centerness, stft_logits, stft_bbox_reg, pre_bbox


    def compute_bbox(self, location, pred_offset):
        detections = torch.stack([
            location[:, :, 0] - pred_offset[:, :, 0],
            location[:, :, 1] - pred_offset[:, :, 1],
            location[:, :, 0] + pred_offset[:, :, 2],
            location[:, :, 1] + pred_offset[:, :, 3]], dim=2)

        return detections

    def compute_border(self, _boxes, fm_i, height, width):
        boxes = _boxes / self.fpn_strides[fm_i]
        boxes[:, :, 0].clamp_(min=0, max=width - 1)
        boxes[:, :, 1].clamp_(min=0, max=height - 1)
        boxes[:, :, 2].clamp_(min=0, max=width - 1)
        boxes[:, :, 3].clamp_(min=0, max=height - 1)

        wh = (boxes[:, :, 2:] - boxes[:, :, :2]).contiguous()
        return boxes, wh

    def compute_ltrb_off(self, align_boxes, location, fm_i, height, width):
        align_loc = location / self.fpn_strides[fm_i]
        align_loc[:, :, 0].clamp_(min=0, max=width - 1)
        align_loc[:, :, 1].clamp_(min=0, max=height - 1)

        align_ltrb = torch.stack([
            align_boxes[:, :, 0] - align_loc[:, :, 0],
            align_boxes[:, :, 1] - align_loc[:, :, 1],
            align_boxes[:, :, 2] - align_loc[:, :, 0],
            align_boxes[:, :, 3] - align_loc[:, :, 1]], dim=2)
        return align_ltrb


class STFTFCOSModule(torch.nn.Module):
    """
    Module for STFTFCOS computation. Takes feature maps from the backbone and
    STFTFCOS outputs and losses. Only Test on FPN now.
    """
    def __init__(self, cfg, in_channels):
        super(STFTFCOSModule, self).__init__()
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.head = STFTFCOSHead(cfg, in_channels)
        self.box_selector_test = make_fcos_stft_postprocessor(cfg)
        self.loss_evaluator = make_fcos_stft_loss_evaluator(cfg)

    def forward(self, images, features, targets=None):
        locations = self.compute_locations(features)
        shifts = [
            copy.deepcopy(locations)
            for _ in range(images.shape[0])
        ]

        box_cls, box_regression, centerness, stft_box_cls, stft_box_reg, stft_based_box = self.head(features, shifts)

        if self.training:
            return self._forward_train(
                [shifts[0]], box_cls, box_regression, centerness, targets,
                stft_based_box, stft_box_cls, stft_box_reg
            )
        else:
            return self._forward_test(
                [shifts[0]], box_cls, centerness, 
                stft_box_cls, stft_box_reg, stft_based_box, [(images.shape[2], images.shape[3])])

    def _forward_train(self, shifts, box_cls, box_regression, centerness, targets, stft_based_box, stft_box_cls, stft_box_reg):
        loss_cls, loss_box_reg, loss_centerness, loss_stft_cls, loss_stft_reg = self.loss_evaluator(
            shifts, box_cls, box_regression, centerness, targets, stft_based_box, stft_box_cls, stft_box_reg
        )
        losses = {
            "loss_cls": loss_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_stft_cls": loss_stft_cls,
            "loss_stft_reg": loss_stft_reg
        }
        return None, losses

    def _forward_test(self, shifts, box_cls, centerness, stft_box_cls, stft_box_reg, stft_based_box, image_sizes):
        boxes = self.box_selector_test(
            shifts, box_cls, centerness, stft_box_cls, stft_box_reg, stft_based_box, image_sizes)
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fcos_stft(cfg, in_channels):
    return STFTFCOSModule(cfg, in_channels)
