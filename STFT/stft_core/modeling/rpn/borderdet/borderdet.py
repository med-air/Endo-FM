import os
import math
import copy
import torch
import torch.nn.functional as F
from torch import nn

from stft_core.layers import Scale
from stft_core.layers import DFConv2d
from stft_core.layers.border_align import BorderAlign

from .inference import make_border_postprocessor
from .loss import make_border_loss_evaluator


class BorderBranch(nn.Module):
    def __init__(self, in_channels, border_channels):
        """
        :param in_channels:
        """
        super(BorderBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                border_channels,
                kernel_size=1),
            nn.InstanceNorm2d(border_channels),
            nn.ReLU())

        self.ltrb_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                border_channels * 4,
                kernel_size=1),
            nn.InstanceNorm2d(border_channels * 4),
            nn.ReLU())

        self.border_align = BorderAlign(pool_size=10)

        self.border_conv = nn.Sequential(
            nn.Conv2d(
                5 * border_channels,
                in_channels,
                kernel_size=1),
            nn.ReLU())

    def forward(self, feature, boxes, wh):
        N, C, H, W = feature.shape

        fm_short = self.cur_point_conv(feature)
        feature = self.ltrb_conv(feature)
        ltrb_conv = self.border_align(feature, boxes)
        ltrb_conv = ltrb_conv.permute(0, 3, 1, 2).reshape(N, -1, H, W)
        align_conv = torch.cat([ltrb_conv, fm_short], dim=1)
        align_conv = self.border_conv(align_conv)
        return align_conv



class BorderHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        implementation from https://github.com/Megvii-BaseDetection/BorderDet
        """
        super(BorderHead, self).__init__()
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

        # border
        self.add_module("border_cls_subnet", BorderBranch(in_channels, 256))
        self.add_module("border_bbox_subnet", BorderBranch(in_channels, 128))

        self.border_cls_score = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1)
        self.border_bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=1, stride=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,self.centerness,
                        self.border_cls_subnet, self.border_bbox_subnet,
                        self.border_cls_score, self.border_bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                if isinstance(l, nn.GroupNorm):
                    torch.nn.init.constant_(l.weight, 1)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB 
        bias_value = -math.log((1 - prior_prob) / prior_prob) 
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.border_cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))]) 

    def forward(self, x, shifts):
        logits = []
        bbox_reg = []
        centerness = []
        border_logits = []
        border_bbox_reg = []
        pre_bbox = []

        shifts = [
            torch.cat([shi.unsqueeze(0) for shi in shift], dim=0) 
            for shift in list(zip(*shifts))
        ]

        for l, (feature, shifts_i) in enumerate(zip(x, shifts)):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg: 
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred) * self.fpn_strides[l]
            else:
                bbox_pred = torch.exp(bbox_pred) * self.fpn_strides[l]
            bbox_reg.append(bbox_pred)

            # border
            N, C, H, W = feature.shape
            pre_off = bbox_pred.clone().detach()
            with torch.no_grad():
                pre_off = pre_off.permute(0, 2, 3, 1).reshape(N, -1, 4)
                pre_boxes = self.compute_bbox(shifts_i, pre_off)
                align_boxes, wh = self.compute_border(pre_boxes, l, H, W)
                pre_bbox.append(pre_boxes)

            border_cls_conv = self.border_cls_subnet(cls_tower, align_boxes, wh)
            border_cls_logits = self.border_cls_score(border_cls_conv)
            border_logits.append(border_cls_logits)

            border_reg_conv = self.border_bbox_subnet(box_tower, align_boxes, wh)
            border_bbox_pred = self.border_bbox_pred(border_reg_conv)
            border_bbox_reg.append(border_bbox_pred)

        if self.training:
            pre_bbox = torch.cat(pre_bbox, dim=1)
        return logits, bbox_reg, centerness, border_logits, border_bbox_reg, pre_bbox


    def compute_bbox(self, location, pred_offset):
        detections = torch.stack([
            location[:, :, 0] - pred_offset[:, :, 0],
            location[:, :, 1] - pred_offset[:, :, 1],
            location[:, :, 0] + pred_offset[:, :, 2],
            location[:, :, 1] + pred_offset[:, :, 3]], dim=2)

        return detections

    def compute_border(self, _boxes, fm_i, height, width):
        """
        :param _boxes:
        :param fm_i:
        :param height:
        :param width:
        :return:
        """
        boxes = _boxes / self.fpn_strides[fm_i]
        boxes[:, :, 0].clamp_(min=0, max=width - 1)
        boxes[:, :, 1].clamp_(min=0, max=height - 1)
        boxes[:, :, 2].clamp_(min=0, max=width - 1)
        boxes[:, :, 3].clamp_(min=0, max=height - 1)

        wh = (boxes[:, :, 2:] - boxes[:, :, :2]).contiguous()
        return boxes, wh



class BorderModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(BorderModule, self).__init__()
        self.head = BorderHead(cfg, in_channels)
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

        self.box_selector_test = make_border_postprocessor(cfg)
        self.loss_evaluator = make_border_loss_evaluator(cfg)



    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        locations = self.compute_locations(features)
        shifts = [
            copy.deepcopy(locations)
            for _ in range(images.tensors.shape[0])
        ]

        box_cls, box_regression, centerness, bd_box_cls, bd_box_reg, bd_based_box = self.head(features, shifts)

        if self.training:
            return self._forward_train(
                shifts, box_cls, 
                box_regression, 
                centerness, targets,
                bd_based_box, bd_box_cls, bd_box_reg
            )
        else:
            return self._forward_test(
                box_cls, centerness, bd_box_cls, 
                bd_box_reg, bd_based_box, 
                images.image_sizes)


    def _forward_train(self, shifts, box_cls, box_regression, centerness, targets, bd_based_box, bd_box_cls, bd_box_reg):
        loss_cls, loss_box_reg, loss_centerness, loss_border_cls, loss_border_reg = self.loss_evaluator(
            shifts, box_cls, box_regression, centerness, targets, bd_based_box, bd_box_cls, bd_box_reg
        )
        losses = {
            "loss_cls": loss_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_border_cls": loss_border_cls,
            "loss_border_reg": loss_border_reg,
        }
        return None, losses


    def _forward_test(self, box_cls, box_center, border_cls, border_delta, bd_based_box, image_sizes):
        boxes = self.box_selector_test(
            box_cls, box_center, border_cls, border_delta, bd_based_box, image_sizes
        )
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


def build_border(cfg, in_channels):
    return BorderModule(cfg, in_channels)
