# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
"""
Implements the STFT framework
"""

import torch
from torch import nn
import numpy
from PIL import Image
from collections import deque
from stft_core.structures.image_list import to_image_list
from torchvision.utils import make_grid
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn


class GeneralizedRCNNSTFT(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNSTFT, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.test_ref_num = cfg.MODEL.VID.STFT.TEST_REF_NUM
        self.test_all_frame = -cfg.MODEL.VID.STFT.MIN_OFFSET + cfg.MODEL.VID.STFT.MAX_OFFSET + 1
        self.key_frame_location = -cfg.MODEL.VID.STFT.MIN_OFFSET
        self.device = cfg.MODEL.DEVICE
        self.call_id = 0

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            return self._forward_train(images, targets)
        else:
            images["cur"] = to_image_list(images["cur"])
            images["ref"] = [to_image_list(image) for image in images["ref"]]
            infos = images.copy()
            infos.pop("cur")
            self.call_id += 1
            return self._forward_test(images["cur"], infos)


    def _forward_train(self, images, targets):
        images["cur"] = to_image_list(images["cur"])
        images["ref"] = [to_image_list(image) for image in images["ref"]]

        concat_imgs = torch.cat([images["cur"].tensors, *[img_ref.tensors for img_ref in images["ref"]]], dim=0)
        concat_imgs = concat_imgs.permute(1, 0, 2, 3).unsqueeze(0)
        # print(concat_imgs.shape)
        # exit(0)
        concat_features = self.backbone(concat_imgs)
        # print(concat_features.shape)

        _, proposal_losses = self.rpn(concat_imgs, concat_features, targets)
        return proposal_losses


    def _forward_test(self, imgs, infos):
        def update_feature(img=None, feats=None):
            assert (img is not None) or (feats is not None)
            if img is not None:
                # feats = self.backbone(img)
                feats = self.backbone(img.permute(1, 0, 2, 3).unsqueeze(0))
            self.feats.append(feats)

        if infos["frame_category"] == 0 or self.call_id == 1:  # a new video
            self.seg_len = infos["seg_len"]
            self.end_id = infos["start_id"]
            self.feats = deque(maxlen=self.test_all_frame)
            self.imgs = deque(maxlen=self.test_all_frame)

            # feats_cur = self.backbone(imgs.tensors)
            feats_cur = self.backbone(imgs.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            while len(self.feats) < self.key_frame_location + 1:
                update_feature(None, feats_cur)
                # self.imgs.append(imgs.tensors)
                self.imgs.append(imgs.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            while len(self.feats) < self.test_all_frame:
                self.end_id = min(self.end_id + 1, self.seg_len)
                end_filename = infos["pattern"] % self.end_id
                end_image = Image.open(infos["img_dir"] % end_filename).convert("RGB")
                end_image = infos["transforms"](end_image)
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                end_image = end_image.view(1, *end_image.shape).to(self.device)
                update_feature(end_image)
                self.imgs.append(end_image)

        elif infos["frame_category"] == 1:
            self.end_id = min(self.end_id + 1, self.seg_len)
            end_image = infos["ref"][0].tensors
            update_feature(end_image)
            self.imgs.append(end_image)

        total_features = [torch.cat(list(feat), dim=0) for feat in list(zip(*self.feats))]
        
        offsets_left = numpy.random.choice(range(0, self.key_frame_location), int(self.test_ref_num/2), replace=False)
        offsets_right = numpy.random.choice(range(self.key_frame_location+1, self.test_all_frame), int(self.test_ref_num/2), replace=False)
        sample_features = []
        for i in range(len(total_features)):
            sample_features.append(torch.cat((total_features[i][self.key_frame_location].unsqueeze(0),
                                             total_features[i][offsets_left,:], 
                                             total_features[i][offsets_right,:]), dim=0))

        proposals, _ = self.rpn(imgs.tensors, sample_features)
        return proposals
