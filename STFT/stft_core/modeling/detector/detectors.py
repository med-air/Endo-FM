# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_rcnn_rdn import GeneralizedRCNNRDN
from .generalized_rcnn_mega import GeneralizedRCNNMEGA
from .generalized_rcnn_fgfa import GeneralizedRCNNFGFA
from .generalized_rcnn_stft import GeneralizedRCNNSTFT

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "GeneralizedRCNNRDN": GeneralizedRCNNRDN,
                                 "GeneralizedRCNNMEGA": GeneralizedRCNNMEGA,
                                 "GeneralizedRCNNFGFA": GeneralizedRCNNFGFA,
                                 "GeneralizedRCNNSTFT": GeneralizedRCNNSTFT}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
