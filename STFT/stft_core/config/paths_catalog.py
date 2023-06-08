# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy

class DatasetCatalog(object):
    DATA_DIR = "./"
    DATASETS = {
        ####################### ImageNet VID #######################
        "DET_train_30classes": {
            "img_dir": "ILSVRC2015/Data/DET",
            "anno_path": "ILSVRC2015/Annotations/DET",
            "img_index": "ILSVRC2015/ImageSets/DET_train_30classes.txt"
        },
        "VID_train_15frames": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_train_15frames.txt"
        },
        "VID_val_videos": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_val_videos.txt"
        },
        ####################### KUMC #######################
        "CVCVideo_train_videos": {
            "img_dir": "../data/downstream/KUMC/processed/train2019/Image",
            "anno_path": "../data/downstream/KUMC/processed/train2019/Annotation",
            "img_index": "../data/downstream/KUMC/processed/ImageSets/train.txt"
        },
        "CVCVideo_val_videos": {
            "img_dir": "../data/downstream/KUMC/processed/val2019/Image",
            "anno_path": "../data/downstream/KUMC/processed/val2019/Annotation",
            "img_index": "../data/downstream/KUMC/processed/ImageSets/val.txt"
        },
        ####################### polyp ASUMayo #######################
        "ASUVideo_train_videos": {
            "img_dir": "ASUVideo/Data",
            "anno_path": "ASUVideo/Annotations",
            "img_index": "ASUVideo/ImageSets/ASUVideo_train_videos.txt"
        },
        "ASUVideo_val_videos": {
            "img_dir": "ASUVideo/Data",
            "anno_path": "ASUVideo/Annotations",
            "img_index": "ASUVideo/ImageSets/ASUVideo_val_videos.txt"
        }
    }

    @staticmethod
    def get(name, method="base"):
        dataset_dict = {
            "base": "VIDDataset",
            "rdn": "VIDRDNDataset",
            "mega": "VIDMEGADataset",
            "fgfa": "VIDFGFADataset",
            "stft": "VIDSTFTDataset",
            "cvc_image": "CVCVIDImageDataset",
            "cvc_rdn": "CVCVIDRDNDataset",
            "cvc_mega": "CVCVIDMEGADataset",
            "cvc_fgfa": "CVCVIDFGFADataset",
            "cvc_stft": "CVCVIDSTFTDataset"
        }
        if ("DET" in name) or ("VID" in name) or ("Video" in name):
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                image_set=name,
                data_dir=data_dir,
                img_dir=os.path.join(data_dir, attrs["img_dir"]),
                anno_path=os.path.join(data_dir, attrs["anno_path"]),
                img_index=os.path.join(data_dir, attrs["img_index"])
            )
            return dict(
                factory=dataset_dict[method],
                args=args,
            )

class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
