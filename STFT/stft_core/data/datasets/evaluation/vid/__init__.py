import logging
import numpy as np
import cv2
import os
import torch

from .vid_eval import do_vid_evaluation

def vid_evaluation(dataset, predictions, output_folder, box_only, motion_specific, vis_thr, **_):
    logger = logging.getLogger("stft_core.inference")
    logger.info("performing vid evaluation, ignored iou_types.")
    return do_vid_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        box_only=box_only,
        motion_specific=motion_specific,
        logger=logger,
        vis_thr=vis_thr,
    )
