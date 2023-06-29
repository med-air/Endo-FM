import os
import cv2
cv2.setNumThreads(1)

from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import shutil


datadir = 'pretrain/'

datasets = {
    'Colonoscopic': 0,
    'SUN-SEG': 0,
    'LDPolypVideo': 0,
    'Hyper-Kvasir': 0,
    'Kvasir-Capsule': 0,
    'CholecT45': 0,
}


for dataset in datasets.keys():
    videolist = glob.glob(f'{datadir}/{dataset}/*.mp4')

    frame_number = 0

    for video in videolist:
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            print(video, 'video not opened')
            continue

        ori_width, ori_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, total_length = int(cap.get(5)), int(cap.get(7))

        frame_number += total_length

        cap.release()

    print(dataset, frame_number)
