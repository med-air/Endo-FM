import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import csv
import glob


datadir = 'pretrain'


public_video = glob.glob(f'{datadir}/public_pretrain_videos_clips_5s/*.mp4')
private_video = glob.glob(f'{datadir}/private_pretrain_videos_clips_5s/*.mp4')

videolist = public_video + private_video


videos = []
for video in tqdm(videolist):
    videos.append([f'{video}', -1])

print(len(videos))

with open(f"{datadir}/train.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(videos)
