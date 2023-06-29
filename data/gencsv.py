import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import csv
import glob


datadir = 'pretrain'

videolist = glob.glob(f'{datadir}/*.mp4')


videos = []
for video in tqdm(videolist):
    videos.append([f'{video}', -1])

print(len(videos))

with open(f"{datadir}/train.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(videos)
