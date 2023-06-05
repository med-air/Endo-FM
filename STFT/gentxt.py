import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

# datadir = 'data/pretrain'
# savedir = 'data/pretrain/all_pretrain_videos'

# datadir = 'data/pretrain/PolypDet/videos'
# savedir = 'data/pretrain/PolypDet/splits'

datadir = '/research/d4/gds/zwang21/svt/data/pretrain/KUMC/processed'
savedir = '/research/d4/gds/zwang21/svt/data/pretrain/KUMC/processed/ImageSets'


def save_txt(data, file):
    f = open(file, "w")
    for line in data:
        f.write(line + '\n')
    f.close()


def read_txt(file):
    tmp = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            tmp.append(line)

    return tmp


annotations = {'train': [], 'val': [], 'test': []}


for split, lst in annotations.items():
    videos = os.listdir(f'{datadir}/{split}2019/Image')

    for video in videos:
        frames = os.listdir(f'{datadir}/{split}2019/Image/{video}')
        frame_number = len(frames)

        for i in range(frame_number):
            msg = f'{video} 0 {i+1} {frame_number}'
            lst.append(msg)


print(len(annotations['train']), len(annotations['val']), len(annotations['test']))
save_txt(annotations['train'], f'{savedir}/train.txt')
save_txt(annotations['val'], f'{savedir}/val.txt')
save_txt(annotations['test'], f'{savedir}/test.txt')
