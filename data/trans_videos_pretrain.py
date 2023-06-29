import os
import cv2
cv2.setNumThreads(1)

from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import shutil


datadir = 'pretrain/'
savedir = 'pretrain/'

datasets = {
    'SUN-SEG': 0,
}


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


def read_csv(file):
    import csv

    data = []
    with open(file, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            data.append(row)
    f.close()

    return data


def save_csv(data, file):
    with open(file, 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in data.items()]


def process_single(video_info, dataset, index_count, mapping_dict):
    video_info = video_info.split(' ')

    video_path = video_info[0]
    start_index = int(video_info[1])
    total_length = int(video_info[2])

    first_frame_path = os.path.join(datadir, video_path, 'img_{:05}.jpg'.format(start_index + 1))
    ori_height, ori_width, _ = cv2.imread(first_frame_path).shape

    fps = 30
    clip_length = 150
    min_scale = min(ori_width, ori_height)
    width = int(ori_width / min_scale * 256)
    height = int(ori_height / min_scale * 256)

    if total_length < clip_length:
        clip_length = total_length
        clip_num = 1
    else:
        clip_num = total_length // clip_length

    for n in range(clip_num):
        if n == (clip_num - 1): clip_length = total_length - n * clip_length

        index_count += 1
        video_name = '-'.join(video_path.split('/')) + f'-{start_index}-{start_index + clip_length}' + '.mp4'
        save_name = '{:05}.mp4'.format(index_count)
        mapping_dict[save_name] = video_name

        save_path = os.path.join(savedir, dataset, save_name)
        videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

        for i in range(start_index, start_index + clip_length):
            try:
                frame = cv2.imread(os.path.join(datadir, video_path, 'img_{:05}.jpg'.format(i + 1)))
                frame = cv2.resize(frame, (width, height))

                videoWriter.write(frame)
            except:
                continue

        videoWriter.release()

        start_index += clip_length

    return index_count


for dataset in datasets.keys():
    os.makedirs(os.path.join(savedir, dataset), exist_ok=True)
    videolist = read_txt(os.path.join(datadir, 'sun_seg.txt'))

    mapping_dict = {}
    index_count = 0
    for video_info in tqdm(videolist, desc=f'Processing {dataset}'):
        index_count = process_single(video_info, dataset, index_count, mapping_dict)

    print(dataset, len(mapping_dict))
    save_csv(mapping_dict, f'data/pretrain/public_pretrain_videos_clips_5s/{dataset}_data_mapping.csv')
