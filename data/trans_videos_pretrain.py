import os
import cv2
cv2.setNumThreads(1)
from joblib import Parallel, delayed
from tqdm import tqdm

datadir = 'pretrain'
savedir = 'pretrain/public_pretrain_videos_clips_5s'


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


def process_single(video_info):
    video_info = video_info.split(' ')

    video_path = video_info[0]
    start_index = int(video_info[1])
    total_length = int(video_info[2])

    first_frame_path = os.path.join(datadir, video_path, 'img_{:05}.jpg'.format(start_index + 1))
    ori_height, ori_width, _ = cv2.imread(first_frame_path).shape

    fps = 30
    min_scale = min(ori_width, ori_height)
    width = int(ori_width / min_scale * 256)
    height = int(ori_height / min_scale * 256)

    clip_length = 150
    clip_num = total_length // clip_length

    for n in range(clip_num):
        if n == (clip_num - 1): clip_length = total_length - n * clip_length

        video_name = '-'.join(video_path.split('/')) + f'-{start_index}-{start_index + clip_length}' + '.mp4'
        save_path = os.path.join(savedir, video_name)
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


videolist = read_txt(os.path.join(datadir, 'sun_seg.txt'))
print(len(videolist))

n_jobs = 10
Parallel(n_jobs=n_jobs)(delayed(process_single)(os.path.join(video_info)) for video_info in tqdm(videolist))
