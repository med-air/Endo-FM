import os
import glob
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

datadir = 'pretrain'
dataset = 'SUN-SEG'

videos1 = [os.path.join('SUN-SEG-Annotation/TrainDataset/Frame', _)
           for _ in os.listdir(os.path.join(datadir, dataset, 'SUN-SEG-Annotation/TrainDataset/Frame'))]
videos2 = [os.path.join('SUN-SEG-Annotation/TestEasyDataset/Seen/Frame', _)
           for _ in os.listdir(os.path.join(datadir, dataset, 'SUN-SEG-Annotation/TestEasyDataset/Seen/Frame'))]
videos3 = [os.path.join('SUN-SEG-Annotation/TestEasyDataset/Unseen/Frame', _)
           for _ in os.listdir(os.path.join(datadir, dataset, 'SUN-SEG-Annotation/TestEasyDataset/Unseen/Frame'))]
videos4 = [os.path.join('SUN-SEG-Annotation/TestHardDataset/Seen/Frame', _)
           for _ in os.listdir(os.path.join(datadir, dataset, 'SUN-SEG-Annotation/TestHardDataset/Seen/Frame'))]
videos5 = [os.path.join('SUN-SEG-Annotation/TestHardDataset/Unseen/Frame', _)
           for _ in os.listdir(os.path.join(datadir, dataset, 'SUN-SEG-Annotation/TestHardDataset/Unseen/Frame'))]

videos = videos1 + videos2 + videos3 + videos4 + videos5
# images = glob.glob(os.path.join(datadir, dataset, 'SUN-Negative/*/*.jpg'))
# images = sorted(images, key=lambda x: int(x.split('/')[-1].split('_')[-1][5:-4]))
print(len(videos))
# print(images[0])
# exit(0)


def index_frame(video, image, idx):
    video_name = video.split('/')[-1]
    save_dir = os.path.join(datadir, dataset, 'indexed', video_name)

    # print(image, video_name, idx, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, 'img_{:05}.jpg'.format(idx))):
        try:
            cv2.imwrite(os.path.join(save_dir, 'img_{:05}.jpg'.format(idx)),
                        cv2.imread(os.path.join(datadir, dataset, video, image)))
        except:
            print(os.path.join(datadir, dataset, video, image))


# for video in videos:
#     images = os.listdir(os.path.join(datadir, dataset, video))
#     for idx, image in tqdm(enumerate(images)):
#         index_frame(video, image, idx + 1)
#         exit(0)


for video in videos:
    print(f'Processing {video}...')
    images = os.listdir(os.path.join(datadir, dataset, video))
    n_jobs = 20
    Parallel(n_jobs=n_jobs)(delayed(index_frame)(video, image, idx + 1) for idx, image in tqdm(enumerate(images)))
