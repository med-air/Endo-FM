import os
import glob
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

datadir = '/research/d4/gds/zwang21/svt/data/pretrain/KUMC/PolypsSet'
savedir = '/research/d4/gds/zwang21/svt/data/pretrain/KUMC/processed'


def process_single(video):
    images = os.listdir(f'{datadir}/{split}2019/Image/{video}')
    idx = 1
    for image in images:
        video_save_dir = f'{datadir}/{split}2019/Image/{video}'.replace(datadir, savedir)

        image_save_path = os.path.join(video_save_dir, '{}.jpg'.format(idx))
        anno_save_path = os.path.join(video_save_dir, '{}.jpg'.format(idx)).replace('Image', 'Annotation').\
            replace('.jpg', '.xml')

        anno_ori_path = f'{datadir}/{split}2019/Annotation/{video}/{image[:-4]}.xml'
        if not os.path.exists(anno_ori_path):
            continue
        try:
            cv2.imread(f'{datadir}/{split}2019/Image/{video}/{image}')
        except:
            continue

        try:
            os.makedirs(video_save_dir, exist_ok=True)
            os.makedirs(video_save_dir.replace('Image', 'Annotation'), exist_ok=True)

            cv2.imwrite(image_save_path, cv2.imread(f'{datadir}/{split}2019/Image/{video}/{image}'))
            os.system(f'cp {anno_ori_path} {anno_save_path}')
        except:
            print(f'{datadir}/{split}2019/Image/{video}/{image}')
            continue

        idx += 1


splits = ['train', 'val', 'test']
# for split in splits:
#     videos = os.listdir(f'{datadir}/{split}2019/Image')
#     for video in videos:
#         index_video_frames(video)
#         exit(0)


for split in splits:
    print(split)
    videos = os.listdir(f'{datadir}/{split}2019/Image')
    n_jobs = 30
    parallel_func = Parallel(n_jobs=n_jobs, backend='threading')
    parallel_func(delayed(process_single)(video) for video in tqdm(videos))


# for video in videos:
#     print(f'Processing {video}...')
#     images = os.listdir(os.path.join(datadir, dataset, video))
#     n_jobs = 20
#     Parallel(n_jobs=n_jobs)(delayed(index_frame)(video, image, idx + 1) for idx, image in tqdm(enumerate(images)))
