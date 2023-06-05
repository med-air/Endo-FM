import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from joblib import Parallel, delayed


datadir = '/research/d4/gds/zwang21/svt/data/pretrain/KUMC/PolypsSet/train2019_backup'
savedir = '/research/d4/gds/zwang21/svt/data/pretrain/KUMC/PolypsSet/train2019'
# tree = ET.parse(self._anno_path % filename).getroot()
# anno = self._preprocess_annotation(tree)


anno_files = os.listdir(os.path.join(datadir, 'Annotation'))
print(len(anno_files))


def process_single(anno_file):
    tree = ET.parse(os.path.join(datadir, 'Annotation', anno_file)).getroot()
    # folder = tree.find('folder').text
    # image = tree.find('filename').text

    # print(anno_file, folder, image)

    # ori_image_path = os.path.join(datadir, 'Image', anno_file.replace('.xml', '.jpg'))
    # ori_anno_path = os.path.join(datadir, 'Annotation', anno_file)
    #
    # new_image_path = os.path.join(savedir, 'Image', folder, image)
    # new_anno_path = os.path.join(savedir, 'Annotation', folder, image.replace('.jpg', '.xml'))

    # os.makedirs(os.path.join(savedir, 'Image', folder), exist_ok=True)
    # os.makedirs(os.path.join(savedir, 'Annotation', folder), exist_ok=True)
    #
    # os.system(f'cp {ori_image_path} {new_image_path}')
    # os.system(f'cp {ori_anno_path} {new_anno_path}')

    # print(ori_image_path)
    # print(ori_anno_path)
    # print(new_image_path)
    # print(new_anno_path)
    #
    #
    # exit(0)

    path = tree.find('path').text.split('/')[-2]

    return path


videos = []
for anno_file in tqdm(anno_files):
    video = process_single(anno_file)
    if video not in videos:
        videos.append(video)

print(videos)
print(len(videos))


# n_jobs = 30
# parallel_func = Parallel(n_jobs=n_jobs, backend='threading')
# parallel_func(delayed(process_single)(os.path.join(anno_file)) for anno_file in tqdm(anno_files))
