import os
import pandas as pd
from tqdm import tqdm
import json

root_path = "/home/kanchanaranasinghe/data/kinetics400"
split = "val"
anno_folder = "new_annotations"

video_list = set(os.listdir(f"{root_path}/{split}_d256"))
save_path = f"{root_path}/{anno_folder}/{split}.csv"
os.makedirs(f"{root_path}/{anno_folder}", exist_ok=True)

labels = pd.read_csv(f"{root_path}/annotations/{split}.csv")
label_dict = {y: x for x, y in enumerate(sorted(labels.label.unique().tolist()))}
json.dump(label_dict, open(f"{root_path}/new_annotations/{split}_label_dict.json", "w"))

with open(f"{root_path}/bad_files_{split}.txt", "r") as fo:
    bad_videos = fo.readlines()
bad_videos = [x.strip() for x in bad_videos]

video_label = []
for idx, row in tqdm(labels.iterrows()):
    video_name = f"{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4"
    if video_name in bad_videos:
        continue
    if video_name not in video_list:
        continue
    assert os.path.exists(f"{root_path}/{split}_d256/{video_name}"), "video not found"
    video_label.append((video_name, label_dict[row.label]))

os.makedirs(os.path.dirname(save_path), exist_ok=True)
pd.DataFrame(video_label).to_csv(save_path, index=False, header=False, sep=" ")
