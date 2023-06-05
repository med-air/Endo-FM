import glob
from datasets.video_container import get_video_container
from datasets.decoder import decode
from tqdm import tqdm
import json

root_path = "/home/kanchanaranasinghe/data/kinetics400"
root_path = "/home/kanchanaranasinghe/repo/mmaction2/data/hmdb51/videos"
split = "hmdb"

file_list = glob.glob(f"{root_path}/{split}_256/*.mp4")
file_list = glob.glob(f"{root_path}/*/*.avi")
good_count = 0
frames = "0"
bad_list = []
for file in tqdm(file_list):
    try:
        container = get_video_container(file, True)
        frames = decode(container, 32, 8)
        assert frames is not None, "frames is None"
    except Exception as e:
        print(e, file)
        bad_list.append(file)
    else:
        good_count += 1

print(f"{len(file_list) - good_count} files bad. {good_count} / {len(file_list)} files are good.")
json.dump({"bad": bad_list}, open(f"{root_path}/{split}_256_bad.json", "w"))

# import os
# import json
# import shutil
# import pandas as pd
#
# root_path = "/home/kanchanaranasinghe/repo/mmaction2/data/hmdb51/splits"
# txt_file_name = "/home/kanchanaranasinghe/repo/mmaction2/data/hmdb51/splits/hmdb51_val_split_1_videos.txt"
# files = json.load(open(f"{root_path}/{split}_256_bad.json", "r"))
# bad_names = [x[59:] for x in files['bad']]
# df = pd.read_csv(f"{txt_file_name}", sep=" ",
#                  header=None)
# df = df[df[0].isin(bad_names) == False]
# df.to_csv(f"{txt_file_name}", sep=" ",
#           header=None, index=None)

# for file in files["bad"]:
#     shutil.move(file, file.replace(f"/{split}_256/", f"/{split}_256_bad/"))
