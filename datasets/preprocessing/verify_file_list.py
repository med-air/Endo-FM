import pandas as pd
from tqdm import tqdm

from datasets.decoder import decode
from datasets.video_container import get_video_container

root_path = "/home/kanchanaranasinghe/data/kinetics400"

for split in ["train", "val", "test"]:
    anno_folder = "new_annotations"

    if split == "train":
        save_path = f"{root_path}/{anno_folder}/{split}_60k.csv"
        file_list = pd.read_csv(save_path, sep=" ")
    elif split in ["val", "test"]:
        save_path = f"{root_path}/{anno_folder}/{split}_10k.csv"
        file_list = pd.read_csv(save_path, sep=" ")
    else:
        raise NotImplementedError("invalid split")

    good_count = 0
    frames = "0"
    bad_list = []
    print(f"Processing files from: {save_path}")
    for file in tqdm(file_list.values[:, 0]):
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
