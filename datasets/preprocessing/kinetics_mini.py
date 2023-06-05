import pandas as pd

# root_path = "/home/kanchanaranasinghe/data/kinetics400"
# anno_folder = "k400-mini"
#
# for split in ["train", "val",]:
#
#     save_path = f"{root_path}/{anno_folder}/{split}.csv"
#     file_list = pd.read_csv(save_path, sep=" ")
#
#     if split == "train":
#         file_list = file_list.sample(n=60000, replace=False)
#         new_save_path = f"{root_path}/{anno_folder}/{split}_60k.csv"
#         file_list.to_csv(new_save_path, index=False, header=False, sep=" ")
#     elif split == "val":
#         file_list = file_list.sample(n=10000, replace=False)
#         new_save_path = f"{root_path}/{anno_folder}/{split}_10k.csv"
#         file_list.to_csv(new_save_path, index=False, header=False, sep=" ")


file_path = "/home/kanchanaranasinghe/data/kinetics400/k400-mini/train_60k.csv"
df = pd.read_csv(file_path, header=None, sep=" ")
print(f"{len(df[1].unique())} unique classes")
df[1].hist(bins=len(df[1].unique()))
