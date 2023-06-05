import os
import subprocess
from tqdm import tqdm
from joblib import Parallel
from joblib import delayed


def downscale_clip(inname, outname):
    inname = '"%s"' % inname
    outname = '"%s"' % outname
    # command = "ffmpeg  -loglevel panic -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {}".format(
    #     inname, outname)
    command = f"ffmpeg -i {inname} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {outname}"
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err)
        return err.output

    return output


def downscale_clip_wrapper(file_name):
    in_name = f"{folder_path}/{file_name}"
    out_name = f"{output_path}/{file_name}"

    log = downscale_clip(in_name, out_name)
    return file_name, log


if __name__ == '__main__':
    root_path = "/home/salman/data/kinetics/400"
    split = "val"

    folder_path = f'{root_path}/{split}'
    output_path = f'{root_path}/{split}_256'
    os.makedirs(output_path, exist_ok=True)

    file_list = os.listdir(folder_path)
    completed_file_list = set(os.listdir(output_path))
    file_list = [x for x in file_list if x not in completed_file_list]

    # file_list = file_list[:100]
    print(f"Starting to downsample {len(file_list)} video files.")

    # split = len(file_list) // 10
    # list_of_lists = [file_list[x * split:(x + 1) * split] for x in range(10)]
    # list_of_lists[-1].extend(file_list[10 * split:])

    for file in tqdm(file_list):
        _, log = downscale_clip_wrapper(file)

    # status_lst = Parallel(n_jobs=16)(delayed(downscale_clip_wrapper)(row) for row in file_list)
    # status_lst = Parallel(n_jobs=16)(downscale_clip_wrapper(row) for row in file_list)
    # with open(f"{root_path}/downsample_{split}_logs.txt", "w") as fo:
    #     fo.writelines([f"{x[0], x[1]}\n" for x in status_lst])
