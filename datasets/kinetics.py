# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import os
import random
import warnings
from PIL import Image
import torch
import torch.utils.data
import torchvision
import kornia

from datasets.transform import resize
from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets.decoder import decode
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationDINO
from einops import rearrange


class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10, get_flow=False):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        if get_flow:
            assert mode == "train", "invalid: flow only for train mode"
        self.get_flow = get_flow

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        print("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                # print(path_label, self.cfg.DATA.PATH_LABEL_SEPARATOR, len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)))
                # exit(0)
                assert (
                        len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                        == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        print(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index]
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                        self._spatial_temporal_idx[index]
                        % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                     + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                print(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                warnings.warn(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decode(
                container=video_container,
                sampling_rate=sampling_rate,
                num_frames=self.cfg.DATA.NUM_FRAMES,
                clip_idx=temporal_sample_index,
                num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                temporal_aug=self.mode == "train" and not self.cfg.DATA.NO_RGB_AUG,
                two_token=self.cfg.MODEL.TWO_TOKEN,
                rand_fr=self.cfg.DATA.RAND_FR
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                # warnings.warn(
                #     "Failed to decode video idx {} from {}; trial {}".format(
                #         index, self._path_to_videos[index], i_try
                #     )
                # )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            if self.mode in ["test", "val"] or self.cfg.DATA.NO_RGB_AUG:
                # Perform color normalization.
                frames = tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )

                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)

                # Perform data augmentation.
                frames = spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

                if not self.cfg.MODEL.ARCH in ['vit']:
                    frames = pack_pathway_output(self.cfg, frames)
                else:
                    # Perform temporal sampling from the fast pathway.
                    frames = torch.index_select(
                        frames,
                        1,
                        torch.linspace(
                            0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                        ).long(),
                    )

            else:
                # T H W C -> T C H W.
                frames = [rearrange(x, "t h w c -> t c h w") for x in frames]

                # Perform data augmentation.
                augmentation = VideoDataAugmentationDINO()
                frames = augmentation(frames, from_list=True, no_aug=self.cfg.DATA.NO_SPATIAL,
                                      two_token=self.cfg.MODEL.TWO_TOKEN)

                # T C H W -> C T H W.
                frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

                # Perform temporal sampling from the fast pathway.
                frames = [torch.index_select(
                    x,
                    1,
                    torch.linspace(
                        0, x.shape[1] - 1, x.shape[1] if self.cfg.DATA.RAND_FR else self.cfg.DATA.NUM_FRAMES

                    ).long(),
                ) for x in frames]

            meta_data = {}
            if self.get_flow:
                assert self.mode == "train", "flow only for train"
                try:
                    flow_path = self._path_to_videos[index].replace("train_d256", "train_flow")[:-4]
                    flow_tensor = self.get_flow_from_folder(flow_path)
                    flow_tensor = kornia.filters.sobel(flow_tensor)
                    if self.cfg.DATA.NO_FLOW_AUG:
                        flow_tensor = resize(flow_tensor, size=self.cfg.DATA.CROP_SIZE, mode="bicubic")
                        flow_tensor = [x for x in flow_tensor]
                    else:
                        flow_tensor = augmentation(flow_tensor)
                        flow_tensor = [rearrange(x, "t c h w -> c t h w") for x in flow_tensor]
                    meta_data["flow"] = flow_tensor
                except Exception as e:
                    print(e)
                    continue
            return frames, label, index, meta_data

        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    @staticmethod
    def get_flow_from_folder(dir_path):
        flow_image_list = sorted(glob.glob(f"{dir_path}/*.jpg"))
        flow_image_list = [Image.open(im_path) for im_path in flow_image_list]
        flow_image_list = [torchvision.transforms.functional.to_tensor(im_path) for im_path in flow_image_list]
        return torch.stack(flow_image_list, dim=0)


if __name__ == '__main__':

    # import torch
    # from timesformer.datasets import Kinetics
    from utils.parser import parse_args, load_config
    from tqdm import tqdm

    args = parse_args()
    args.cfg_file = "/home/kanchanaranasinghe/repo/timesformer/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    config = load_config(args)
    config.DATA.PATH_TO_DATA_DIR = "/home/kanchanaranasinghe/data/kinetics400/new_annotations"
    # config.DATA.PATH_TO_DATA_DIR = "/home/kanchanaranasinghe/data/kinetics400/k400-mini"
    config.DATA.PATH_PREFIX = "/home/kanchanaranasinghe/data/kinetics400"
    # dataset = Kinetics(cfg=config, mode="val", num_retries=10)
    dataset = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=True)
    print(f"Loaded train dataset of length: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
    for idx, i in enumerate(dataloader):
        print([x.shape for x in i[0]], i[1:3], [x.shape for x in i[3]['flow']])
        break

    do_vis = False
    if do_vis:
        from PIL import Image
        from transform import undo_normalize

        vis_path = "/home/kanchanaranasinghe/data/kinetics400/vis/spatial_aug"

        for aug_idx in range(len(i[0])):
            temp = i[0][aug_idx][3].permute(1, 2, 3, 0)
            temp = undo_normalize(temp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            for idx in range(temp.shape[0]):
                im = Image.fromarray(temp[idx].numpy())
                im.resize((224, 224)).save(f"{vis_path}/aug_{aug_idx}_fr_{idx:02d}.jpg")
