# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from thop import profile

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from utils import utils
import vision_transformer as vits
from vision_transformer import DINOHead, MultiDINOHead

from datasets import Kinetics
from datasets.rand_conv import RandConv
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D, S3D
from utils.parser import load_config
# from eval_knn import extract_features, knn_classifier, UCFReturnIndexDataset, HMDBReturnIndexDataset

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('SVT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small', 'timesformer',
                                 'swin'] + torchvision_archs,
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--pretrained_rgb', default=None, type=str, help='Path to pretrained RGB model.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    # online knn eval
    parser.add_argument('--eval_batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=5, type=int, help='Number of NN to use. We use 5 for online.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')

    return parser


def train_svt(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    config = load_config(args)
    if utils.is_main_process():
        json.dump(vars(args), open(Path(args.output_dir) / "config.txt", "w"), indent=4)
    config.DATA.PATH_TO_DATA_DIR = args.data_path
    # config.DATA.PATH_PREFIX = os.path.dirname(args.data_path)
    dataset = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=config.DATA.USE_FLOW)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train data loaded: there are {len(dataset)} images.")

    if config.DATA.RAND_CONV:
        rand_conv = RandConv(temporal_input=True).cuda()
    else:
        rand_conv = None

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch == "timesformer":
        if config.MODEL.TWO_TOKEN:
            student = get_aux_token_vit(cfg=config, no_head=True)
            teacher = get_aux_token_vit(cfg=config, no_head=True)
        else:
            student = get_vit_base_patch16_224(cfg=config, no_head=True)
            teacher = get_vit_base_patch16_224(cfg=config, no_head=True)
        embed_dim = student.embed_dim

        if args.pretrained_rgb is not None:
            state_dict = torch.load(args.pretrained_rgb)["teacher"]
            state_dict = {x[len("backbone."):]: y for x, y in state_dict.items() if x.startswith("backbone.")}
            msg = student.load_state_dict(state_dict)
            print(f"Loaded pretrained rgb student: {msg}")
            msg = teacher.load_state_dict(state_dict)
            print(f"Loaded pretrained rgb teacher: {msg}")

        if config.MODEL.TWO_STREAM:
            motion_student = vits.vit_small()
            motion_teacher = vits.vit_small()
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            msg = motion_student.load_state_dict(state_dict)
            print(f"Loaded motion-student with status: {msg}")
            msg = motion_teacher.load_state_dict(state_dict)
            print(f"Loaded motion-teacher with status: {msg}")
            motion_embed_dim = motion_student.embed_dim
        else:
            motion_student = None
            motion_teacher = None
            motion_embed_dim = None

    if args.arch == "swin":
        student = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
        teacher = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])

        embed_dim = 1024
        print("Loaded swin transformer network")

        motion_student = None
        motion_teacher = None
        motion_embed_dim = None

    elif args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    if config.MODEL.CNN_DISTILL:
        cnn_model = S3D()
        cnn_ckpt = torch.load("checkpoints/pretrained/CoCLR-k400-rgb-128-s3d.pth.tar")
        new_ckpt = {}
        for k, v in cnn_ckpt['state_dict'].items():  # only take the encoder_q
            if k.startswith("encoder_q.0."):
                new_ckpt[k[12:]] = v
        msg = cnn_model.load_state_dict(new_ckpt, strict=False)
        print(f"Loaded cnn model with msg: {msg}")

        cnn_model = cnn_model.cuda()
        cnn_model = nn.SyncBatchNorm.convert_sync_batchnorm(cnn_model)
        cnn_model = nn.parallel.DistributedDataParallel(cnn_model, device_ids=[args.gpu], find_unused_parameters=False)
    else:
        cnn_model = None

    # multi-crop wrapper handles forward with inputs of different resolutions
    if config.MODEL.TWO_STREAM or config.MODEL.TWO_TOKEN:
        student = utils.MultiCropWrapper(student, MultiDINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ))
        teacher = utils.MultiCropWrapper(
            teacher,
            MultiDINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )
    else:
        student = utils.MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ), vary_fr=config.DATA.RAND_FR)
        teacher = utils.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
            vary_fr=config.DATA.RAND_FR
        )

    if config.MODEL.TWO_STREAM:
        motion_student = utils.MultiCropWrapper(
            motion_student,
            DINOHead(motion_embed_dim, args.out_dim, args.use_bn_in_head),
        )
        motion_teacher = utils.MultiCropWrapper(
            motion_teacher,
            DINOHead(motion_embed_dim, args.out_dim, args.use_bn_in_head),
        )
        # move networks to gpu
        motion_student, motion_teacher = motion_student.cuda(), motion_teacher.cuda()

    # kinetics400 pretrained weights
    pretrained_weights = torch.load('checkpoints/kinetics400_vitb_ssl.pth')
    student.load_state_dict(pretrained_weights)
    teacher.load_state_dict(pretrained_weights)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=False)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)
    msg = teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    print(f"initialized teacher with student msg: {msg}")
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    if config.MODEL.TWO_STREAM:
        if utils.has_batchnorms(motion_student):
            motion_student = nn.SyncBatchNorm.convert_sync_batchnorm(motion_student)
            motion_teacher = nn.SyncBatchNorm.convert_sync_batchnorm(motion_teacher)
            motion_teacher = nn.parallel.DistributedDataParallel(motion_teacher, device_ids=[args.gpu])
            motion_teacher_without_ddp = motion_teacher.module
        else:
            # teacher_without_ddp and teacher are the same thing
            motion_teacher_without_ddp = motion_teacher

        motion_student = nn.parallel.DistributedDataParallel(motion_student, device_ids=[args.gpu])
        motion_teacher_without_ddp.load_state_dict(motion_student.module.state_dict())
        for p in motion_teacher.parameters():
            p.requires_grad = False
        print(f"Motion Student and Teacher are built: they are both 2D ViT networks.")

    else:
        motion_teacher_without_ddp = None

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        global_crops=2,
        two_token=config.MODEL.TWO_TOKEN
    ).cuda()

    if config.MODEL.TWO_STREAM:
        dino_flow_loss = DINOLoss(args.out_dim, 2, args.warmup_teacher_temp,
                                  args.teacher_temp, args.warmup_teacher_temp_epochs, args.epochs).cuda()
        dino_cross_loss = DINOLoss(args.out_dim, args.local_crops_number + 2, args.warmup_teacher_temp,
                                   args.teacher_temp, args.warmup_teacher_temp_epochs, args.epochs).cuda()
    else:
        dino_flow_loss = None
        dino_cross_loss = None

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if config.MODEL.TWO_STREAM:
        motion_params_groups = utils.get_params_groups(motion_student)
        params_groups[0]['params'] += motion_params_groups[0]['params']
        params_groups[1]['params'] += motion_params_groups[1]['params']

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args, cfg=config,
                                      motion_loss=dino_flow_loss, cross_loss=dino_cross_loss,
                                      motion_student=motion_student, motion_teacher=motion_teacher,
                                      motion_teacher_without_ddp=motion_teacher_without_ddp, rand_conv=rand_conv)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'motion_student': motion_student.state_dict() if motion_student is not None else 0,
            'motion_teacher': motion_teacher.state_dict() if motion_teacher is not None else 0,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}  # **{f'val_{k}': v for k, v in val_stats.items()},
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, cfg=None, motion_teacher=None, motion_student=None,
                    motion_loss=None, cross_loss=None, motion_teacher_without_ddp=None, rand_conv=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _, _, meta) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        if cfg.MODEL.TWO_STREAM:
            if cfg.DATA.NO_FLOW_AUG:
                # meta['flow'] = [x.cuda(non_blocking=True) for x in meta['flow']]
                idx = np.random.choice(range(cfg.DATA.NUM_FRAMES), 2, replace=False)
                flow_images = [meta['flow'][x].cuda(non_blocking=True) for x in idx]
        elif cfg.MODEL.TWO_TOKEN:
            pass

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):

            if cfg.MODEL.TWO_STREAM:
                student_output_rgb, student_output_flow = student(images)
                teacher_output_rgb, _ = teacher(images[:2])  # only the 2 global views pass through the teacher
                teacher_flow = motion_teacher(flow_images[:2])
                student_flow = motion_student(flow_images)

                loss = dino_loss(student_output_rgb, teacher_output_rgb, epoch) + \
                       motion_loss(student_flow, teacher_flow, epoch) + \
                       cross_loss(student_output_flow, teacher_flow, epoch)
            elif cfg.MODEL.TWO_TOKEN:
                student_output = student(images[2:])  # 2 spatially local and 2 temporally global local views
                teacher_output = teacher(images[:2])  # only 2 global views through the teacher
                loss = dino_loss(student_output, teacher_output, epoch)
            else:
                student_output = student(images)
                if rand_conv is not None:
                    teacher_output = teacher([images[0], rand_conv(images[1])])
                else:
                    teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if cfg.MODEL.TWO_STREAM:
                for param_q, param_k in zip(motion_student.module.parameters(),
                                            motion_teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_knn(train_loader, test_loader, model, train_dataset, test_dataset, opt):
    # model.eval()  # teacher model already on eval
    print("Extracting features for train set...")
    train_features = extract_features(model, train_loader)
    print("Extracting features for val set...")
    test_features = extract_features(model, test_loader)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s for s in train_dataset._labels]).long()
    test_labels = torch.tensor([s for s in test_dataset._labels]).long()

    if utils.get_rank() == 0:
        train_features = train_features.cuda()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()

    print("Features are ready!\nStart the k-NN classification.")
    top1, top5 = knn_classifier(train_features, train_labels,
                                test_features, test_labels, opt.nb_knn, opt.temperature)
    return {"knn_top1": top1, "knn_top5": top5}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, two_token=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.two_token = two_token
        if self.two_token:
            self.n_crops = 4
            self.global_crops = 2
            self.register_buffer("center", torch.zeros(2, out_dim))
        else:
            self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0
        if self.two_token:
            student_out = [x / self.student_temp for x in student_output]
            student_out = [x.chunk(self.n_crops) for x in student_out]

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = [F.softmax((x - self.center[idx]) / temp, dim=-1) for idx, x in enumerate(teacher_output)]
            teacher_out = [x.detach().chunk(self.global_crops) for x in teacher_out]

            for iv in range(len(student_out[0])):
                if iv < 2:
                    q = teacher_out[0][0]
                    v = student_out[0][iv]
                    loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                else:
                    q = teacher_out[1][1]
                    v = student_out[1][iv]
                    loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        else:
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.n_crops)

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(self.global_crops)

            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        # we skip cases where student and teacher operate on the same view
                        continue
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        if isinstance(teacher_output, (tuple, list)):
            batch_center = [torch.sum(x, dim=0, keepdim=True) for x in teacher_output]
            dist.all_reduce(batch_center[0])
            dist.all_reduce(batch_center[1])
            batch_center = [x / (len(teacher_output[0]) * dist.get_world_size()) for x in batch_center]
            self.center[0, :] = self.center[0, :] * self.center_momentum + batch_center[0] * (1 - self.center_momentum)
            self.center[1, :] = self.center[1, :] * self.center_momentum + batch_center[1] * (1 - self.center_momentum)
        else:
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

            # ema update
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SVT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_svt(args)
