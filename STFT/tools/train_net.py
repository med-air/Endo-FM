# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
from stft_core.utils.env import setup_environment
from stft_core.utils.dist_env import init_dist
from stft_core.utils.distributed import ompi_rank, ompi_local_rank

import argparse
import time
import os
from tensorboardX import SummaryWriter

import torch
from stft_core.config import cfg
from stft_core.data import make_data_loader
from stft_core.solver import make_lr_scheduler
from stft_core.solver import make_optimizer
from stft_core.engine.inference import inference
from stft_core.engine.trainer import do_train
from stft_core.modeling.detector import build_detection_model
from stft_core.utils.checkpoint import DetectronCheckpointer
from stft_core.utils.collect_env import collect_env_info
from stft_core.utils.comm import synchronize, get_rank
from stft_core.utils.imports import import_file
from stft_core.utils.logger import setup_logger
from stft_core.utils.miscellaneous import mkdir, save_config


def train(cfg, local_rank, distributed, logger, tb_writer):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info("\nmodel:\n{}".format(model))

    if cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.SOLVER.BASE_LR)
    elif cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = make_optimizer(cfg, model)
    else:
        raise NotImplementedError("Unsupported optimizer type {}.".format(cfg.SOLVER.OPTIMIZER))

    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, ignore=cfg.MODEL.VID.IGNORE)
    if cfg.MODEL.VID.METHOD in ("fgfa", "dff", "cvc_fgfa"):
        checkpointer.load_flownet(cfg.MODEL.VID.FLOWNET_WEIGHT)

    if not cfg.MODEL.VID.IGNORE:
        arguments.update(extra_checkpoint_data)
    logger.info("\narguments:\n{}".format(arguments))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        tb_writer=tb_writer,
    )

    return model



def main():
    parser = argparse.ArgumentParser(description="PyTorch Video Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--master_port", "-mp", type=str, default='29999')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        init_dist(args.launcher, args=args)
        synchronize()

    BASE_CONFIG = "configs/BASE_RCNN_{}gpu.yaml".format(num_gpus)
    if os.path.exists(BASE_CONFIG):
        cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    cfg.freeze()

    logger = setup_logger("stft_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    if get_rank()>0:
        tb_writer = None
    else:
        tb_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'tb_train'))

    if args.launcher == "mpi":
        args.local_rank = ompi_local_rank()
    model = train(cfg, args.local_rank, args.distributed, logger, tb_writer)

    exit(0)



if __name__ == "__main__":
    main()
