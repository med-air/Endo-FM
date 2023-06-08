# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from ..utils.comm import is_main_process
from tqdm import tqdm

from stft_core.data import make_data_loader
from stft_core.utils.comm import get_world_size, synchronize
from stft_core.utils.metric_logger import MetricLogger
from stft_core.engine.inference import inference

# from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
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
    tb_writer=None,
):
    logger = logging.getLogger("stft_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST
    logger.info("iou_types:{}\n".format(iou_types))

    rank_metric = {}
    rank_metric['map'] = 0 #for VID

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            # logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        if not cfg.MODEL.VID.ENABLE:
            images = images.to(device)
            targets = [target.to(device) for target in targets]
        else:
            method = cfg.MODEL.VID.METHOD
            if method in ("base", "cvc_image"):
                images = images.to(device)
            elif method in ("rdn", "mega", "fgfa", "stft", "cvc_fgfa", "cvc_mega", "cvc_rdn", "cvc_stft"):
                images["cur"] = images["cur"].to(device)
                for key in ("ref", "ref_l", "ref_m", "ref_g"):
                    if key in images.keys():
                        images[key] = [img.to(device) for img in images[key]]
            else:
                raise ValueError("method {} not supported yet.".format(method))
            targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # print(iteration, losses_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with amp.scale_loss(losses, optimizer) as scaled_losses:
        #     scaled_losses.backward()
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            if tb_writer is not None:
                tb_writer.add_scalar('Training/LR', optimizer.param_groups[0]["lr"], iteration)
                for each_name, each_meter in meters.meters.items():
                    tb_writer.add_scalar('Training/{}'.format(each_name), each_meter.global_avg, iteration)
            if is_main_process():
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        inference_flag = False
        if test_period > 0 and iteration % test_period == 0:
            inference_flag = True

        if inference_flag and iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if inference_flag:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            eval_metric_list = inference(cfg,  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()

            if tb_writer is not None:
                if 'cvc_' in cfg.MODEL.VID.METHOD:
                    for eval_idx, eval_metric in eval_metric_list.items():
                        for each_name, each_meter in eval_metric.items():
                            tb_writer.add_scalar('{}_{}'.format(eval_idx, each_name), each_meter, iteration)
                else:
                    for eval_idx, eval_metric in eval_metric_list.items():
                        tb_writer.add_scalar('Validation/{}_{}'.format(str(eval_idx),'map'), eval_metric['map'], iteration)
                        if eval_metric['map'] > rank_metric['map']:
                            rank_metric['map'] = eval_metric['map']
                            checkpointer.save("model_best_map_{}".format(str(eval_idx)), **arguments)
                        tb_writer.add_scalar('Validation/{}_best_{}'.format(str(eval_idx),'map'), rank_metric['map'], iteration)

            model.train()

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
