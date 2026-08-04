#!/usr/bin/env python
import argparse
import importlib
import os
import os.path as osp
import sys
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmseg import __version__ as mmseg_version

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
importlib.import_module('projects.mmdet3d_plugin')
from projects.mmdet3d_plugin.dgad.apis import custom_train_detector

DEFAULT_CONFIG = 'projects/configs/dgad.py'


def parse_args():
    parser = argparse.ArgumentParser(description='Train DGAD')
    parser.add_argument('config', nargs='?', default=DEFAULT_CONFIG)
    parser.add_argument('--work-dir')
    parser.add_argument('--load-from')
    parser.add_argument('--resume-from')
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--gpu-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--autoscale-lr', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def configure_batch_scaling(cfg, logger):
    scale_cfg = cfg.get('auto_scale_lr', {})
    if not scale_cfg.get('enable', False):
        return

    base_batch_size = scale_cfg.get('base_batch_size', 8)
    actual_batch_size = cfg.data.samples_per_gpu * len(cfg.gpu_ids)
    cumulative_iters = 1
    if scale_cfg.get('auto_accumulate', False) and actual_batch_size < base_batch_size:
        cumulative_iters = max(1, round(base_batch_size / actual_batch_size))
        cfg.optimizer_config = cfg.optimizer_config.copy()
        cfg.optimizer_config['type'] = 'GradientCumulativeOptimizerHook'
        cfg.optimizer_config['cumulative_iters'] = cumulative_iters
        if cfg.lr_config.get('warmup_iters') is not None:
            cfg.lr_config['warmup_iters'] *= cumulative_iters

    effective_batch_size = actual_batch_size * cumulative_iters
    cfg.optimizer['lr'] *= effective_batch_size / base_batch_size
    logger.info(
        'Effective batch size: %d; learning rate: %.6g',
        effective_batch_size,
        cfg.optimizer['lr'],
    )


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir or cfg.get('work_dir') or 'work_dirs/dgad'
    cfg.load_from = args.load_from or cfg.get('load_from')
    cfg.resume_from = args.resume_from or cfg.get('resume_from')
    cfg.gpu_ids = args.gpu_ids or [0]

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = list(range(world_size))

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(
        log_file=osp.join(cfg.work_dir, f'{timestamp}.log'),
        log_level=cfg.log_level,
        name='mmdet',
    )
    logger.info('Distributed training: %s', distributed)
    logger.info('Environment:\n%s', '\n'.join(
        f'{key}: {value}' for key, value in collect_env().items()))

    if args.autoscale_lr:
        cfg.auto_scale_lr.enable = True
    configure_batch_scaling(cfg, logger)
    set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            CLASSES=datasets[0].CLASSES,
            PALETTE=getattr(datasets[0], 'PALETTE', None),
        )

    logger.info('Model parameters: %d', sum(
        parameter.numel() for parameter in model.parameters()))
    custom_train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not args.no_validate,
        timestamp=timestamp,
        meta={'seed': args.seed, 'exp_name': osp.basename(args.config)},
    )


if __name__ == '__main__':
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
