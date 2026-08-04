#!/usr/bin/env python
import argparse
import importlib
import json
import os
import os.path as osp
import sys

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
importlib.import_module('projects.mmdet3d_plugin')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.dgad.apis import custom_multi_gpu_test

DEFAULT_CONFIG = 'projects/configs/dgad.py'
DEFAULT_CHECKPOINT = 'checkpoint.pth'


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DGAD')
    parser.add_argument('config', nargs='?', default=DEFAULT_CONFIG)
    parser.add_argument('checkpoint', nargs='?', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out', help='save predictions as a pickle file')
    parser.add_argument('--output-dir', default='work_dirs/evaluation')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--eval', nargs='+', default=['bbox'])
    parser.add_argument('--show-progress', action='store_true')
    parser.add_argument('--tmpdir')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--eval-options', nargs='+', action=DictAction)
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if args.out and not args.out.endswith(('.pkl', '.pickle')):
        parser.error('--out must use the .pkl or .pickle extension')
    return args


def prepare_test_config(cfg):
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    cfg.model.train_cfg = None
    test_cfg = cfg.data.test
    test_cfg.test_mode = True
    samples_per_gpu = test_cfg.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        test_cfg.pipeline = replace_ImageToTensor(test_cfg.pipeline)
    return samples_per_gpu


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    samples_per_gpu = prepare_test_config(cfg)

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        pin_memory=cfg.data.get('pin_memory', False),
        persistent_workers=cfg.data.get('persistent_workers', False),
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model, args.checkpoint, map_location='cpu', strict=True)
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Evaluation samples: {len(dataset):,}')

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
    else:
        model = MMDataParallel(model, device_ids=[0])

    outputs = custom_multi_gpu_test(
        model,
        data_loader,
        tmpdir=args.tmpdir,
        show_progress=args.show_progress,
    )
    rank, _ = get_dist_info()
    if rank != 0:
        return

    mmcv.mkdir_or_exist(args.output_dir)
    if args.out:
        mmcv.dump(outputs, args.out)

    eval_options = args.eval_options or {}
    eval_options['jsonfile_prefix'] = args.output_dir
    if args.format_only:
        dataset.format_results(outputs, **eval_options)
        return

    eval_cfg = cfg.get('evaluation', {}).copy()
    for key in ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule'):
        eval_cfg.pop(key, None)
    eval_cfg.update(metric=args.eval, **eval_options)
    metrics = dataset.evaluate(outputs, **eval_cfg)
    metrics_path = osp.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print(f'Metrics saved to {metrics_path}')


if __name__ == '__main__':
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
