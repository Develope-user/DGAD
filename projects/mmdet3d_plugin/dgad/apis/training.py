import os.path as osp

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmcv.utils import build_from_cfg
from mmdet.core import EvalHook
from mmdet.utils import get_root_logger

from projects.mmdet3d_plugin.core.evaluation import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets.builder import (
    build_dataloader,
    custom_build_dataset,
)


def custom_train_detector(model, dataset, cfg, distributed=False,
                          validate=False, timestamp=None, meta=None):
    """Train DGAD using the temporal scene-aware data loader."""
    logger = get_root_logger(cfg.log_level)
    datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            item,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
            pin_memory=cfg.data.get('pin_memory', False),
            persistent_workers=cfg.data.get('persistent_workers', False),
        )
        for item in datasets
    ]

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False),
        )
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )
    runner.timestamp = timestamp

    fp16_cfg = cfg.get('fp16')
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config'),
    )
    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    if validate:
        val_cfg = cfg.data.val.copy()
        samples_per_gpu = val_cfg.pop('samples_per_gpu', 1)
        val_dataset = custom_build_dataset(val_cfg, dict(test_mode=True))
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
            pin_memory=cfg.data.get('pin_memory', False),
            persistent_workers=cfg.data.get('persistent_workers', False),
        )
        eval_cfg = cfg.get('evaluation', {}).copy()
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join(cfg.work_dir, 'validation')
        hook_type = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(hook_type(val_loader, **eval_cfg))

    for hook_cfg in cfg.get('custom_hooks', []):
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        runner.register_hook(build_from_cfg(hook_cfg, HOOKS), priority=priority)

    if cfg.get('resume_from'):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from'):
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
