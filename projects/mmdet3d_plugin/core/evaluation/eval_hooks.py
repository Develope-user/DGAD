import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class CustomDistEvalHook(DistEvalHook):
    """Distributed evaluation hook that preserves scene ordering."""

    def _do_evaluate(self, runner):
        if self.broadcast_bn_buffer:
            for module in runner.model.modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        from projects.mmdet3d_plugin.dgad.apis.inference import (
            custom_multi_gpu_test,
        )

        tmpdir = self.tmpdir or osp.join(runner.work_dir, '.eval_hook')
        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
        )
        if runner.rank == 0:
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, key_score)
