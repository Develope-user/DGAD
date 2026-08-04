from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class CustomSetEpochInfoHook(Hook):
    """Expose the current epoch to schedule DGAD losses."""

    def before_train_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(runner.epoch)
