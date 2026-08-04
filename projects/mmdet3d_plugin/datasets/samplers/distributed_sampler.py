import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 scene_aware=False):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.seed = seed if seed is not None else 0
        self.scene_aware = scene_aware
        self.scene_indices = self._build_scene_indices() if scene_aware else None
        if self.scene_indices is not None:
            self.num_samples = len(self.scene_indices)
            self.total_size = sum(len(scene) for scene in self._split_scenes())

    def __iter__(self):
        if self.shuffle:
            assert False
        elif self.scene_indices is not None:
            indices = self.scene_indices
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        if self.scene_indices is not None:
            return iter(indices)
        indices = (indices *
                   math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size

        per_replicas = self.total_size//self.num_replicas
        indices = indices[self.rank*per_replicas:(self.rank+1)*per_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def _split_scenes(self):
        if not hasattr(self.dataset, 'data_infos'):
            return None

        scenes = []
        current_scene = []
        current_token = None
        for index, info in enumerate(self.dataset.data_infos):
            scene_token = info.get('scene_token', None)
            if current_scene and scene_token != current_token:
                scenes.append(current_scene)
                current_scene = []
            current_scene.append(index)
            current_token = scene_token
        if current_scene:
            scenes.append(current_scene)
        return scenes

    def _build_scene_indices(self):
        scenes = self._split_scenes()
        if not scenes:
            return None

        rank_scenes = scenes[self.rank::self.num_replicas]
        return [index for scene in rank_scenes for index in scene]
