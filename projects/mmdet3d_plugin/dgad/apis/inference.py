import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def custom_multi_gpu_test(model, data_loader, tmpdir=None,
                          gpu_collect=False, show_progress=True):
    """Run inference while preserving temporal sample order."""
    del gpu_collect
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    progress = mmcv.ProgressBar(len(dataset)) if rank == 0 and show_progress else None

    for data in data_loader:
        with torch.no_grad():
            batch_results = model(return_loss=False, rescale=True, **data)
        if isinstance(batch_results, dict):
            batch_results = batch_results.get('bbox_results', [])
        results.extend(batch_results)
        if progress is not None:
            for _ in range(len(batch_results) * world_size):
                progress.update()

    results = _collect_results(results, len(dataset), tmpdir)
    if rank == 0:
        return _sort_by_sample_token(results, dataset)
    return None


def _sort_by_sample_token(results, dataset):
    if not hasattr(dataset, 'data_infos'):
        return results

    token_to_position = {
        info.get('token'): index
        for index, info in enumerate(dataset.data_infos)
        if info.get('token') is not None
    }
    ordered = [None] * len(dataset)
    unmatched = []
    for result in results:
        sample_token = None
        if isinstance(result, dict):
            prediction = result.get('pts_bbox', {})
            sample_token = prediction.get('sample_idx') if isinstance(prediction, dict) else None
            sample_token = result.get('sample_idx', sample_token)
        position = token_to_position.get(sample_token)
        if position is None or ordered[position] is not None:
            unmatched.append(result)
        else:
            ordered[position] = result

    ordered = [result for result in ordered if result is not None]
    return (ordered + unmatched)[:len(dataset)]


def _collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    if tmpdir is None:
        max_length = 512
        directory = torch.full(
            (max_length,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            encoded = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            directory[:len(encoded)] = encoded
        dist.broadcast(directory, 0)
        tmpdir = directory.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    if rank != 0:
        return None

    results = []
    for worker in range(world_size):
        results.extend(mmcv.load(osp.join(tmpdir, f'part_{worker}.pkl')))
    shutil.rmtree(tmpdir)
    return results[:size]
