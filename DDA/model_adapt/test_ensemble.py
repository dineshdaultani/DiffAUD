# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.engine import single_gpu_test, multi_gpu_test

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import get_root_logger, setup_multi_processes

from model_adapt.evaluation import (select_logits_to_idx, tackle_img_from_idx,
                            single_gpu_test_ensemble, multi_gpu_test_ensemble)
from parse_config import ConfigParser
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='DDA test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--device', default=None, help='device used for testing. (Deprecated)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--ensemble', 
        type=str, 
        default='sum',
        choices=['first', 'second', 'entropy', 'confidence', 'var', 
        'entropy_fuse', 'confidence_fuse', 'var_fuse',
        'sum', 'entropy_sum', 'confidence_sum'])
    # Added arguments
    parser.add_argument('--corruption', type=str, default='shot_noise')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--data_prefix1', type=str, default='dataset/cifar10c')
    parser.add_argument('--data_prefix2', type=str, default='dataset/generated/cifar10c')
    parser.add_argument('--second_model_prefix', type=str, default=None)
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Overriding the dataset corruption and severity if passes as arguments. 
    if 'corruption' in args:
        cfg.data.test['corruption'] = args.corruption
    if 'severity' in args:
        cfg.data.test['severity'] = args.severity
    if 'data_prefix1' in args:
        cfg.data.test['data_prefix'] = args.data_prefix1
    if 'data_prefix2' in args:
        cfg.data.test['data_prefix2'] = args.data_prefix2
    test(args, cfg, 'ensemble')


def test(args, cfg, test_fn: str):
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'test2', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    if dataset.ATTRIBUTE['dataset'] == 'CIFAR10C' or dataset.ATTRIBUTE['dataset'] == 'CIFAR10' \
                or dataset.ATTRIBUTE['dataset'] == 'IMAGENETC' or dataset.ATTRIBUTE['dataset'] == 'IMAGENET':
        if args.second_model_prefix:
            print('Loading second model from checkpoint:', args.second_model_prefix)
            model2 = copy.deepcopy(model)
            checkpoint2 = torch.load(args.second_model_prefix)
            model2.load_state_dict(checkpoint2['state_dict'])
        else:
            model2 = None
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise NotImplementedError('Dataset is not supported here!')
        
    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    elif dataset.ATTRIBUTE['dataset'] == 'IMAGENETC' or dataset.ATTRIBUTE['dataset'] == 'IMAGENET':
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES
    elif dataset.ATTRIBUTE['dataset'] == 'CIFAR10C' or dataset.ATTRIBUTE['dataset'] == 'CIFAR10':
        from mmcls.datasets import CIFAR10
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use cifar10 by default.')
        CLASSES = CIFAR10.CLASSES
    else:
        raise NotImplementedError('Class names are not saved in datasets!') 

    if 'ensemble' in test_fn:
        print(f"Ensemble: {args.ensemble}")
    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
            model2 = model2.cpu() if model2 else None
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            model2 = MMDataParallel(model2, device_ids=cfg.gpu_ids) if model2 else None
            if not model.device_ids:
                assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
                    'To test with CPU, please confirm your mmcv version ' \
                    'is not lower than v1.4.4'
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        if 'ensemble' in test_fn:
            outputs = single_gpu_test_ensemble(model, data_loader, args.ensemble, args.show, args.show_dir, 
                                  model2 = model2, **show_kwargs)
        else:
            outputs = single_gpu_test(model, data_loader)    
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        model2 = MMDistributedDataParallel(
            model2.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if 'ensemble' in test_fn:
            outputs = multi_gpu_test_ensemble(model, data_loader, args.ensemble, args.tmpdir,
                                 args.gpu_collect, model2 = model2, )
        else:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, )
        
            

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        logger = get_root_logger()
        if args.metrics:
            eval_results = dataset.evaluate(
                results=outputs,
                metric=args.metrics,
                metric_options=args.metric_options,
                logger=logger)
            results.update(eval_results)
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray):
                    v = [round(out, 2) for out in v.tolist()]
                elif isinstance(v, Number):
                    v = round(v, 2)
                else:
                    raise ValueError(f'Unsupport metric type: {type(v)}')
                print(f'\n{k} : {v}')
        if args.out:
            if 'none' not in args.out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    # Saving only none for scores to save space in dumped files. 
                    'class_scores': None, #scores,
                    'pred_score': None, #pred_score,
                    'pred_label': None, #pred_label,
                    'pred_class': None, #pred_class
                }
                if 'all' in args.out_items:
                    results.update(res_items) 
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
            fname = os.path.join(args.out, str(args.severity), dataset.corruption[0] + '.pkl')
            print(f'\ndumping results to {fname}')
            mmcv.dump(results, fname)
    return [v for v in eval_results.values()]


if __name__ == '__main__':
    main()
