import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.metric as module_metric
from parse_config import ConfigParser
from utils.data import degradedimagedata as deg_data
from logger import TensorboardWriter
from utils.util import set_seeds
from utils import prepare_device
from logger import setup_logging
import csv, os

# fix random seeds for reproducibility
set_seeds()

def get_corruptions(config):
    corruption_type = config['data_loader']['corruption_type']
    severities = None
    if 'severity' in config['data_loader'] and config['data_loader']['severity'] is not None:
        severities = config['data_loader']['severity'] 
        severities = severities.split(',')
        severities = list(map(int, severities))
        
    if corruption_type == 'single' or corruption_type is None:
        corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]
        if severities is None:
            severities = range(1, 6)
    elif corruption_type == 'sequence':
        corruptions = [
            'weak_seq', 'medium_seq', 'strong_seq',
        ]
        if severities is None:
            severities = range(1) # To run the severity loop only once
    else:
        raise NotImplementedError(f'corruption_type: {corruption_type} is not defined.')
    return corruptions, severities

def main(config):
    logger = config.get_logger('test')
    logger.info(config)
    device, device_ids = prepare_device(config['n_gpu'])
    
    writer = TensorboardWriter(config.log_dir, logger, 
                               config['trainer']['args']['tensorboard'])
    # deg_range = deg_data.get_type_range(config['data_loader']['args']['deg_type'])

    # build model architecture
    if 'model' in config and config['model']['type'] == 'MM_Classifier':
        from mmcls.models import build_classifier
        model = build_classifier(config['model']['args'])
        # Manual fix for loading mmcls methods
        # Since it interferes with the logger, so reloading the logger 
        # after loading the model
        setup_logging(config.log_dir)
    elif 'student_model' in config and config['student_model']['type'] == 'MM_Classifier':
        from mmcls.models import build_classifier
        model = build_classifier(config['student_model']['args'])
        # Manual fix for loading mmcls methods
        # Since it interferes with the logger, so reloading the logger 
        # after loading the model
        setup_logging(config.log_dir)
    else:
        model = config.get_class('model')

    logger.info(model)

    # Define the CSV file path and write the header
    csv_file_path = os.path.join(config.log_dir, 'corruption_log.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['corruption', 'severity', 'accuracy@1', 'accuracy@5'])  # Write the header row

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    corruption_type = config['data_loader']['corruption_type']
    corruptions, severities = get_corruptions(config)
    for corruption in corruptions:
        for severity in severities:
        # setup data_loader instances
            data_loader = getattr(module_data, config['data_loader']['type'])(
                config['data_loader']['args']['data_dir'],
                batch_size=128,
                shuffle=False,
                validation_split=0.0,
                num_workers=2,
                train=False,
                deg_type=corruption, 
                severity=str(severity) if corruption_type == 'single' else None,
            )
            
            total_loss = 0.0
            total_metrics = torch.zeros(len(config['metrics']))

            with torch.no_grad():
                for batch_idx, (images, targets, paths) in enumerate(data_loader):
                    labels = targets['y']
                
                    images = images.to(device)
                    target = labels.to(device)
                    
                    mm_outputs = model(images, gt_label=target)
                    
                    batch_size = images.shape[0]
                    total_metrics[0] += mm_outputs['accuracy']['top-1'].item() * batch_size
                    total_metrics[1] += mm_outputs['accuracy']['top-5'].item() * batch_size

            n_samples = len(data_loader.sampler)
            log = {'corruption': corruption, 'severity': severity}
            
            log.update({
                met: total_metrics[i].item() / n_samples for i, met in enumerate(config['metrics'])
            })
            
            logger.info(log)
            
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(log.values()))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Degraded Image Classification - KD')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-m', '--mode', default='eval', type=str,
                      help='Activate eval mode for config')
    # custom cli options to modify configuration from default values given in yaml file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--dt', '--deg_type'], type=str, target='data_loader;args;deg_type'),
        CustomArgs(['--dd', '--data_dir'], type=str, target='data_loader;args;data_dir'),
        # CustomArgs(['--cn', '--config_name'], type=str, target='name', default='C'),
        CustomArgs(['--dl', '--data_loader'], type=str, target='data_loader;type'),
        CustomArgs(['--ct', '--corruption_type'], type=str, target='data_loader;corruption_type'),
        CustomArgs(['--sev', '--severity'], type=str, target='data_loader;severity'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
