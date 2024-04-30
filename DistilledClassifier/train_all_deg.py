import argparse
import collections
import data_loader.data_loaders as module_data
import model.metric as module_metric
from parse_config import ConfigParser
from utils.util import set_seeds, set_seeds_prev

def main(config):
    logger = config.get_logger('train')
    logger.info(config)

    # setup data_loader instances
    degs_all = config['data_loader']['args']['deg_types']
    train_loaders_all = []
    val_loaders_all = []
    prev_deg = config.config['data_loader']['args']['deg_type']
    for deg in degs_all:
        config.config['data_loader']['args']['deg_type'] = deg
        data_args = config['data_loader']['args']
        train_loaders_all.append(config.init_obj('data_loader', module_data))
        val_loaders_all.append(getattr(module_data, config['data_loader']['type'])(
                        data_args['data_dir'] if 'data_val_dir' not in data_args else data_args['data_dir'],
                        batch_size=data_args['batch_size'],
                        shuffle=False,
                        validation_split=0.0,
                        num_workers=data_args['num_workers'],
                        train=False,
                        deg_type = data_args['deg_type'],
                        ))  
    config.config['data_loader']['args']['deg_type'] = prev_deg
    Trainer = config.get_class('trainer', init = False)
    trainer = Trainer(config=config, 
                      train_loaders_all=train_loaders_all,
                      val_loaders_all=val_loaders_all)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Degraded Image Classification - KD')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--dt', '--deg_type'], type=str, target='data_loader;args;deg_type'),
        CustomArgs(['--rs', '--random_seed'], type=int, target='random_seed')
    ]
    config = ConfigParser.from_args(args, options)
    
    # fix random seeds for reproducibility
    if 'random_seed' in config:
        set_seeds(config['random_seed'])
    else:
        # Provides backward compability for previous experiments
        set_seeds_prev()
    main(config)
