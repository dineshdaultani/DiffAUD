import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import model.loss as module_loss
import copy
from logger import setup_logging
import warmup_scheduler

class IndDATrainer(BaseTrainer):
    """
    Trainer class for training Individual model for combination of several degradation.
    This trainer is used for training methods such as: Scratch, Vanilla, and Fused.
    """
    def __init__(self, config, train_loaders_all, val_loaders_all=None, 
                 len_epoch=None):
        super().__init__(config, train_loaders_all[0], val_loaders_all[0], len_epoch)
        self.model = self._build_model(config)
        self.criterion = self._load_loss(config)
        self.optimizer = self._load_optimizer(self.model, config)
        self.lr_scheduler = self._load_scheduler(self.optimizer, config)
        self.config = config

        self.log_step = int(np.sqrt(train_loaders_all[0].batch_size))
        train_misc_metrics = ['loss', 'lr']
        valid_misc_metrics = ['loss']
        self.train_metrics = MetricTracker(*train_misc_metrics, 
                                            *[m for m in self.config['metrics']], 
                                            writer=self.writer)
        self.valid_metrics = MetricTracker(*valid_misc_metrics, 
                                            *[m for m in self.config['metrics']], 
                                            writer=self.writer)
        self.train_loaders_all = train_loaders_all
        self.val_loaders_all = val_loaders_all

    def _build_model(self, config):
        """
        Building model from the configuration file

        :param config: config file
        :return: model with loaded state dict
        """
        # build model architecture, then print to console
        if config['model']['type'] == 'MM_Classifier':
            from mmcls.models import build_classifier
            model = build_classifier(config['model']['args'])
            # Manual fix for loading mmcls methods
            # Since it interferes with the logger, so reloading the logger 
            # after loading the model
            setup_logging(config.log_dir)
        else:
            model = config.get_class('model')
        
        self.logger.info(model)
        
        if 'pretrained_path' in config['model']:
            self.logger.info('Loading pretrained weights for the model from path: {}'.format(config['model']['pretrained_path']))
            checkpoint = torch.load(config['model']['pretrained_path'])
            model.load_state_dict(checkpoint['state_dict'])
        elif 'init_cfg' not in config['model']['args']['backbone']:
            # Initialize weights if no pretrained weights are given.
            self.logger.info('Initializing random weights for the model')
            model.init_weights()
        else:
            self.logger.info('Weights are not initialized!')
            
            
        model = model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model
    
    def _load_loss(self, config):
        """
        Build model from the configuration file

        :param config: config file
        :return: criterion dictionary in the format: {loss_type: loss}
        """
        criterion = {type: getattr(module_loss, type)(loss) for losses in config['loss'] \
                        for type, loss in losses.items()}
        return criterion

    def _load_optimizer(self, model, config):
        """
        Load optimizer from the configuration file

        :param model: model for which optimizer is to be initialized
        :param config: config file
        :return: initialized optimizer
        """
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        return optimizer

    def _load_scheduler(self, optimizer, config):
        """
        Load scheduler from the configuration file

        :param optimizer: optimizer for which scheduler is to be initialized
        :param config: config file
        :return: initialized scheduler
        """
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        if 'lr_warmup' in config and config['lr_warmup'] is not None:
            lr_scheduler = config.init_obj('lr_warmup', warmup_scheduler, optimizer, after_scheduler = lr_scheduler)
        return lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, loaders_all in enumerate(zip(*self.train_loaders_all)):
            images_all, labels_all = None, None
            for loader in loaders_all:
                images, targets, paths = loader
                labels = targets['y']
                if images_all is None:
                    images_all = copy.deepcopy(images)
                    labels_all = copy.deepcopy(labels)
                else:
                    images_all = torch.cat((images_all, images))
                    labels_all = torch.cat((labels_all, labels))
                
            images_all = images_all.to(self.device)
            labels_all = labels_all.to(self.device)

            self.optimizer.zero_grad()
            
            mm_outputs = self.model(images_all, gt_label=labels_all)
            mm_outputs['loss'].backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', mm_outputs['loss'].item())
            self.train_metrics.update('accuracy@1', mm_outputs['accuracy']['top-1'].item())
            self.train_metrics.update('accuracy@5', mm_outputs['accuracy']['top-5'].item())
            self.train_metrics.update('lr', self.lr_scheduler.get_last_lr()[0])
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    mm_outputs['loss'].item()))
            
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            self.logger.info('Testing on validation data')
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, loaders_all in enumerate(zip(*self.val_loaders_all)):
                images_all, labels_all = None, None
                for loader in loaders_all:
                    images, targets, paths = loader
                    labels = targets['y']
                    if images_all is None:
                        images_all = copy.deepcopy(images)
                        labels_all = copy.deepcopy(labels)
                    else:
                        images_all = torch.cat((images_all, images))
                        labels_all = torch.cat((labels_all, labels))
                    
                images_all = images_all.to(self.device)
                labels_all = labels_all.to(self.device)

                mm_outputs = self.model(images_all, gt_label=labels_all)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', mm_outputs['loss'].item())
                self.valid_metrics.update('accuracy@1', mm_outputs['accuracy']['top-1'].item())
                self.valid_metrics.update('accuracy@5', mm_outputs['accuracy']['top-5'].item())

        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_name = type(self.model).__name__
        state = {
            'model': model_name,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        # torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, config):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(config.resume)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
