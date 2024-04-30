import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import model.loss as module_loss
import torch.nn as nn
import torch.nn.functional as F
import copy
from logger import setup_logging

class SLDA_Trainer(BaseTrainer):
    """
    Trainer class for training our proposed method FusionDistill based on distillation and fusion, 
    i.e., Step-4 of our proposed method. 
    """
    def __init__(self, config, train_loaders_all, val_loaders_all=None, 
                 len_epoch=None):
        super().__init__(config, train_loaders_all[0], val_loaders_all[0], len_epoch)
        self.degs_all = config['data_loader']['args']['deg_types']
        self.teachers, self.student = self._build_model(config)
        self.criterion = self._load_loss(config)
        self.optimizer = self._load_optimizer(self.student, config)
        self.lr_scheduler = self._load_scheduler(self.optimizer, config)
        self.config = config
        self.loss_names = {type: loss for losses in config['loss'] for type, loss in losses.items()}
        self.log_step = int(np.sqrt(train_loaders_all[0].batch_size))
        train_misc_metrics = ['loss', 'sup_loss', 'inh_loss', 'lr']
        valid_misc_metrics = ['loss', 'sup_loss', 'inh_loss']
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
        if config['teacher_model']['type'] == 'MM_Classifier' and config['student_model']['type'] == 'MM_Classifier':
            from mmcls.models import build_classifier
            teacher = build_classifier(config['teacher_model']['args'])
            student = build_classifier(config['student_model']['args'])
            # Manual fix for loading mmcls methods
            # Since it interferes with the logger, so reloading the logger 
            # after loading the model
            setup_logging(config.log_dir)
        else:
            raise NotImplementedError('Teacher and Student models should be of type MM_Classifier')

        self.logger.info('Teacher Network: {} \n Student Network: {}'.format(teacher, student))

        model_paths = [config['teacher_model']['pretrained_path_' + deg_type] for deg_type in self.degs_all]
        checkpoints = [torch.load(path) for path in model_paths]
        teachers = [copy.deepcopy(teacher) for _ in range(len(checkpoints))]
        # Feezing parameters of teachers
        for i, model in enumerate(teachers):
            model.load_state_dict(checkpoints[i]['state_dict']) 
            for param in model.parameters():
                param.requires_grad = False
            model = model.to(self.device)

        student = student.to(self.device)
        if 'pretrained_path' in config['student_model']:
            checkpoint = torch.load(config['student_model']['pretrained_path'])
            student.load_state_dict(checkpoint['state_dict'])
        
        return teachers, student
    
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
        return lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for teacher in self.teachers:
            teacher.eval()    
        self.student.train()
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
            
            batch_size = int(images_all.size(0)/len(self.teachers))
            if self.loss_names['inheritance_loss'] == 'COS':
                dum = torch.ones(batch_size,)
                dum = dum.to(self.device)
            
            self.optimizer.zero_grad()
            t_int_outs = []
            for i, teacher in enumerate(self.teachers):
                outs = teacher.extract_feat(images_all[batch_size*i:batch_size*(i+1)], stage='neck')
                t_int_outs.append(outs[-1])
            
            # This two statements are equivalent to model(inputs, gt_label = labels)
            s_int_out = self.student.extract_feat(images_all, stage='neck')
            s_out = self.student.head.forward_train(s_int_out, gt_label = labels_all)
            s_int_out = s_int_out[-1]
            
            sup_loss = s_out['loss'] * self.config['loss_weights'][0]
            inh_loss = 0
            for i, t_out in enumerate(t_int_outs):
                inh_loss += self.criterion['inheritance_loss'](s_int_out[batch_size*i:batch_size*(i+1)], t_out, dum) * self.config['loss_weights'][1]
            loss = sup_loss + inh_loss
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('inh_loss', inh_loss.item())
            self.train_metrics.update('sup_loss', sup_loss.item())
            self.train_metrics.update('accuracy@1', s_out['accuracy']['top-1'].item())
            self.train_metrics.update('accuracy@5', s_out['accuracy']['top-5'].item())
            self.train_metrics.update('lr', self.lr_scheduler.get_last_lr()[0])
            
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} {} Loss: {:.6f} Sup Loss: {:.6f} Inh Loss: {:.6f}'.format(
                        epoch, self._progress(batch_idx), loss.item(), 
                        sup_loss.item(), inh_loss.item()))
            
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
        for teacher in self.teachers:
            teacher.eval()    
        self.student.eval()
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
                
                batch_size = int(images_all.size(0)/len(self.teachers))
                if self.loss_names['inheritance_loss'] == 'COS':
                    dum = torch.ones(batch_size,)
                    dum = dum.to(self.device)
                
                t_int_outs = []
                for i, teacher in enumerate(self.teachers):
                    outs = teacher.extract_feat(images_all[batch_size*i:batch_size*(i+1)], stage='neck')
                    t_int_outs.append(outs[-1])
                
                # This two statements are equivalent to model(inputs, gt_label = labels)
                s_int_out = self.student.extract_feat(images_all, stage='neck')
                s_out = self.student.head.forward_train(s_int_out, gt_label = labels_all)
                s_int_out = s_int_out[-1]
                
                sup_loss = s_out['loss'] * self.config['loss_weights'][0]
                inh_loss = 0
                for i, t_out in enumerate(t_int_outs):
                    inh_loss += self.criterion['inheritance_loss'](s_int_out[batch_size*i:batch_size*(i+1)], t_out, dum) * self.config['loss_weights'][1]

                loss = sup_loss + inh_loss

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('inh_loss', inh_loss.item())
                self.valid_metrics.update('sup_loss', sup_loss.item())
                self.valid_metrics.update('accuracy@1', s_out['accuracy']['top-1'].item())
                self.valid_metrics.update('accuracy@5', s_out['accuracy']['top-5'].item())
                
        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_name = type(self.student).__name__
        state = {
            'model': model_name,
            'epoch': epoch,
            'state_dict': self.student.state_dict(),
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
        if checkpoint['config']['student_model'] != self.config['student_model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.student.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
