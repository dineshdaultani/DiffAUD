import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, deg_flag = 'clean'):
        """
        Constructor
        Args:
            num_of_features (int) : The number of features extracted by the feature extractor.
            init_weights (bool) : True if you initialize weights.
            deg_flag (int) : clean when training with clean images
                            deg when training with clean images
        """
        super(BaseModel, self).__init__()
        self.deg_flag = deg_flag
        print('Creating network object with deg_flag: ', self.deg_flag)

    def define_input(self, *inputs):
        """
        Forward pass logic
        if deg_flag is True, run forward pass on degraded images
        else run forward pass on clean images 
        :return: Model output
        """
        clean_img, deg_img = inputs
        if self.deg_flag == 'clean':
            return clean_img
        elif self.deg_flag == 'deg':
            return deg_img
        else:
            raise NotImplementedError


    def _init_weight_layers(self, layers):
        """
        Initialize each layer depends on the layer type
        """
        for layer in layers.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    @abstractmethod
    def forward():
        """
        forward function for the model
        """
        msg = "forward functionhas not been implemeted."
        raise NotImplementedError(msg)

    