
import torch.nn as nn

def get_loss(config):
    loss_name = config['loss_name']
    
    if loss_name == 'cross_entropy_loss':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f'invalid loss name {loss_name}')
