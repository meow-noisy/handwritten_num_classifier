
import sys
from pathlib import Path

import torch

this_file_dir = Path(__file__).parent.resolve()

sys.path += [
    str(this_file_dir),
]

def get_model(config):     
    model_name = config['model_name']
    if model_name == 'alexnet':
        from networks.alexnet import load_alexnet, AlexNet
        net = load_alexnet(config)
    else:
        raise ValueError(f'invalid model name {model_name}')
    return net

def load_weight(net, weight_file_path):
    model_state_dict = torch.load(str(weight_file_path))
    net.load_state_dict(model_state_dict)
