import torch.optim as optim

def get_optimizer(net, config):
    
    optimizer_name = config['optimizer_name']
    
    if optimizer_name == 'sgd':
        return optim.SGD(
            net.parameters(), 
            lr=config['lr'], 
            momentum=config['momentum'], 
            weight_decay=config['weight_decay'])
    else:
        raise ValueError(f'invalid optimizer name {optimizer_name}')