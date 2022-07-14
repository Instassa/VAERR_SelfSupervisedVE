import torch.optim as optim
import math



def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CosineScheduler:
    def __init__(self, optimizer, init_lr: float, epochs: int):
        self.init_lr = init_lr
        self.epochs = epochs
        self.optimizer = optimizer

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict) -> None:
        self.__dict__.update(state_dict)

    def step(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.init_lr*reduction_ratio)


def get_optimizer(optimizer: str, params, lr: float, weight_decay: float):

  
    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(params,lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9 )

    return optimizer