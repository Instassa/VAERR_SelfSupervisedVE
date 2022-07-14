from logging import logProcesses
import numpy as np
import torch
import torch.nn as nn


def compute_loss_cat(pred, targets, cat_targets, lossType: int, startpts_ar, startpts_val, params, bins, batch_idx, interval, print_loss, every_frame):

    if lossType >= 15 and lossType <= 19: # Classic Composite Loss as CCC + CE + MSE
            CE = nn.CrossEntropyLoss()
            if every_frame == False:
                # print(np.shape(pred))
                lossCat = CE(pred, torch.argmax(cat_targets[:,:,0].long(), axis=1))
            else:
                lossCat = 0;#Variable(torch.zeros(1), requires_grad=True)
                for batch in range(np.shape(pred)[0]):
                    lossCat += CE(pred[batch,:], torch.argmax(cat_targets[batch,0,:].long(), axis=1))

            loss = lossCat/pred.size()[0]
            if batch_idx % interval == 0 and print_loss==True:
                print('LOSS: ', loss.item(), 'CATEGORIC: ', lossCat.item())
    else:
        raise Exception('LossType unknown')

    return loss, lossCat/pred.size()[0], startpts_ar, startpts_val