from logging import logProcesses
import numpy as np
import torch
import torch.nn as nn
from loss.Kmeans import KMeans
from loss.cccLoss import ConcordanceCorCoeffLoss

def compute_loss_CCE(pred, targets, lossType: int, startpts_ar, startpts_val, params, bins, batch_idx, interval, print_loss):

    if lossType == 1: # MSE
        criterion = criterion = nn.MSELoss()
        loss = criterion(pred, targets) 
        # print('loss: ', loss.item())
        
    elif lossType == 2: # CCC
        criterion = nn.ConcordanceCorCoeffLoss()
        if len(targets.size()) == 2:
            # print("T2", np.shape(pred), np.shape(targets))
            loss1 = criterion(pred, targets)
            loss = loss1
            print('loss1: ', loss1.item())
            
        elif len(targets.size()) == 3:
            loss1 = criterion(pred[:,:,0], targets[:,:,0])
            loss2 = criterion(pred[:,:,1], targets[:,:,1])
            loss = loss1 + loss2
            print('loss1: ', loss1.item(), 'loss2: ', loss2.item())
        else:
            raise Exception('Number of networks outputs is wrong')

    elif lossType == 3: #Cross-Entropy
            criterion = nn.CrossEntropyLoss()
            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)

            loss1 = 0; loss2 = 0
            for batch in range(np.shape(pred)[0]):
                loss1 += criterion(pred[batch,:,0:3], torch.argmax(targets[batch,:,0:3].long(), axis=1))
                loss2 += criterion(pred[batch,:,3:6], torch.argmax(targets[batch,:,3:6].long(), axis=1))
            loss1 = loss1/np.shape(pred)[0]
            loss2 = loss2/np.shape(pred)[0]
            loss = loss1 + loss2
            if batch_idx % interval == 0 and print_loss==True:
                print('loss1: ', loss1.item(), 'loss2: ', loss2.item())

    elif lossType == 4: #BCE
            criterion = nn.BCELoss()
            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)
            
            loss1 = criterion(pred[:,:,0:3], targets[:,:,0:3])
            loss2 = criterion(pred[:,:,3:6], targets[:,:,3:6])
            loss = loss1 + loss2
            if batch_idx % interval == 0 and print_loss==True:
                print('loss1: ', loss1.item(), 'loss2: ', loss2.item())

    elif lossType >= 5 and lossType < 9: # Classic Composite Loss as CCC + CE + MSE
            criterion = ConcordanceCorCoeffLoss()
            CCC_1 = criterion(pred[:,:, bins], targets[:,:,0, bins])
            CCC_2 = criterion(pred[:,:,bins*2+1], targets[:,:,1,bins])
            L = nn.MSELoss()
            MSE_1 = L(pred[:,:, bins], targets[:,:,0, bins])
            MSE_2 = L(pred[:,:,bins*2+1], targets[:,:,1,bins])

            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)

            loss1 = 0; loss2 = 0; 
            for batch in range(np.shape(pred)[0]):
                CE = nn.CrossEntropyLoss()
                loss1 += CE(pred[batch,:,0:bins], torch.argmax(targets[batch,:,0:bins].long(), axis=1))
                loss2 += CE(pred[batch,:,(bins+1):(bins*2+1)], torch.argmax(targets[batch,:,(bins+1):(bins*2+1)].long(), axis=1))
            
            if lossType == 5:
                loss1 = 0.5*CCC_1+0.5*loss1/(np.shape(pred)[0])
                loss2 = 0.5*CCC_2+0.5*loss2/(np.shape(pred)[0])
            elif lossType == 6:
                loss1 = 0.5*CCC_1+0.25*loss1/(np.shape(pred)[0])+0.25*MSE_1
                loss2 = 0.5*CCC_2+0.25*loss2/(np.shape(pred)[0])+0.25*MSE_2
            elif lossType == 7:
                loss1 = 2*0.5*CCC_1+0.5*loss1/(np.shape(pred)[0])
                loss2 = CCC_2
            else:
                loss1 = 2*0.5*CCC_1+0.5*loss1/(np.shape(pred)[0])+0.5*MSE_1
                loss2 = CCC_2                

            loss = loss1 + loss2  
            if batch_idx % interval == 0 and print_loss==True:
                print('LOSS: ', loss.item(), 'CCC: ', CCC_1.item(), CCC_2.item(), 'CE: ', loss1.item(), loss2.item(), 'MSE:', MSE_1.item(), MSE_2.item())  

    elif lossType > 9: # Composite with the nCCE (Albadawy 2018)
            criterion = ConcordanceCorCoeffLoss()
            CCC_1 = criterion(pred[:,:, bins], targets[:,:,0, bins])
            CCC_2 = criterion(pred[:,:,bins*2+1], targets[:,:,1,bins])
            L = nn.MSELoss()
            MSE_1 = L(pred[:,:, bins], targets[:,:,0, bins])
            MSE_2 = L(pred[:,:,bins*2+1], targets[:,:,1,bins])
            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)

            def norm_cost(y, y_pred, K, startpts):
                weighted_sum_y, startpts = KMeans(y, K, n_init=5, centroids=startpts)
                weighted_sum_yp, startpts = KMeans(y_pred, K, n_init=5, centroids=startpts)
                # print(weighted_sum_y.shape, weighted_sum_yp.shape)
                return 1+torch.sqrt((torch.square(weighted_sum_y-weighted_sum_yp))), startpts

            centroids_arousal, startpts_ar = norm_cost(targets[:,:,0:bins],pred[:,:,0:bins], bins, startpts_ar)
            centroids_valence, startpts_val = norm_cost(targets[:,:,(bins+1):(bins*2+1)],pred[:,:,(bins+1):(bins*2+1)], bins, startpts_val)

            def modified_CE(logits=None, labels=None, k=None):
                shape_log = logits.shape[0]*logits.shape[1]
                logits = torch.reshape(logits, shape=(shape_log, bins))
                labels = torch.reshape(labels, shape=(shape_log, bins))
                _, max_ = torch.max(logits,axis=1)
                shape_=(max_.shape[0], 1)
                scaled_logits = logits - torch.reshape(max_,shape=shape_)
                normalized_logits = scaled_logits - torch.reshape(torch.logsumexp(scaled_logits,1),shape=shape_) 
                return torch.mean(-torch.sum(labels*normalized_logits,1))
            
            l2_lambda = 0.01
            # l2_reg = torch.tensor(0.)
            # l2_reg.to(dev)

            for i, param in enumerate(params):
                if i ==0:
                    l2_reg = torch.norm(param)
                else:
                    l2_reg += torch.norm(param)
            L2 = l2_lambda * l2_reg
            # print('L2:', L2.shape)

            c20_ar = centroids_arousal
            c20_val = centroids_valence
            CE20_ar = modified_CE(logits=pred[:,:,0:bins], labels=targets[:,:,0:bins], k=20)
            CE20_val = modified_CE(logits=pred[:,:,(bins+1):(bins*2+1)], labels=targets[:,:,(bins+1):(bins*2+1)], k=20)
            loss20_ar = torch.mean(c20_ar * CE20_ar) + L2 #+ self.config.l2_beta * tf.nn.l2_loss(self.weights['w10']))
            loss20_val = torch.mean(c20_val * CE20_val) + L2 #+ self.config.l2_beta * tf.nn.l2_loss(self.weights['w10']))


            if lossType == 9:
                loss = CCC_1 + CCC_2 + loss20_ar/10 + loss20_val/10
            else:
                loss = CCC_1 + CCC_2 + loss20_ar/10 + loss20_val/10 + MSE_1 + MSE_2
            if batch_idx % interval == 0 and print_loss==True:
                print('LOSS: ', loss.item(), 'CCE_ar: ', loss20_ar.item(), 'CCE_val: ', loss20_val.item(), 'CCC: ', CCC_1.item(), CCC_2.item(), 'L2: ', L2.item(), 'MSE: ', MSE_1.item(), MSE_2.item())  
    else:
        raise Exception('LossType unknown')

    return loss, startpts_ar, startpts_val
     