from logging import logProcesses
import numpy as np
import torch
import torch.nn as nn
# from sklearn.cluster import KMeans
from Kmeans import KMeans

def compute_loss_CCE(pred, targets, criterion, lossType: int, startpts_ar, startpts_val, params):

    #pred, targets torch tensor (batch_size, sample_len, no_classes)
    # print('Sizes: ', np.shape(pred), np.shape(targets))
    if lossType == 1: #MSE
        
        loss = criterion(pred, targets) 
        # print('loss: ', loss.item())
        
    elif lossType == 2: # CCC
            
        if len(targets.size()) == 2:
            print("T2", np.shape(pred), np.shape(targets))
            loss1 = criterion(pred, targets)
            loss = loss1
            print('loss1: ', loss1.item())
            
        elif len(targets.size()) == 3:
            # print("T3", np.shape(pred), np.shape(targets))
            loss1 = criterion(pred[:,:,0], targets[:,:,0])
            loss2 = criterion(pred[:,:,1], targets[:,:,1])
            loss = loss1 + loss2
            print('loss1: ', loss1.item(), 'loss2: ', loss2.item())
        # elif len(targets.size()) == 3:
        #     # print("T3", np.shape(pred), np.shape(targets))
        #     loss1 = criterion(pred[:,:,0], targets[:,:,0])
        #     loss2 = criterion(pred[:,:,1], targets[:,:,1])
        #     loss = loss1 + loss2
        #     print('loss1: ', loss1.item(), 'loss2: ', loss2.item())
            
        else:
            raise Exception('Number of networks outputs is wrong')

    elif lossType == 3: #Cross-Entropy
            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)
            # print(pred.size(), targets.size())
            # print(targets[:,-1,0:7].long())

            loss1 = 0; loss2 = 0
            # print('LOSS SHAPES: ', np.shape(pred)[0], np.shape(pred[:,:,0:7]), np.shape(torch.argmax(targets[:,:,0:7].long(), axis=2)))
            for batch in range(np.shape(pred)[0]):
                loss1 += criterion(pred[batch,:,0:3], torch.argmax(targets[batch,:,0:3].long(), axis=1))
                loss2 += criterion(pred[batch,:,3:6], torch.argmax(targets[batch,:,3:6].long(), axis=1))
            loss1 = loss1/np.shape(pred)[0]
            loss2 = loss2/np.shape(pred)[0]
            loss = loss1 + loss2
            print('loss1: ', loss1.item(), 'loss2: ', loss2.item())

    elif lossType == 4: #BCE
            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)
            # print(pred.size(), targets.size())
            # print(targets[:,-1,0:7].long())
            
            loss1 = criterion(pred[:,:,0:3], targets[:,:,0:3])
            loss2 = criterion(pred[:,:,3:6], targets[:,:,3:6])
            loss = loss1 + loss2
            print('loss1: ', loss1.item(), 'loss2: ', loss2.item())

    # elif lossType == 5: #composite
    #         targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)
    #         # print(pred.size(), targets.size())
    #         # print(targets[:,-1,0:7].long())

    #         loss1 = 0; loss2 = 0; loss1_2 =0; loss2_2 = 0
    #         # print('LOSS SHAPES: ', np.shape(pred)[0], np.shape(pred[:,:,0:7]), np.shape(torch.argmax(targets[:,:,0:7].long(), axis=2)))
    #         for batch in range(np.shape(pred)[0]):
    #             CE = nn.CrossEntropyLoss()
    #             loss1 += CE(pred[batch,:,0:7], torch.argmax(targets[batch,:,0:7].long(), axis=1))
    #             loss2 += CE(pred[batch,:,8:15], torch.argmax(targets[batch,:,8:15].long(), axis=1))
    #         loss1_2 = criterion(pred[:,:,7], targets[:,:,7])
    #         loss2_2 = criterion(pred[:,:,15], targets[:,:,15])
    #         # print('LOSS: ', (loss1.item()/80 + loss2.item()/80 + loss1_2.item()/5 + loss2_2.item()/5), 'loss1: ', loss1.item()/80, 'loss2: ', loss2.item()/80, 'loss1_2: ', loss1_2.item()/5, 'loss2_2: ', loss2_2.item()/5)  
    #         loss1 = (loss1+0*loss1_2)/(np.shape(pred)[0])
    #         loss2 = (loss2+0*loss2_2)/(np.shape(pred)[0])
    #         loss = loss1 + loss2  
    #         print('LOSS: ', loss.item())  

    elif lossType == 5: #composite
            bins =20
            # print(pred.size(), targets.size())
            loss1_2 = criterion(pred[:,:, bins], targets[:,:,0, bins])
            loss2_2 = criterion(pred[:,:,bins*2+1], targets[:,:,1,bins])
            L = nn.MSELoss()
            loss1_MSE = L(pred[:,:, bins], targets[:,:,0, bins])
            loss2_MSE = L(pred[:,:,bins*2+1], targets[:,:,1,bins])

            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)

            # print(pred.size(), targets.size())
            # print(targets[:,-1,0:7].long())

            loss1 = 0; loss2 = 0; 
            # print('LOSS SHAPES: ', np.shape(pred)[0], np.shape(pred[:,:,0:7]), np.shape(torch.argmax(targets[:,:,0:7].long(), axis=2)))
            for batch in range(np.shape(pred)[0]):
                CE = nn.CrossEntropyLoss()
                loss1 += CE(pred[batch,:,0:bins], torch.argmax(targets[batch,:,0:bins].long(), axis=1))
                loss2 += CE(pred[batch,:,(bins+1):(bins*2+1)], torch.argmax(targets[batch,:,(bins+1):(bins*2+1)].long(), axis=1))
            
            # loss1 = 0.5*loss1_2+loss1/(30*np.shape(pred)[0])
            # loss2 = 0.5*loss2_2+loss2/(30*np.shape(pred)[0])
            # loss1 = 2*0.5*loss1_2+0.5*loss1/(np.shape(pred)[0])+0.5*loss1_MSE
            # loss2 = loss2_2
            loss1 = 0.5*loss1_2+0.25*loss1/(np.shape(pred)[0])+0.25*loss1_MSE
            loss2 = 0.5*loss2_2+0.25*loss2/(np.shape(pred)[0])+0.25*loss2_MSE
            # loss1 = 0.5*loss1_2+0.5*loss1/(np.shape(pred)[0])
            # loss2 = 0.5*loss2_2+0.5*loss2/(np.shape(pred)[0])
            loss = loss1 + loss2  
            print('LOSS: ', loss.item(), 'loss1: ', loss1.item()/10*np.shape(pred)[0], 'loss2: ', loss2.item()/10*np.shape(pred)[0], 'loss1_2: ', loss1_2.item(), 'loss2_2: ', loss2_2.item())  
            # print('LOSS: ', loss.item())  


    # elif lossType == 5: #composite
    #         loss1_2 = criterion(pred[:,:,7], targets[:,:,0,7])
    #         loss2_2 = criterion(pred[:,:,15], targets[:,:,1,7])
    #         targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)
    #         # print(pred.size(), targets.size())
    #         # print(targets[:,-1,0:7].long())

    #         loss1 = 0; loss2 = 0; 
    #         # print('LOSS SHAPES: ', np.shape(pred)[0], np.shape(pred[:,:,0:7]), np.shape(torch.argmax(targets[:,:,0:7].long(), axis=2)))
    #         for batch in range(np.shape(pred)[0]):
    #             CE = nn.CrossEntropyLoss()
    #             loss1 += CE(pred[batch,:,0:7], torch.argmax(targets[batch,:,0:7].long(), axis=1))
    #             loss2 += CE(pred[batch,:,8:15], torch.argmax(targets[batch,:,8:15].long(), axis=1))
            
    #         loss1 = 0.5*loss1_2+loss1/(30*np.shape(pred)[0])
    #         loss2 = 0.5*loss2_2+loss2/(30*np.shape(pred)[0])
    #         loss = loss1 + loss2  
    #         print('LOSS: ', loss.item(), 'loss1: ', loss1.item()/10*np.shape(pred)[0], 'loss2: ', loss2.item()/10*np.shape(pred)[0], 'loss1_2: ', loss1_2.item(), 'loss2_2: ', loss2_2.item())  
    #         # print('LOSS: ', loss.item())  

    elif lossType == 6: #CCE Albadawy 2018
            bins =20
            # print(pred.size(), targets.size())
            loss1_2 = criterion(pred[:,:, bins], targets[:,:,0, bins])
            loss2_2 = criterion(pred[:,:,bins*2+1], targets[:,:,1,bins])
            L = nn.MSELoss()
            loss1_MSE = L(pred[:,:, bins], targets[:,:,0, bins])
            loss2_MSE = L(pred[:,:,bins*2+1], targets[:,:,1,bins])
            targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)

            def norm_cost(y, y_pred, K, startpts):
                weighted_sum_y, startpts = KMeans(y, K, n_init=5, centroids=startpts)
                weighted_sum_yp, startpts = KMeans(y_pred, K, n_init=5, centroids=startpts)
                # print(weighted_sum_y.shape, weighted_sum_yp.shape)
                return 1+torch.sqrt((torch.square(weighted_sum_y-weighted_sum_yp))), startpts

            centroids_arousal, startpts_ar = norm_cost(targets[:,:,0:bins],pred[:,:,0:bins], bins, startpts_ar)#KMeans(pred[:,:,0:bins], K=20, n_init=5, centroids=startpts_ar)
            centroids_valence, startpts_val = norm_cost(targets[:,:,(bins+1):(bins*2+1)],pred[:,:,(bins+1):(bins*2+1)], bins, startpts_val)#KMeans(pred[:,:,(bins+1):(bins*2+1)], K=20, n_init=5, centroids=startpts_val)

            # targets = torch.cat((targets[:, :, 0, :], targets[:, :, 1, :]), dim=2)
            # print(pred.size(), targets.size())
            # print(targets[:,-1,0:7].long())
            def modified_CE(logits=None, labels=None, k=None):
                # scaled_logits = logits - tf.reshape(tf.reduce_max(logits,1),shape=(7500,1))
                # normalized_logits = scaled_logits - tf.reshape(tf.reduce_logsumexp(scaled_logits,1),shape=(7500,1)) 
                shape_log = logits.shape[0]*logits.shape[1]
                logits = torch.reshape(logits, shape=(shape_log, bins))
                labels = torch.reshape(labels, shape=(shape_log, bins))
                _, max_ = torch.max(logits,axis=1)
                shape_=(max_.shape[0], 1)
                # print('max_:', max_.shape, 'shape_:', shape_, 'logits:', logits.shape)
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


            # loss1 = 0; loss2 = 0; 
            # # print('LOSS SHAPES: ', np.shape(pred)[0], np.shape(pred[:,:,0:7]), np.shape(torch.argmax(targets[:,:,0:7].long(), axis=2)))
            # for batch in range(np.shape(pred)[0]):
            #     CE = nn.CrossEntropyLoss()
            #     loss1 += CE(pred[batch,:,0:bins], torch.argmax(targets[batch,:,0:bins].long(), axis=1))
            #     loss2 += CE(pred[batch,:,(bins+1):(bins*2+1)], torch.argmax(targets[batch,:,(bins+1):(bins*2+1)].long(), axis=1))

            # loss1 = 2*0.5*loss1_2+0.5*loss1/(np.shape(pred)[0])#+0.25*loss1_MSE
            # loss2 = loss2_2

            # loss = loss1_2 + loss2_2 + loss20_ar/10 + loss20_val/10
            loss = loss1_2 + loss2_2 + loss20_ar/10 + loss20_val/10 + loss1_MSE +loss2_MSE
            print('LOSS: ', loss.item(), 'CCE_ar: ', loss20_ar.item(), 'CCE_val: ', loss20_val.item(), 'loss1_2: ', loss1_2.item(), 'loss2_2: ', loss2_2.item(), 'L2: ', L2.item(), 'MSE: ', loss1_MSE.item(), loss2_MSE.item())  
            # print('LOSS: ', loss.item())  


    return loss, startpts_ar, startpts_val
     