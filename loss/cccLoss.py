
import torch
import torch.nn as nn


class ConcordanceCorCoeffLoss(nn.Module):
    
    def __init__(self):
        super(ConcordanceCorCoeffLoss, self).__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std
        self.eps = 1e-10
        
        
        
    def forward(self, prediction, ground_truth):
        
        prediction = torch.reshape(prediction, (-1,))
        ground_truth = torch.reshape(ground_truth, (-1,))
        
        mean_gt = self.mean (ground_truth) + self.eps
        mean_pred = self.mean (prediction) + self.eps
        var_gt = self.var (ground_truth) + self.eps
        var_pred = self.var (prediction) + self.eps
        
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        
        cor = self.sum(v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2)))
        sd_gt = self.std(ground_truth) + self.eps
        sd_pred = self.std(prediction) + self.eps
        numerator=2*cor*sd_gt*sd_pred
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        ccc = numerator/denominator
        lossValue = 1 - ccc
        return lossValue

