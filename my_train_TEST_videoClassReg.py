
import argparse
import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.signal import medfilt

#from emotionRec.utils import compute_bias, compute_scale, get_median_filt_win
from modelVideoClassReg import EmotionModelVideo
from utils2 import load_model, get_ccc, get_logger, save_as_json, CheckpointSaver, get_lr, update_logger_batch, get_targets, compute_bias, compute_scale, get_median_filt_win
from dataloaders import get_data_loaders
from optim_utils import get_optimizer, CosineScheduler
from cccLoss import ConcordanceCorCoeffLoss
import sklearn
# from CE import CrossEntropyLoss
from compute_loss_CCE import compute_loss_CCE

import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description='Emotion AV Training')
    parser.add_argument('--dataset', type=str, default='sewa', choices = ['sewa', 'sewa_video', 'MSP_video', 'RECOLA'])
    # parser.add_argument('--modality', type=str, default='raw_audio', choices = ['raw_audio'])
   
    parser.add_argument('--model-path', default='', help='loaded model path')
    parser.add_argument('--logging-dir', type = str, default = '.', help = 'path to the directory in which to save the log file')

# model architecture definition
    parser.add_argument('--every-frame', default=True, action='store_true', help='RNN predicition based on every frame')
    # parser.add_argument('--resnet-pretrain', default = False, action = 'store_true', help = 'Use an ImageNet-pretrained model to initialize the Resnet part')
    # parser.add_argument('--backbone-type', type = str, default = 'resnet', help = 'What architecture to use for the chunk between the frontend and the backend' )
    parser.add_argument('--relu-type', type = str, default = 'relu', choices = ['relu','prelu','leaky'], help = 'what relu to use' )
    parser.add_argument('--hiddenDim', type = int, default = 256, help = "Number of hidden states on the GRU or filters in the TCN")
    # parser.add_argument('--avg-pool-downsample', default=False, action='store_true', help="Whether use avg pooling 2d on downsample layers")
    # parser.add_argument('--gamma-zero', default=False, action='store_true', help="Initialize gamma of last BN in ResNet block to zeros")


    # optimizer options
    parser.add_argument('--optimizer',type=str, default = 'adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--allow-size-mismatch', default=True, action='store_true', 
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr-scheduler-type',type=str, default = 'cosine', choices = ['cosine'])
    parser.add_argument('--min-lr', default=1e-6, type=float, help='minimal learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    # parser.add_argument('--reduce-and-reload', default=False, action='store_true', help='reload model if the validation accuracy decreases')
    # parser.add_argument('--reload-state', default=False, action='store_true', help='reload optimizer state')

    # transform options
    # parser.add_argument('--fixed-length', default=True, action='store_true', help='Set to True to use fixed length sequences (no variable length augmentation')
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # other vars
    
    parser.add_argument('--batch-size', default=5, type=int, help='mini-batch size (default: 32)') 
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--gpu-idx', default = 0, type = int )
    parser.add_argument('--interval', default=10, type=int, help='display interval')

    
    parser.add_argument('--annot_list', default=[0,1,2,4,5], action='store_true', help='List with annotators IDs')
    parser.add_argument('--annot_type', default = 'V', help = 'A, V or AV, applies to SEWA only')
    parser.add_argument('--loss_type', default=5, type=int, help='1=MSE, 2=CCC, 3=Cross-Entropy, 4=BCE, 5=composite, 6=CCE')
    parser.add_argument('--clip_length', default=0, type=float, help='length of audio clip in seconds randomply picked from each training file, if 0 then use all segments')
    parser.add_argument('--segments_per_file', default=3, type=int, help='Number of segments to pick from training files')
    parser.add_argument('--output_type', default = 'Arousal_Valence', help = ' "Arousal", "Valence" or "Arousal_Valence" ')


    args = parser.parse_args()

    return args

args = parse_arguments()
bins = 20
args.strpoints_ar = []
args.strpoints_val = []

if 'sewa' in args.dataset:
    args.fps = 50
    args.annot_list = [0] # this is not really needed - we have averaged annotations
    print(args.dataset)

    args.annot_dir = {'train': '/fsx/marijajegorova/pre-process/step2_affine_transformation/SEWA_annot_binned20/Train_Set_binned', 
                        'val' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/SEWA_annot_binned20/Validation_Set_binned',
                        'test' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/SEWA_annot_binned20/Test_Set_binned'}

    args.video_dir = {'train': '/fsx/marijajegorova/pre-process/step2_affine_transformation/sewa13crop_videos_npz_preprocessed_full/trNames', 
            'val' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/sewa13crop_videos_npz_preprocessed_full/valNames',
            'test' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/sewa13crop_videos_npz_preprocessed_full/testNames'}
    
    
    if args.clip_length == 0:  
        args.clip_length_dict = {'train': 0, 'val' :0, 'test' : 0}
        args.segments_per_file_dict = {'train': 1, 'val' :1, 'test' : 1}
        
    else:
        args.clip_length_dict = {'train': args.clip_length, 'val' :0, 'test' : 0}
        args.segments_per_file_dict = {'train': args.segments_per_file, 'val' :1, 'test' : 1}

elif 'RECOLA' in args.dataset:
    args.fps = 25
    args.annot_list = [0] # this is not really needed - we have averaged annotations
    print(args.dataset)

    args.annot_dir = {'train': '/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_annot_20secs_binned_new/trNames_binned', 
                        'val' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_annot_20secs_binned_new/testNames_binned',
                        'test' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_annot_20secs_binned_new/testNames_binned'}
    
    args.video_dir = {'train': '/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_split/trNames', 
                'val' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_split/testNames',
                'test' : '/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_split/testNames'}            

    if args.clip_length == 0:  
        args.clip_length_dict = {'train': 0, 'val' :0, 'test' : 0}
        args.segments_per_file_dict = {'train': 1, 'val' :1, 'test' : 1}
        
    else:

        args.clip_length_dict = {'train': args.clip_length, 'val' :0, 'test' : 0}
        args.segments_per_file_dict = {'train': args.segments_per_file, 'val' :1, 'test' : 1}
elif 'MSP_video' in args.dataset:
    args.fps = 50
    args.annot_list = [0] # this is not really needed - we have averaged annotations
    print(args.dataset)

    
    args.annot_dir = {'train': '/fsx/marijajegorova/AE1/AE_native/AE/MSP_annotations_200_trimmed_norm_bin20/Train_Set_binned', 
                        'val' : '/fsx/marijajegorova/AE1/AE_native/AE/MSP_annotations_200_trimmed_norm_bin20/Validation_Set_binned',
                        'test' : '/fsx/marijajegorova/AE1/AE_native/AE/MSP_annotations_200_trimmed_norm_bin20/Test_Set_binned'}     
    args.video_dir = {'train': '/fsx/marijajegorova/dino/MSP_widerCrop13_24window_attention_mp4_preprocessed/trNames', 
                        'val' : '/fsx/marijajegorova/dino/MSP_widerCrop13_24window_attention_mp4_preprocessed/valNames',
                        'test' : '/fsx/marijajegorova/dino/MSP_widerCrop13_24window_attention_mp4_preprocessed/testNames'}
    if args.clip_length == 0:
        args.clip_length_dict = {'train': 0, 'val' :0, 'test' : 0}
        args.segments_per_file_dict = {'train': 1, 'val' :1, 'test' : 1}
        
    else:

        args.clip_length_dict = {'train': args.clip_length, 'val' :0, 'test' : 0}
        args.segments_per_file_dict = {'train': args.segments_per_file, 'val' :1, 'test' : 1}
   

if args.output_type == 'Arousal':
    args.class_names = ['Arousal']
elif args.output_type == 'Valence':
    args.class_names = ['Valence']
elif args.output_type == 'Arousal_Valence':
    args.class_names = ['Arousal', 'Valence']
else:
    raise Exception('Wrong output Name')

args.n_classes = 2*(bins+1);#16;#len(args.class_names)

args.weight_decay = 1e-4 if args.optimizer != 'adamw' else 1e-2
args.width_mult = 1
args.in_stream = 'raw_audio'

print('Dataset: ', args.dataset)
print('AnnotList: ', args.annot_list)
print('Optimizer: ', args.optimizer)
print('Batch Size: ', args.batch_size)
print('Annot Type: ', args.annot_type)
print('LR: ', args.lr)
print('Loss Type: ', args.loss_type)
print('Output Type: ', args.output_type)
print('Clip Length: ', args.clip_length)
print('SegmentsPerFile: ', args.segments_per_file)        

args.preprocessing_func = []
# args.logging_dir = '/fsx/stavrosp/temp_data'
# args.model_path = ''

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache() 

# pick device (which GPU or CPU)
#device = torch.device("cuda:{}".format(args.gpu_idx) if ( torch.cuda.is_available() and args.gpu_idx >= 0 ) else "cpu")
device = torch.device('cuda')

def get_save_folder( ):
    
    folder_name = 'training'

    append_str = '_every_frame' if args.every_frame else '_last_frame'
    folder_name += append_str
    
    save_path = os.path.join(args.logging_dir, folder_name, datetime.datetime.now().isoformat().split('.')[0])
    
    os.makedirs(save_path, exist_ok=True)
        
    return save_path

def test(model, dset_loader, criterion, logger, phase, ppParams = {}):

    model.eval()
    running_loss = 0.
    running_all = 0.

    stats_per_video = [ {} for _ in range(args.n_classes)]
    stats_conc = [ {} for _ in range(args.n_classes)]

    targets_all_videos_list = np.zeros((2, 670*200, (bins+1)))#[ [] for _ in range(args.n_classes)]
    pred_all_videos_list = np.zeros((2, 670*200, (bins+1)))#[ [] for _ in range(args.n_classes)]

    if phase == 'val':
        ppParams = [ {} for _ in range(args.n_classes)]
    

    with torch.no_grad():
        # errors = 0
        targets_all_videos_list = []
        pred_all_videos_list = []
        for batch_idx, data in enumerate(dset_loader):
            print(batch_idx)
            # we first load all targets and compute predictions for all videos
            inputs = data['video']
            
            targets = get_targets(data, args.output_type)
            # print('TARGETS: ', np.shape(targets))
            lengths = tuple([inputs.size(1)] * len(targets))
            # print('LENGTHS: ', lengths)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, lengths)
            
            # we need this because sometimes targets length (2nd element) might be longer by 1 than the outputs length
            if outputs.size(1) > targets.size(1):
                outputs = outputs[:,:targets.size(1),:]
            elif targets.size(1) > outputs.size(1):
                targets = targets[:,:outputs.size(1),:]
            
            outputs = outputs.double()
            
            # print('BATCH: ', batch_idx)
            # loss = compute_loss(outputs, targets, criterion, args.loss_type)
            loss, args.strpoints_ar, args.strpoints_val = compute_loss_CCE(outputs, targets, criterion, args.loss_type, args.strpoints_ar, args.strpoints_val, model.parameters()) 
                
            out = outputs.cpu().numpy()
            trg = targets.cpu().numpy()
            trg = np.concatenate((trg[:, :, 0, :], trg[:, :, 1, :]), 2)
            # print('TRG: ', np.shape(trg))
            
            if pred_all_videos_list == []:
                # print('BATCH 1')
                targets_all_videos_list = np.concatenate((trg[:,:,:(bins+1)], trg[:,:,(bins+1):2*(bins+1)]), axis=0)
                pred_all_videos_list = np.concatenate((out[:,:,:(bins+1)], out[:,:,(bins+1):(2*(bins+1))]), axis=0)
            else:
                targets_all_videos_list_add = np.concatenate((trg[:,:,:(bins+1)], trg[:,:,(bins+1):2*(bins+1)]), axis=0)
                pred_all_videos_list_add = np.concatenate((out[:,:,:(bins+1)], out[:,:,(bins+1):(2*(bins+1))]), axis=0)
                targets_all_videos_list = np.concatenate((targets_all_videos_list, targets_all_videos_list_add), 1)
                pred_all_videos_list = np.concatenate((pred_all_videos_list, pred_all_videos_list_add), 1)

            # stastics
            running_loss += loss.item() * inputs.size(0)
            running_all += outputs.size(0)
                
    #We compute the performance measures per video and after concatenating all predictions/targets

    for ind in range(2):

        single_pred_list = pred_all_videos_list[ind,:,:]
        single_target_list = targets_all_videos_list[ind,:,:]

        single_pred_list_ = np.argmax(single_pred_list[:, :bins], axis=1)
        single_target_list_ = np.argmax(single_target_list[:, :bins], axis=1)
        single_pred_list_reg = single_pred_list[:, bins]
        single_target_list_reg = single_target_list[:, bins]

        # concatenate all predictions and targets from all videos for computing the final stats
        all_preds_conc = np.hstack(single_pred_list)
        all_trg_conc = np.hstack(single_target_list)

        all_preds_conc_reg = np.hstack(single_pred_list_reg)
        all_trg_conc_reg = np.hstack(single_target_list_reg)

        corr_per_video = pearsonr(single_pred_list[:][bins], single_target_list[:][bins])[0]
        ccc_per_video = [get_ccc(single_pred_list[i][bins:(bins+1)], single_target_list[i][bins:(bins+1)]) for i in range(len(single_pred_list))]
        # print('last shape:', np.shape(single_pred_list))
        f1_per_video = np.zeros(bins)
        # single_pred_list = (single_pred_list > 0.5) 
        
        for k in range(bins):
            try:
                f1_per_video[k] = sklearn.metrics.f1_score((single_pred_list[:][k]> 0.5) ,single_target_list[:][k], average='weighted')
            except:
                print('F1 per video is not computed.')
        mse_per_video = [mean_squared_error(single_pred_list[i][bins:(bins+1)], single_target_list[i][bins:(bins+1)]) for i in range(len(single_pred_list))]
        mae_per_video = [mean_absolute_error(single_pred_list[i][bins:(bins+1)], single_target_list[i][bins:(bins+1)]) for i in range(len(single_pred_list))]
                
            
        stats_per_video[ind]['Corr'] = np.nanmean(corr_per_video)
        stats_per_video[ind]['CCC'] = np.mean(ccc_per_video)
        stats_per_video[ind]['F1'] = np.mean(f1_per_video)
        stats_per_video[ind]['F1_7'] = f1_per_video
        stats_per_video[ind]['MAE'] = np.mean(mse_per_video)
        stats_per_video[ind]['MSE'] = np.mean(mae_per_video)
        
        all_preds_conc = (all_preds_conc > 0.5) 
        
        stats_conc[ind]['CCC'] = get_ccc(all_preds_conc_reg, all_trg_conc_reg)
        all_preds_conc_f1 = all_preds_conc[[0 and np.mod(np.arange(all_preds_conc.size),bins)!=0]]
        all_trg_conc_f1 = all_trg_conc[[0 and np.mod(np.arange(all_trg_conc.size),bins)!=0]]
        print(np.count_nonzero(( all_preds_conc_f1!=0) & (all_preds_conc_f1!=1)))
        print(np.count_nonzero((all_trg_conc_f1!=0) & (all_trg_conc_f1!=1)))
        try:
            f1_per_video_ = np.zeros(bins)
            for k in range(bins):
                f1_per_video_[k] = sklearn.metrics.f1_score((all_preds_conc_f1[:][k]> 0.5), all_trg_conc_f1[:][k], average='weighted')
            stats_conc[ind]['F1'] = np.mean(f1_per_video_)
            stats_conc[ind]['F7'] = f1_per_video_
        except:
            stats_conc[ind]['F1'] = sklearn.metrics.f1_score((np.hstack(all_preds_conc_f1)> 0.5), all_trg_conc_f1, average='weighted')
            stats_conc[ind]['F7'] = 0
        

        stats_conc[ind]['CM'] = sklearn.metrics.confusion_matrix(single_target_list_,single_pred_list_)
        #sklearn.metrics.f1_score(all_preds_conc, all_trg_conc, average='weighted')
        stats_conc[ind]['Corr'] = pearsonr(all_preds_conc[::bins], all_trg_conc[::bins])[0]
        stats_conc[ind]['MSE'] = mean_squared_error(all_preds_conc[::bins], all_trg_conc[::bins])
        stats_conc[ind]['MAE'] = mean_absolute_error(all_preds_conc[::bins], all_trg_conc[::bins])
        

    return  stats_per_video, stats_conc, ppParams, f1_per_video_


def train_one_epoch(model, dset_loaders: dict, criterion, epoch: int, phase: str, optimizer, logger):


    lr = get_lr(optimizer)

    logger.info('-' * 10)
    logger.info(f'Epoch {epoch}/{args.epochs-1}')
    logger.info(f'Current learning rate: {lr}')

    model.train()
    running_loss = 0.
    sum_corr_per_seq = [0., 0.]
    sum_ccc_per_seq = [0., 0.]
    sum_f1_per_seq = [0., 0.]
    running_all = 0.
    sum_mae_per_seq = [0., 0.]
    sum_mse_per_seq = [0., 0.]
    
    for batch_idx, data in enumerate(dset_loaders[phase]):

        inputs = data['video']
            
        targets = get_targets(data, args.output_type)
        # print('target: ', np.shape(targets))
   
        lengths = tuple([inputs.size(1)] * len(targets))
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        # print('LENGTHS: ', lengths)
        outputs = model(inputs, lengths)

        outputs = outputs.double()

        loss, args.strpoints_ar, args.strpoints_val = compute_loss_CCE(outputs, targets, criterion, args.loss_type, args.strpoints_ar, args.strpoints_val, model.parameters())
    
        with torch.no_grad():
            out = outputs.cpu().numpy()
            trg = targets.cpu().numpy()
            trg = np.concatenate((trg[:, :, 0, :], trg[:, :, 1, :]), 2)
            # print('trg: ', np.shape(trg))

        for ind in range(2):#range(args.n_classes):
                    
            if args.n_classes == 1:
                single_out = out[:,:,0]
                single_targ = trg
            elif args.n_classes == 2:
                single_out = out[:,:,ind]
                single_targ = trg[:,:,ind]
            elif args.n_classes >2 :
                
                # print('SHAPE: ', np.shape(out), np.shape(trg))
                single_out = out[:,:,(ind*(bins+1)):(ind*(bins+1)+(bins+1))]
                single_targ = trg[:,:,(ind*(bins+1)):(ind*(bins+1)+(bins+1))]
                # print('SINGLE SHAPE: ', np.shape(single_out), np.shape(single_targ))
                single_targ_ = np.argmax(single_targ, axis=2)
                single_out_ = np.argmax(single_out, axis=2)
                # torch.cat((x1, x2), dim=2)
                
                
                
                    
            ccc_per_seq = [get_ccc(single_out[i,:],single_targ[i,:]) for i in range(len(single_out))]
            f1_per_seq = [sklearn.metrics.f1_score(single_out_[i,:],single_targ_[i,:], average='weighted') for i in range(len(single_out_))]
            # corr_per_seq = [pearsonr(single_out[i,:], single_targ[i,:])[0] for i in range(len(single_out))]
            mse_per_seq = 0; #[mean_squared_error(single_out[i,:], single_targ[i,:]) for i in range(len(single_out))]
            mae_per_seq = 0;#[mean_absolute_error(single_out[i,:], single_targ[i,:]) for i in range(len(single_out))]
            
            # sum_corr_per_seq[ind] = sum_corr_per_seq[ind] + np.nansum(corr_per_seq)
            sum_corr_per_seq = [0]
            sum_f1_per_seq[ind] = sum_f1_per_seq[ind] + np.sum(f1_per_seq)
            sum_ccc_per_seq[ind] = sum_ccc_per_seq[ind] + np.sum(ccc_per_seq)
            sum_mae_per_seq[ind] = sum_mae_per_seq[ind] + np.sum(mae_per_seq)
            sum_mse_per_seq[ind] = sum_mse_per_seq[ind] + np.sum(mse_per_seq) 
 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*inputs.size(0)
        running_all += inputs.size(0)

        #TODO: we print stats for arousal ONLY
        # -- log running stats every x batches
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            # update_logger_batch( args, logger, dset_loaders[phase], batch_idx, running_loss, sum_corr_per_seq[0], sum_ccc_per_seq[0], sum_mae_per_seq[0], sum_mse_per_seq[0], running_all)
            update_logger_batch( args, logger, dset_loaders[phase], batch_idx, running_loss, sum_corr_per_seq[0], sum_ccc_per_seq[0], sum_mae_per_seq[0], sum_mse_per_seq[0], running_all)
            
    for ind, name in enumerate(args.class_names):
        
        # -- log epoch statistics
        logger.info('{name}: {phase} Epoch:\t{epoch}\tLoss: {running_loss / running_all}\t Corr:{sum_corr_per_seq[ind] / running_all} \tCCC {sum_ccc_per_seq[ind] / running_all} \t MAE:{sum_mae_per_seq[ind] / running_all} \tMSE {sum_mse_per_seq[ind] / running_all}')
    
    return model

def get_model(args, device):
    if args.dataset == 'sewa':
        model = EmotionModel(n_classes=args.n_classes, device=device).to(device)
    else:
        model = EmotionModelVideo(n_classes=args.n_classes, device=device).to(device)
    return model

def main():

    save_path = get_save_folder( )
    print(f"Model and log being saved in: {save_path}")
    logger = get_logger(save_path)
    save_as_json( vars(args), f'{save_path}/args.json')

    ckpt_saver = CheckpointSaver(save_path)

    model = get_model(args, device)


    optimizer = get_optimizer(args.optimizer, model.parameters(), 
                              lr=args.lr, weight_decay=args.weight_decay)

    dset_loaders = get_data_loaders(args)

    if args.loss_type == 1: 
        criterion = nn.MSELoss()
    elif args.loss_type == 2:
        criterion = ConcordanceCorCoeffLoss()
    elif args.loss_type == 3:
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 4:
        criterion = nn.BCELoss()
    elif args.loss_type == 5:
        criterion = ConcordanceCorCoeffLoss()
    elif args.loss_type == 6:
        criterion = ConcordanceCorCoeffLoss()

    else:
        raise Exception('Prediction type know known')

    if args.lr_scheduler_type == 'cosine':
        scheduler = CosineScheduler( args.lr, args.epochs )
    else:
        raise Exception('Scheduler option not implemented!!')
        
    corr_val_per_video_per_epoch_arousal = []
    ccc_val_per_video_per_epoch_arousal = []
    
    ccc_val_all_conc_arousal = []
    corr_val_all_conc_arousal = []
    ccc_val_all_conc_valence = []
    corr_val_all_conc_valence = []
    
    best_params = []
    
    current_perf_per_epoch = []
    current_perf_video_per_epoch = []
    best_epoch = 0
    
    # save best performance, initialise at -Inf
    best_perf = - math.inf
    
    #load model if path is given
    if args.model_path != '':
        model = load_model(args.model_path, model, args.dataset, allow_size_mismatch = args.allow_size_mismatch )
        print('Model was succesfully loaded')
        
    epoch = 0
    f1 = np.zeros((args.epochs, 2, bins))
    f1_7 = np.zeros((args.epochs, 2, bins))
    try:
        os.mkdir('CM20_LR{}_batch{}'.format(args.lr, args.batch_size))
    except:
        print('DONE')

    CCC_ = np.zeros((2, args.epochs))
    while epoch < args.epochs:
        if ( get_lr(optimizer) > args.min_lr ):
            
            # -- train for 1 epoch and save checkpoint
            phase = 'train'
            # try:
            model = train_one_epoch(model, dset_loaders, criterion, epoch, phase, optimizer, logger)
            # except:
            #     print('skipped a batch')
                
            
            # -- validate
            phase = 'val'
            stats_per_video_val, stats_conc_val, ppParams, f1_per_video = test(model, dset_loaders[phase], criterion, logger, phase)

            for ind, name in enumerate(args.class_names):
                logger.info(f'{name} - Epoch:\t{epoch} , LR: {get_lr(optimizer)}')
                logger.info(f'{name} - Mean of all Validation Videos: Corr: {np.mean(stats_per_video_val[ind]["Corr"])}, CCC: {np.mean(stats_per_video_val[ind]["CCC"])}, MAE: {np.mean(stats_per_video_val[ind]["MAE"])}, MSE: {np.mean(stats_per_video_val[ind]["MSE"])}')  
                logger.info(f'{name} - All validation videos concatenated: Corr: {stats_conc_val[ind]["Corr"]}, F1: {stats_conc_val[ind]["F1"]}, CCC: {stats_conc_val[ind]["CCC"]}, MAE: {stats_conc_val[ind]["MAE"]}, MSE: {stats_conc_val[ind]["MSE"]}')
                CCC_[ind, epoch] = stats_conc_val[ind]["CCC"]
                # f1[epoch, ind] = stats_conc_val[ind]["F1"]
                # f1[epoch, ind, :] = f1_per_video 
                f1_7[epoch, ind, :] = stats_per_video_val[ind]['F1_7']
                f1[epoch, ind, :] = stats_conc_val[ind]['F7']
            # np.save('CCC_0reg_1class_print.npy', CCC_)
            # np.save('./CM7_LR{}_batch{}/CM7_epoch{}_ars.npy'.format(args.lr, args.batch_size, epoch), stats_conc_val[0]["CM"])
            # np.save('./CM7_LR{}_batch{}/CM7_epoch{}_val.npy'.format(args.lr, args.batch_size, epoch), stats_conc_val[1]["CM"]) 

            # save correlation and CCC (per epoch) per video and for all videos concatenated
            corr_val_per_video_per_epoch_arousal.append(stats_per_video_val[0]['Corr'])
            ccc_val_per_video_per_epoch_arousal.append(stats_per_video_val[0]['CCC'])
            ccc_val_all_conc_arousal.append(stats_conc_val[0]['CCC'])
            corr_val_all_conc_arousal.append(stats_conc_val[0]['Corr'])
            
            # if we have 2 outputs, then save ccc and correlation for 2nd output as well
            if args.n_classes >=2:
                
                ccc_val_all_conc_valence.append(stats_conc_val[1]['CCC'])
                corr_val_all_conc_valence.append(stats_conc_val[1]['Corr'])

                current_per_video = (np.mean(stats_per_video_val[0]['CCC']) + np.mean(stats_per_video_val[1]['CCC'])) / 2
            
                # compute current performance as average of arousal and valence CCC (concatenated)
                current_perf = (np.mean(stats_conc_val[0]['CCC']) + np.mean(stats_conc_val[1]['CCC'])) / 2
                
            elif args.n_classes == 1:
                
                current_per_video = np.mean(stats_conc_val[0]['CCC'])
                
                # if we have 1 output, then current performance is the CCC computed over concatenated validation sets
                current_perf = np.mean(stats_conc_val[0]['CCC'])

            current_perf_per_epoch.append(current_perf)
            current_perf_video_per_epoch.append(current_per_video)
            
            # -- save checkpoint
            save_dict = {'epoch_idx': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
            ckpt_saver.save(save_dict, current_perf)
            
            if current_perf > best_perf:
                print('New best in main: ',current_perf)
                best_params = ppParams[:]
                best_perf = current_perf
                best_per_video = current_per_video
                best_epoch = epoch
                save_params_path = os.path.join(save_path, 'bestParams.npz')
                np.savez(save_params_path, best_params=best_params)

                print(best_params)


            scheduler.adjust_lr(optimizer, epoch)
            # --

            epoch += 1
            
        else:
            logger.info('Reached minimum learning rate, stopping training')
            break
    np.save('f1_7_concat_LR{}_bs{}.npy'.format(args.lr, args.batch_size), f1)
    np.save('f1_7_perVideo_LR{}_bs{}.npy'.format(args.lr, args.batch_size), f1_7)
    logger.info('===================================================')        
    logger.info('Reached maximum number of epochs, stopping training')
    logger.info(f'Best validation performance: {ckpt_saver.current_best}')    

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    args.dataset = 'final_load'
    _ = load_model(best_fp, model, args.dataset)
    
    print(f'Best params: {best_params}')
    print(f'Best perf Conc: {best_perf}')
    print(f'Best Perf per Video: {best_per_video}')
    print(f'Best epoch: {best_epoch}')
    
    print('-------Results on Validation Set-BestEpoch Model---------')
    phaseData = 'val'
    phaseEval = 'test'
    statsPerVideo_test, statsConc_test, _ , _ = test(model, dset_loaders[phaseData], criterion, logger, phaseEval, best_params)
    for ind, name in enumerate(args.class_names):
        logger.info('{} - )'.format(name ))
        logger.info('{} - Test per Video performance of best epoch: Corr: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, np.mean(statsPerVideo_test[ind]['Corr']) , np.mean(statsPerVideo_test[ind]['CCC']), np.mean(statsPerVideo_test[ind]['MAE']) , np.mean(statsPerVideo_test[ind]['MSE']) )  )
        logger.info('{} - Test all Videos Conc. performance of best epoch: Corr: {}, F1: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, statsConc_test[ind]['Corr'] , statsConc_test[ind]['F1'] , statsConc_test[ind]['CCC'], statsConc_test[ind]['MAE'] , statsConc_test[ind]['MSE'] )  )
    
    print('-------Results on Test Set - No post-processing - BestEpoch Model---------')
    phaseData = 'test'
    phaseEval = 'test'
    # since we use no post-processing we use bias = 0, scaling = 1, size of median filtering = 1
    tempParams = [ [1,0,1], [1,0,1] ]
    statsPerVideo_test, statsConc_test, _ , _ = test(model, dset_loaders[phaseData], criterion, logger, phaseEval, tempParams)
    for ind, name in enumerate(args.class_names):
        logger.info('{} - '.format(name))
        logger.info('{} - Test per Video performance of best epoch: Corr: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, np.mean(statsPerVideo_test[ind]['Corr']) , np.mean(statsPerVideo_test[ind]['CCC']), np.mean(statsPerVideo_test[ind]['MAE']) , np.mean(statsPerVideo_test[ind]['MSE']) )  )
        logger.info('{} - Test all Videos Conc. performance of best epoch: Corr: {}, F1: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, statsConc_test[ind]['Corr'] , statsConc_test[ind]['F1'] , statsConc_test[ind]['CCC'], statsConc_test[ind]['MAE'] , statsConc_test[ind]['MSE'] )  )

    print('-------Results on Test Set - BestEpoch Model---------')
    phase = 'test'
    statsPerVideo_test, statsConc_test, _ , _ = test(model, dset_loaders[phase], criterion, logger, phase, best_params)
    for ind, name in enumerate(args.class_names):
        logger.info('{} - '.format(name))
        logger.info('{} - Test per Video performance of best epoch: Corr: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, np.mean(statsPerVideo_test[ind]['Corr']) , np.mean(statsPerVideo_test[ind]['CCC']), np.mean(statsPerVideo_test[ind]['MAE']) , np.mean(statsPerVideo_test[ind]['MSE']) )  )
        logger.info('{} - Test all Videos Conc. performance of best epoch: Corr: {}, F1: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, statsConc_test[ind]['Corr'] , statsConc_test[ind]['F1'] , statsConc_test[ind]['CCC'], statsConc_test[ind]['MAE'] , statsConc_test[ind]['MSE'] )  )


if __name__ == '__main__':
    main()

