
import argparse
import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
import sklearn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from model.modelVideoClassReg import EmotionModelVideo
from utils.utils2 import load_model, get_ccc, get_logger, save_as_json, CheckpointSaver, get_lr, update_logger_batch, get_targets, get_folder_names
from dataloaders.dataloaders import get_data_loaders
from utils.optim_utils import get_optimizer, CosineScheduler
from loss.compute_loss_CCE import compute_loss_CCE

import datetime

# main() very brief summary:
# -- Contains               Parser, definitions for the dataset partitions for video and annotations;
# -- Importantly contains   get_model() - defines the architecture of the downstream model via model.modelVideoClassReg; 
#                           test() - evaluates the model on the validation set in between the training epochs and the test set after the training is over;
#                           train_one_epoch() - downstream training with backprop for one epoch, to be called for as many epochs as needed;
#                           main() - essentially:  (1) loads the checkpoint from the pretext pretraining - initialization for the downstream model via load_model(),
#                                                  (2) calls [train_one_epoch(training_set) & test(validation_set)] x args.epochs
#                                                  (3) calls test(test_set) for the best model parameters discovered on the validation set.
#                                                  ...and logs and prints out the results.
#  
# -- Final labels are       Arousal and Valence (the main task is regression). However there are +20 of auxiliary classification labels for each.
#  
# -- Several loss functions were implemented, as different loss functions work for different datasets. We favour 6 and 9 especcially.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Emotion AV Training')
    parser.add_argument('--dataset', type=str, default='sewa_video', choices = ['sewa_video', 'MSP_video', 'RECOLA'])
    parser.add_argument('--model-path', default='', help='loaded model path')
    parser.add_argument('--video-path', default='/fsx/marijajegorova/pre-process/step2_affine_transformation/sewa13crop_videos_npz_preprocessed_full/', help='video data path')
    parser.add_argument('--annot-path', default='/fsx/marijajegorova/pre-process/step2_affine_transformation/SEWA_annot_binned20/', help='annotation path')
    
    # IMPORTANT: ablation flags
    parser.add_argument('--fine-tuning', type=str, default='FT', choices = ['FT', 'Frozen3DConv', 'FullyFrozen'])
    parser.add_argument('--loss-type', default=6, type=int, help='1=MSE, 2=CCC, 3=Cross-Entropy, 4=BCE, 5=CCC+CE, 6=CCC+CE+MSE, 7=Arousal:CCC+CE, 8=Arousal:CCC+CE+MSE, 9=CCC+nCCE, 10=CCC+nCCE+MSE')
    parser.add_argument('--bins', default=20, type=int, help='number of bins for discretized valence and arousal')
    parser.add_argument('--augment', default=False, action='store_true', help='by default only horizonal flip is active, but more augmentations can be added in datasetSEWA')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--clip_length', default=0, type=float, help='length of video segment in seconds, randomly sampled from each training file. If 0 then the entire length is used.')
    parser.add_argument('--segments_per_file', default=3, type=int, help='Number of video segments to sample from training files')
    

    # -- ALL OF THE FOLLOWING FLAGS ARE OK AT DEFAULTS:
    parser.add_argument('--logging-dir', type = str, default = '.', help = 'path to the directory in which to save the log file')
    # model architecture definition
    parser.add_argument('--every-frame', default=True, action='store_true', help='RNN predicition based on every frame')
    parser.add_argument('--relu-type', type = str, default = 'relu', choices = ['relu','prelu','leaky'], help = 'what relu to use' )
    parser.add_argument('--hiddenDim', type = int, default = 256, help = "Number of hidden states on the GRU or filters in the TCN")
    
    # optimizer options
    parser.add_argument('--optimizer',type=str, default = 'adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--allow-size-mismatch', default=True, action='store_true', 
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr-scheduler-type',type=str, default = 'cosine', choices = ['cosine'])
    parser.add_argument('--min-lr', default=1e-6, type=float, help='minimal learning rate')

    # transform options
    # parser.add_argument('--fixed-length', default=True, action='store_true', help='Set to True to use fixed length sequences (no variable length augmentation')
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # other vars
    parser.add_argument('--batch-size', default=5, type=int, help='mini-batch size (default: 32)') 
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--gpu-idx', default = 0, type = int )
    parser.add_argument('--interval', default=10, type=int, help='display interval')
    parser.add_argument('--print-loss', default=False, action='store_true', 
                        help='If True, allows to print combined loss components at the args.interval')
    
    parser.add_argument('--annot_list', default=[0], action='store_true', help='List with annotators IDs, 0 means we operate on the average annotation')
    parser.add_argument('--annot_type', default = 'V', help = 'A, V or AV, applies to SEWA only')
    parser.add_argument('--output_type', default = 'Arousal_Valence', help = ' "Arousal", "Valence" or "Arousal_Valence" ')

    args = parser.parse_args()

    return args

args = parse_arguments()
bins = 20
args.strpoints_ar = []
args.strpoints_val = []

#  -- On AWS the typical folders per dataset are as follows:
# SEWA: 
# args.annot_path='/fsx/marijajegorova/pre-process/step2_affine_transformation/SEWA_annot_binned20/'
# args.video_path='/fsx/marijajegorova/pre-process/step2_affine_transformation/sewa13crop_videos_npz_preprocessed_full/'
# MSP:
# args.annot_path='/fsx/marijajegorova/AE1/AE_native/AE/MSP_annotations_200_trimmed_norm_bin20/'
# args.video_path='/fsx/marijajegorova/dino/MSP_widerCrop13_24window_attention_mp4_preprocessed/'
# RECOLA:
# args.annot_path='/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_annot_20secs_binned_new/'
# args.video_path='/fsx/marijajegorova/pre-process/step2_affine_transformation/RECOLA_split/'

video_tr, video_test, video_val = get_folder_names(args.video_path)
annot_tr, annot_test, annot_val = get_folder_names(args.annot_path)

args.fps = 50 # frames per second - is 25 for some videos, but that bears little to no significance to this model

# -- defining the data partitions for the dataloader:
args.annot_dir = {'train': annot_tr, 
                    'val' : annot_val,
                    'test' : annot_test}
args.video_dir = {'train': video_tr, 
        'val' : video_val,
        'test' : video_test}

if args.clip_length == 0:  
    args.clip_length_dict = {'train': 0, 'val' :0, 'test' : 0}
    args.segments_per_file_dict = {'train': 1, 'val' :1, 'test' : 1}
else:
    args.clip_length_dict = {'train': args.clip_length, 'val' :0, 'test' : 0}
    args.segments_per_file_dict = {'train': args.segments_per_file, 'val' :1, 'test' : 1}
   
# Usually only using 'Arousal_Valence'
if args.output_type == 'Arousal':
    args.class_names = ['Arousal']
elif args.output_type == 'Valence':
    args.class_names = ['Valence']
elif args.output_type == 'Arousal_Valence':
    args.class_names = ['Arousal', 'Valence']
else:
    raise Exception('Wrong output Name')

# the n_classes defines the number of the model outputs - 2 for arousal and valence actual values and 20x2 for their discretized versions:
args.n_classes = 2*(bins+1)
args.weight_decay = 1e-4 if args.optimizer != 'adamw' else 1e-2

print('Dataset: ', args.dataset)
print('Optimizer: ', args.optimizer)
print('Batch Size: ', args.batch_size)
print('Annot Type: ', args.annot_type)
print('LR: ', args.lr)
print('Loss Type: ', args.loss_type)
print('Output Type: ', args.output_type)
print('Clip Length: ', args.clip_length)
print('SegmentsPerFile: ', args.segments_per_file)        

args.preprocessing_func = []

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache() 

# pick device (which GPU or CPU)
#device = torch.device("cuda:{}".format(args.gpu_idx) if ( torch.cuda.is_available() and args.gpu_idx >= 0 ) else "cpu")
device = torch.device('cuda')

def get_save_folder( ): # makes the log folder. Comment out if not needed.
    folder_name = 'training'
    append_str = '_every_frame' if args.every_frame else '_last_frame'
    folder_name += append_str
    save_path = os.path.join(args.logging_dir, folder_name, datetime.datetime.now().isoformat().split('.')[0])
    os.makedirs(save_path, exist_ok=True)
    
    return save_path

def get_model(args, device): # defines the model architecture
    model = EmotionModelVideo(fine_tuning=args.fine_tuning, bins=args.bins, n_classes=args.n_classes, device=device).to(device)
    return model

def test(model, dset_loader, logger, phase):

    model.eval()
    running_loss = 0.
    running_all = 0.

    stats_conc = [ {} for _ in range(args.n_classes)]
    
    with torch.no_grad():
        # errors = 0
        targets_all_videos_list = []
        pred_all_videos_list = []
        for batch_idx, data in enumerate(dset_loader):
            # we first load all targets and compute predictions for all videos
            inputs = data['video']
            
            targets = get_targets(data, args.output_type)
            lengths = tuple([inputs.size(1)] * len(targets))
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, lengths)
            
            # we need this because sometimes targets length (2nd element) might be longer by 1 than the outputs length
            if outputs.size(1) > targets.size(1):
                outputs = outputs[:,:targets.size(1),:]
            elif targets.size(1) > outputs.size(1):
                targets = targets[:,:outputs.size(1),:]
            
            outputs = outputs.double()
            
            loss, args.strpoints_ar, args.strpoints_val = compute_loss_CCE(outputs, targets, args.loss_type, args.strpoints_ar, args.strpoints_val, model.parameters(), args.bins, batch_idx, args.interval, args.print_loss) 
                
            out = outputs.cpu().numpy()
            trg = targets.cpu().numpy()
            trg = np.concatenate((trg[:, :, 0, :], trg[:, :, 1, :]), 2)
            
            if pred_all_videos_list == []:
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
                
    # -- We compute the performance measures per video and after concatenating all predictions/targets (a common way of reporting CCC in CV)

    for ind in range(2): # 2 for valence and arousal

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
        
        all_preds_conc = (all_preds_conc > 0.5) 
        
        stats_conc[ind]['CCC'] = get_ccc(all_preds_conc_reg, all_trg_conc_reg)
        all_preds_conc_f1 = all_preds_conc[[0 and np.mod(np.arange(all_preds_conc.size),bins)!=0]]
        all_trg_conc_f1 = all_trg_conc[[0 and np.mod(np.arange(all_trg_conc.size),bins)!=0]]
        try:
            f1_per_video_ = np.zeros(bins)
            for k in range(bins):
                f1_per_video_[k] = sklearn.metrics.f1_score((all_preds_conc_f1[:][k]> 0.5), all_trg_conc_f1[:][k], average='weighted')
            stats_conc[ind]['F1'] = np.mean(f1_per_video_)
        except:
            stats_conc[ind]['F1'] = sklearn.metrics.f1_score((np.hstack(all_preds_conc_f1)> 0.5), all_trg_conc_f1, average='weighted')

        stats_conc[ind]['CM'] = sklearn.metrics.confusion_matrix(single_target_list_,single_pred_list_)
        stats_conc[ind]['Corr'] = pearsonr(all_preds_conc[::bins], all_trg_conc[::bins])[0]
        stats_conc[ind]['MSE'] = mean_squared_error(all_preds_conc[::bins], all_trg_conc[::bins])
        stats_conc[ind]['MAE'] = mean_absolute_error(all_preds_conc[::bins], all_trg_conc[::bins])

    return  stats_conc


def train_one_epoch(model, dset_loaders: dict, epoch: int, phase: str, optimizer, logger):
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
    
    for batch_idx, data in enumerate(dset_loaders[phase]):

        inputs = data['video']
        targets = get_targets(data, args.output_type)
        # print('target: ', np.shape(targets))
        lengths = tuple([inputs.size(1)] * len(targets))
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)

        outputs = outputs.double()

        loss, args.strpoints_ar, args.strpoints_val = compute_loss_CCE(outputs, targets, args.loss_type, args.strpoints_ar, args.strpoints_val, model.parameters(), args.bins, batch_idx, args.interval, args.print_loss)
    
        with torch.no_grad():
            out = outputs.cpu().numpy()
            trg = targets.cpu().numpy()
            trg = np.concatenate((trg[:, :, 0, :], trg[:, :, 1, :]), 2)

        for ind in range(2):
                    
            if args.n_classes == 1:
                single_out = out[:,:,0]
                single_targ = trg
            elif args.n_classes == 2:
                single_out = out[:,:,ind]
                single_targ = trg[:,:,ind]
            elif args.n_classes >2 :
                
                single_out = out[:,:,(ind*(bins+1)):(ind*(bins+1)+(bins+1))]
                single_targ = trg[:,:,(ind*(bins+1)):(ind*(bins+1)+(bins+1))]
                single_targ_ = np.argmax(single_targ, axis=2)
                single_out_ = np.argmax(single_out, axis=2)
                     
            ccc_per_seq = [get_ccc(single_out[i,:],single_targ[i,:]) for i in range(len(single_out))]
            f1_per_seq = [sklearn.metrics.f1_score(single_out_[i,:],single_targ_[i,:], average='weighted') for i in range(len(single_out_))]
            
            sum_f1_per_seq[ind] = sum_f1_per_seq[ind] + np.sum(f1_per_seq)
            sum_ccc_per_seq[ind] = sum_ccc_per_seq[ind] + np.sum(ccc_per_seq)
 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*inputs.size(0)
        running_all += inputs.size(0)

        #TODO: we print stats for arousal ONLY
        # -- log running stats every x batches
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            update_logger_batch( args, logger, dset_loaders[phase], batch_idx, running_loss, sum_ccc_per_seq[0], sum_f1_per_seq[0], running_all)
            
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

    if args.lr_scheduler_type == 'cosine':
        scheduler = CosineScheduler( args.lr, args.epochs )
    else:
        raise Exception('Scheduler option not implemented!!')
    
    ccc_val_all_conc_arousal = []
    corr_val_all_conc_arousal = []
    ccc_val_all_conc_valence = []
    corr_val_all_conc_valence = []
    
    current_perf_per_epoch = []
    best_epoch = 0
    
    # save best performance, initialise at -Inf
    best_perf = - math.inf
    
    #load model if path is given - that is where the pretext model checkpoint gets uploaded, in our case pretrained on LiRA.
    if args.model_path != '':
        model = load_model(args.model_path, model, args.dataset, allow_size_mismatch = args.allow_size_mismatch )
        print('Model was succesfully loaded')
        
    epoch = 0

    while epoch < args.epochs:
        if ( get_lr(optimizer) > args.min_lr ):
            # -- train for 1 epoch and save checkpoint
            phase = 'train'
            model = train_one_epoch(model, dset_loaders, epoch, phase, optimizer, logger)
                    
            # -- validate
            phase = 'val'
            stats_conc_val = test(model, dset_loaders[phase], logger, phase)

            for ind, name in enumerate(args.class_names):
                logger.info(f'{name} - Epoch:\t{epoch} , LR: {get_lr(optimizer)}')
                logger.info(f'{name} - All validation videos concatenated: Corr: {stats_conc_val[ind]["Corr"]}, F1: {stats_conc_val[ind]["F1"]}, CCC: {stats_conc_val[ind]["CCC"]}, MAE: {stats_conc_val[ind]["MAE"]}, MSE: {stats_conc_val[ind]["MSE"]}')

            ccc_val_all_conc_arousal.append(stats_conc_val[0]['CCC'])
            corr_val_all_conc_arousal.append(stats_conc_val[0]['Corr'])
            
            # if we have 2 outputs (i.e. Arousal and Valence), then save ccc and correlation for 2nd output as well
            if args.n_classes >=2: #it is >=2 because there might be the auxiliary predictions for the discretised losses
                ccc_val_all_conc_valence.append(stats_conc_val[1]['CCC'])
                corr_val_all_conc_valence.append(stats_conc_val[1]['Corr'])
                current_perf = (np.mean(stats_conc_val[0]['CCC']) + np.mean(stats_conc_val[1]['CCC'])) / 2
                
            elif args.n_classes == 1:
                # if we have 1 output (just Arousal or Valence alone), then current performance is the CCC computed over concatenated validation sets
                current_perf = np.mean(stats_conc_val[0]['CCC'])

            current_perf_per_epoch.append(current_perf)
            # current_perf_video_per_epoch.append(current_per_video)
            
            # -- save checkpoint
            save_dict = {'epoch_idx': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
            ckpt_saver.save(save_dict, current_perf)
            
            if current_perf > best_perf:
                print('New best in main: ',current_perf)
                best_perf = current_perf
                best_epoch = epoch

            scheduler.adjust_lr(optimizer, epoch)
            # --
            epoch += 1
            
        else:
            logger.info('Reached minimum learning rate, stopping training')
            break

    logger.info('===================================================')        
    logger.info('Reached maximum number of epochs, stopping training')
    logger.info(f'Best validation performance: {ckpt_saver.current_best}')    

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    args.dataset = 'final_load'
    _ = load_model(best_fp, model, args.dataset)
    
    print(f'Best perf Conc: {best_perf}')
    # print(f'Best Perf per Video: {best_per_video}')
    print(f'Best epoch: {best_epoch}')
    
    print('-------Results on Validation Set-BestEpoch Model---------')
    phaseData = 'val'
    phaseEval = 'test'
    statsConc_test  = test(model, dset_loaders[phaseData], logger, phaseEval)
    for ind, name in enumerate(args.class_names):
        logger.info('{} - )'.format(name ))
        logger.info('{} - Test all Videos Conc. performance of best epoch: Corr: {}, F1: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, statsConc_test[ind]['Corr'] , statsConc_test[ind]['F1'] , statsConc_test[ind]['CCC'], statsConc_test[ind]['MAE'] , statsConc_test[ind]['MSE'] )  )
    
    print('-------Results on Test Set - No post-processing - BestEpoch Model---------')
    phaseData = 'test'
    phaseEval = 'test'

    statsConc_test = test(model, dset_loaders[phaseData], logger, phaseEval)
    for ind, name in enumerate(args.class_names):
        logger.info('{} - '.format(name))
        logger.info('{} - Test all Videos Conc. performance of best epoch: Corr: {}, F1: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, statsConc_test[ind]['Corr'] , statsConc_test[ind]['F1'] , statsConc_test[ind]['CCC'], statsConc_test[ind]['MAE'] , statsConc_test[ind]['MSE'] )  )

    print('-------Results on Test Set - BestEpoch Model---------')
    phase = 'test'
    statsConc_test  = test(model, dset_loaders[phase], logger, phase)
    for ind, name in enumerate(args.class_names):
        logger.info('{} - '.format(name))
        logger.info('{} - Test all Videos Conc. performance of best epoch: Corr: {}, F1: {}, CCC: {}, MAE: {}, MSE: {}'.format(name, statsConc_test[ind]['Corr'] , statsConc_test[ind]['F1'] , statsConc_test[ind]['CCC'], statsConc_test[ind]['MAE'] , statsConc_test[ind]['MSE'] )  )

if __name__ == '__main__':
    main()

