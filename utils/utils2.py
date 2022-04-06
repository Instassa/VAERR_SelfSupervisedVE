from __future__ import annotations

import os

import numpy as np
import torch
import logging
import json
import math
import shutil
from scipy.signal import medfilt
import numpy as np


def sec2frames(sec, fps):
    
    frames = round(sec * fps)
    return frames

def get_ccc(a: float, b: float):
    ccc = (
        2
        * np.cov(a, b)[0][1]
        / (np.var(a) + np.var(b) + np.square(np.mean(a) - np.mean(b)))
    )
    return ccc


def convert_str_list_to_float(list_of_str_lines: list[str]) -> np.ndarray:

    float_list = []
    for line in list_of_str_lines:
        elem_list = line.split(",")
        elem_list_float = [float(x) for x in elem_list]
        del elem_list_float[0]  # first element is timestamp
        float_list.append(elem_list_float)

    float_list_np = np.array(float_list)
    float_list_np = np.squeeze(float_list_np)

    return float_list_np


def read_lines_from_file(filepath: str) -> list[str]:
    # print('filepath:', filepath)
    with open(filepath) as fp:
        list_of_lines = fp.readlines()
    list_of_lines = [x.strip() for x in list_of_lines]
    return list_of_lines


def load_model(load_path, model, dataset_type, optimizer=None, allow_size_mismatch=False):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile(
        load_path
    ), f"Error when loading the model, provided path not found: {load_path}"
    checkpoint = torch.load(load_path, map_location=model.device)
    # for k, v in checkpoint.items():
    #     print(k)
    if dataset_type =='sewa_video' or dataset_type =='MSP_video' or dataset_type =='RECOLA':
        try:
            loaded_state_dict = checkpoint["netG"]
        except:
            try:
                loaded_state_dict = checkpoint["state_dict"]
            except:
                loaded_state_dict = checkpoint["student"] #DINO
    elif dataset_type =='MSP_video_140':
        loaded_state_dict = checkpoint["state_dict"]
    else:
        loaded_state_dict = checkpoint["model_state_dict"]
 

    if allow_size_mismatch:
        # print(loaded_state_dict[i])

        model_state_dict = model.state_dict()
        loaded_state_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if (k in model_state_dict)
            and (model_state_dict[k].shape == loaded_state_dict[k].shape)
        }

    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, checkpoint["epoch_idx"], checkpoint
    return model

def get_folder_names(master_folder):
    name_list = os.listdir(master_folder)
    # print(name_list)
    for name in name_list:
        if 'tr' in name or 'Tr' in name:
            train_folder = str(master_folder +'/' +name)
        elif 'test' in name or 'Test' in name:
            test_folder = str(master_folder +'/' +name)
        elif 'val' in name or 'Val' in name:
            val_folder = str(master_folder +'/' +name)
    try:
        val_folder
    except:
        val_folder = test_folder #RECOLA has no test set

    return train_folder, test_folder, val_folder

def get_logger(save_path: str):
    
    app_str = '_log.txt'

    log_path = os.path.join(save_path, app_str)
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger

def update_logger_batch( args, logger, dset_loader, batch_idx: int, running_loss: float, sum_ccc_per_seq: float, sum_f1_per_seq: float, running_all: float):
    perc_epoch = 100. * batch_idx / (len(dset_loader)-1)
    logger.info(f'[{running_all}/{len(dset_loader.dataset)} ({perc_epoch}%)] \t Loss: {running_loss / running_all}\t CCC{sum_ccc_per_seq / running_all}\t F1:{sum_f1_per_seq / running_all}')


def save_as_json(d: dict, filepath: str):
    with open(filepath, 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)
        
        
class CheckpointSaver:
    def __init__(self, save_dir: str, checkpoint_fn: str = 'ckpt.pth.tar', best_fn: str = 'ckpt.best.pth.tar'):
        
        self.save_dir = save_dir

        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
    
        # init var to keep track of best performing checkpoint
        self.current_best = -math.inf

        
    def save(self, save_dict, current_perf, epoch=-1):
        """
            Save checkpoint and keeps copy if current perf is best overall 
        """

        # keep track of best model
        self.is_best = current_perf > self.current_best
        
        if self.is_best:
            print('New best in saving function: ', current_perf)
            self.current_best = current_perf
            best_fp = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_prec'] = self.current_best

        # save
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)
        torch.save(save_dict, checkpoint_fp)
        print("Checkpoint saved at {}".format(checkpoint_fp))
        if self.is_best:
            shutil.copyfile(checkpoint_fp, best_fp)
        


    def set_best_from_ckpt(self, ckpt_dict):
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage',None)   
        
        
        
def get_start_end_ind(clip_length_in_sec, fps, len_in_frames):
    
    clip_length_in_frames = sec2frames(clip_length_in_sec, fps)
    last_ind_allowed = len_in_frames - clip_length_in_frames # ideally we should add 1, however we don't add it because audio samples sometimes are fewer than what they are supposed to be, e.g., we have 50=1sec frames but audio duration is sliglty less than 16000 
            
    if last_ind_allowed < 0:
        raise Exception('Clip lenght is shorter than what it should be')
                 
    start_ind = 0    
    try:
        start_ind = np.random.randint(low=0, high=last_ind_allowed, dtype=int)
    except:
        print()
        # print(clip_length_in_frames)
        # print(last_ind_allowed)    
    end_ind = start_ind + clip_length_in_frames
    
    return start_ind, end_ind

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']      

def get_targets(data, output_type: str):
        
    if output_type == 'Arousal':
        targets = data['arousal_annot']
    elif output_type == 'Valence':
        targets = data['valence_annot']
    elif output_type == 'Arousal_Valence':
        t1 = data['arousal_annot']
        # print('T1:', np.shape(t1))
        t2 = data['valence_annot']
        targets = torch.stack([t1, t2], dim = 2)
    # print('TARGETS: ', np.shape(targets))
    return targets

# def get_median_filt_win(pred, trg):
#     win_list = np.arange(25,100,12) # sewa
    
#     best_ccc = get_ccc(pred, trg)
#     best_win = 1
    
#     for w in win_list:
#         filtered_pred = medfilt(pred, kernel_size = w)
#         ccc_filt = get_ccc(filtered_pred, trg)
#        #temp.append(ccc_filt)
#         if ccc_filt > best_ccc:
#             best_win = w
            
#     return best_win

# def compute_bias(pred,trg):
    
#     mean_pred = np.mean(pred)
#     mean_targ = np.mean(trg)
    
#     bias = mean_targ - mean_pred
    
#     ccc_no_bias = get_ccc(pred, trg)
#     ccc_with_bias = get_ccc(pred + bias, trg)

#     if ccc_with_bias > ccc_no_bias:
#         return bias
#     else:
#         return 0
    
# def compute_scale(pred,trg):
    
#     st_dev_pred = np.std(pred)
#     st_dev_targ = np.std(trg)
    
#     scale = st_dev_targ / st_dev_pred
    
#     ccc_no_scale = get_ccc(pred, trg)
#     ccc_with_scale = get_ccc(pred * scale, trg)

#     if ccc_with_scale > ccc_no_scale:
#         return scale
#     else:
#         return 1  

