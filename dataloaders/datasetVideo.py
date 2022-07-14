from __future__ import annotations

import os
import sys
import numpy as np
import torch
    
from utils.utils2 import get_start_end_ind, read_lines_from_file
from statistics import mean
from dataloaders.transforms import (
    Scale,
    RandomCrop,
    CenterCrop,
    HorizontalFlip,
    Compose,
    Normalize,
    Solarization,
    SaltAndPepper,
    TimeOut,
    Cutout,
)

class datasetVideo(object):
    
    def __init__( self, data_part_str: str, data_dir: str, cannot_dir: str, annot_type: str, output_type: str, annot_list: list[int], clip_length_in_sec: int, segments_per_file: int, fps: int, augs: bool, every_frame: bool):
        self.bins = 6
        self._data_partition_str = data_part_str
        self._data_dir = data_dir
        # self._annot_dir = annot_dir
        self._cannot_dir = cannot_dir
        self._annot_type = annot_type
        self._output_type = output_type
        self._annotators_list = annot_list
        self.augs = augs
        self.every_frame = every_frame

        # self._arousal_files = []
        self._categorical_files = []
        # self._valence_files = []
        self._video_files = []
        self._subj_id_list = []
        # self._instance_id_list = []
        
        self._clip_length_in_sec = clip_length_in_sec
        self._fps = fps
        self._segments_per_file = segments_per_file
        
        print("Video dir: %s" % (self._data_dir))

        self.load_paths()

    def load_paths(self):

        new_read_data_dir = self._data_dir 
        # new_read_annot_dir = self._annot_dir 
        new_read_cannot_dir = self._cannot_dir 
        
        video_dir_list = os.listdir(new_read_data_dir)
        # annot_dir_list = os.listdir(new_read_annot_dir)
        
        # if len(video_dir_list) != len(annot_dir_list):
        #     raise Exception('Number of audio files is different than number of annotation files')
 
        for video_folder in video_dir_list:
            new_read_data_dir2 = os.path.join(new_read_data_dir, video_folder)
            # if 'RECOLA' in self._data_dir:
            #     new_read_annot_dir2_ar = os.path.join(new_read_annot_dir, video_folder[:3])# + '_arousal')
            #     new_read_annot_dir2_val = os.path.join(new_read_annot_dir, video_folder[:3])# + '_valence')
            # else:
            #     new_read_annot_dir2_ar = os.path.join(new_read_annot_dir, video_folder)
            #     new_read_annot_dir2_val = os.path.join(new_read_annot_dir, video_folder)
            
            if self._output_type == "Categorical":
                # print()
                new_read_annot_dir2_cat = os.path.join(new_read_cannot_dir, video_folder)
            
            video_files_list = [x for x in os.listdir(new_read_data_dir2) if x.endswith('npz')]
            
            video_files_list_sorted = self._sort_files(video_files_list)
            
            # repeat file names as many times as the number of random clips we wish to extract per file
            video_files_list_sorted = video_files_list_sorted * self._segments_per_file

            for file in video_files_list_sorted:
                if 'RECOLA' in self._data_dir:
                    filename = file.split('.')[0]
                    token_list = filename.split('_')
                    subj_id =token_list[0][1:]
                    # print(token_list)
                    token_list_arousal = [token_list[0]] + ['arousal'] + [token_list[-1]]
                    token_list_valence = [token_list[0]] + ['valence'] + [token_list[-1]]
                    arousal_filename = str(['_'.join(token_list_arousal[:])][0]) + '.csv' 
                    valence_filename = str(['_'.join(token_list_valence[:])][0]) + '.csv' 
                else:
                    filename = file.split('.')[0]
                    token_list = filename.split('_')
                    subj_id = token_list[2]
                    # instance_id = token_list[-1]
                    
                    token_list_arousal = token_list + ['Arousal', self._annot_type, 'Aligned'] #+ [instance_id]
                    token_list_valence = token_list + ['Valence', self._annot_type, 'Aligned'] #+ [instance_id]
                    if self._output_type == "Categorical":
                        token_list_categorical = token_list + ['Categorical', self._annot_type, 'Aligned'] #+ [instance_id]
                    # print('ANNOT: ', self._data_dir)
                    if 'sewa' in self._data_dir:
                        # print('SEWA ANN 0')
                        arousal_filename = str(['_'.join(token_list_arousal[:])][0]) #+ '_0.csv' 
                        valence_filename = str(['_'.join(token_list_valence[:])][0]) #+ '_0.csv' 
                    else:
                        arousal_filename = str(['_'.join(token_list_arousal[:])][0]) + '_0.csv' 
                        valence_filename = str(['_'.join(token_list_valence[:])][0]) + '_0.csv' 
                        if self._output_type == "Categorical":
                            categorical_filename = str(['_'.join(token_list_categorical[:])][0]) + '_0.csv'
                
                video_file_path = os.path.join(new_read_data_dir2, file)
                # arousal_file_path = os.path.join(new_read_annot_dir2_ar, arousal_filename)
                
                # valence_file_path = os.path.join(new_read_annot_dir2_val, valence_filename)
                if self._output_type == "Categorical":
                    categorical_file_path = os.path.join(new_read_annot_dir2_cat, categorical_filename)
                    self._categorical_files.append(categorical_file_path)
                
                self._video_files.append(video_file_path)
                # self._arousal_files.append(arousal_file_path)
                # self._valence_files.append(valence_file_path)
                
                self._subj_id_list.append(subj_id)
                # self._instance_id_list.append(instance_id)
            
                        
        print('Partition {} loaded'.format(self._data_partition_str))
    
    def load_data_video(self, filename: str):
        try:
            video = np.load(filename)['arr_0']
            # -- If normalization was not performed at the pre-processing stage:
            # if 'sewa' in self._data_dir:
            #     (mean, std) = (0.421, 0.165) #SEWA
            # elif 'RECOLA' in self._data_dir:
            #     (mean, std) = (0.27079829566888647, 0.1581663185206455) #RECOLA
            # elif 'MSP' in self._data_dir:
            #     (mean, std) = (0.493, 0.207) #MSP
            # else:
            #     'WARNING: The dataset is neither SEWA, nor MSP Face, nor RECOLA. No normalization parameters are stored.'
            # crop_size = (80, 80)
            # # print('MAX: ', np.max(video))
            # if np.max(video) > 1:
            #     mean255 = 0.0
            #     std255 = 255.0
            # else:
            #     mean255 = 0.0
            #     std255 = 1.0
            if 'train' in self._data_partition_str and self.augs:
                self.video_transforms_cv = Compose([
                                        # Normalize( mean255, std255),
                                        # RandomCrop(crop_size),
                                        # Cutout(5, 21),#Cutout(5, 28), SEWA
                                        HorizontalFlip(0.5),
                                        # Normalize(mean, std), 
                                        # TimeOut(0.2),
                                        # SaltAndPepper(0.05)
                                        # Solarization(0.2, 0.9), #
                                        ])
                video = self.video_transforms_cv(video)

            video = torch.from_numpy(video).float()

            return video 

        except IOError:
            print( f"Error when reading file: {filename}" )
            sys.exit()

    def load_annotations_orig(self, annot_path: str):
        try:
            list_of_lines_arousal = read_lines_from_file(annot_path)
                        
        except IOError:
            print( f"Error when reading file: {annot_path}")
            # sys.exit()
            
        #del listOfLinesArousal[0] # delete headers if using original files
        annot_list = []
        time_list = []    
            
        for l in list_of_lines_arousal:
            elem_list = l.split(',')
            elem_list_float = [float(x) for x in elem_list]
            
            time_ind = elem_list_float[0]
            time_list.append(time_ind)
            del elem_list_float[0] # first element is timestamp
            
            selected_annot = [elem_list_float[x] for x in self._annotators_list]
            mean_annot = mean(selected_annot)
            annot_list.append(mean_annot)
            
        annot_list = np.array(annot_list)
        time_list = np.array(time_list)
        
                
        return annot_list, time_list  
            
    def load_annotations(self, annot_path: str):
        # print(annot_path)

        try:
            list_of_lines = read_lines_from_file(annot_path)
                        
        except IOError:
            print( f"Error when reading file: {annot_path}")
        
        if 'cate' in annot_path:
            annot_list = np.zeros((len(list_of_lines), 6))
            time_list = []    
            indx = 0    
            for l in list_of_lines:
                elem_list = l.split(',')
                elem_list_float = [int(x) for x in elem_list]
                # print(np.shape(elem_list_float))
                # elem_list_float.append(float(elem_list[-1]))
                
                time_ind = elem_list_float[0]
                time_list.append(time_ind)
                del elem_list_float[0] # first element is timestamp
                # print(np.shape(elem_list_float))

                selected_annot = elem_list_float
                annot_list[indx, :] = selected_annot
                indx +=1
        else:
            annot_list = np.zeros((len(list_of_lines), self.bins+1))
            time_list = []    
            indx = 0    
            for l in list_of_lines:
                elem_list = l.split(',')
                elem_list_float = [int(x) for x in elem_list[:-1]]
                elem_list_float.append(float(elem_list[-1]))
                
                time_ind = elem_list_float[0]
                time_list.append(time_ind)
                del elem_list_float[0] # first element is timestamp
                
                selected_annot = elem_list_float
                annot_list[indx, :] = selected_annot
                indx +=1
            
        # annot_list = np.array(annot_list).reshape((len(list_of_lines_arousal), 7))
        # print('ANN. LIST: ', np.shape(annot_list))
        # time_list = np.array(time_list)
        # print('EVERY FRAME:', self.every_frame)
        if self.every_frame == False:
            return annot_list[-1, :], time_list
        else:
            return annot_list, time_list                   
    
    def _sort_files(self, string_list: list[str]):
        try:
            instance_list = [int(s.split('.')[0].split('_')[-1]) for s in string_list] # get instances as an int list
        except:
            instance_list = [int(s.split('.')[0].split('_')[1]) for s in string_list] # get instances as an int list
        sorted_ind = np.argsort(instance_list)
        sorted_list = [string_list[i] for i in sorted_ind]
        return sorted_list
    
    def __getitem__(self, idx: int):

        if self._output_type == "Categorical":
            categorical_path = self._categorical_files[idx]
        video_path = self._video_files[idx]
        subj_id = self._subj_id_list[idx]
        
        if self._output_type == "Categorical":
            categorical_annot, timestamps_categorical = self.load_annotations(categorical_path)
            
        raw_video = self.load_data_video(video_path)

        if self._clip_length_in_sec != 0:
                
            len_in_frames = len(categorical_annot)
            start_ind, end_ind = get_start_end_ind(self._clip_length_in_sec, self._fps, len_in_frames)
            
            if self._output_type == "Categorical":
                categorical_annot = categorical_annot[start_ind:end_ind]
                timestamps_categorical = timestamps_categorical[start_ind:end_ind]

            raw_video = raw_video[start_ind:end_ind]

        if self._output_type == "Categorical":
            sample = {'video': raw_video, 'categorical_annot': categorical_annot, 'subj_id': subj_id, 'paths': [video_path], 'timestamps': timestamps_categorical}

        return sample                                

    def __len__(self):
        return len(self._video_files)
    