from __future__ import annotations

import os
import sys
import numpy as np
import torch
    
from utils2 import get_start_end_ind, read_lines_from_file
from statistics import mean
from transforms import (
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

class datasetSEWAvideoClassReg(object):
    
    def __init__( self, data_part_str: str, data_dir: str, annot_dir: str, annot_type: str, annot_list: list[int], clip_length_in_sec: int, segments_per_file: int, fps: int):

        self.bins = 20
        self._data_partition_str = data_part_str
        self._data_dir = data_dir
        self._annot_dir = annot_dir
        self._annot_type = annot_type
        self._annotators_list = annot_list

        self._arousal_files = []
        self._valence_files = []
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
        new_read_annot_dir = self._annot_dir 
        
        video_dir_list = os.listdir(new_read_data_dir)
        annot_dir_list = os.listdir(new_read_annot_dir)
        
        # if len(video_dir_list) != len(annot_dir_list):
        #     raise Exception('Number of audio files is different than number of annotation files')
 
        for video_folder in video_dir_list:
            new_read_data_dir2 = os.path.join(new_read_data_dir, video_folder)
            if 'RECOLA' in self._data_dir:
                new_read_annot_dir2_ar = os.path.join(new_read_annot_dir, video_folder[:3])# + '_arousal')
                new_read_annot_dir2_val = os.path.join(new_read_annot_dir, video_folder[:3])# + '_valence')
            else:
                new_read_annot_dir2_ar = os.path.join(new_read_annot_dir, video_folder)
                new_read_annot_dir2_val = os.path.join(new_read_annot_dir, video_folder)
            
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
                    # print('ANNOT: ', self._data_dir)
                    if 'sewa' in self._data_dir:
                        # print('SEWA ANN 0')
                        arousal_filename = str(['_'.join(token_list_arousal[:])][0]) #+ '_0.csv' 
                        valence_filename = str(['_'.join(token_list_valence[:])][0]) #+ '_0.csv' 
                    else:
                        arousal_filename = str(['_'.join(token_list_arousal[:])][0]) + '_0.csv' 
                        valence_filename = str(['_'.join(token_list_valence[:])][0]) + '_0.csv' 
                
                video_file_path = os.path.join(new_read_data_dir2, file)
                arousal_file_path = os.path.join(new_read_annot_dir2_ar, arousal_filename)
                
                valence_file_path = os.path.join(new_read_annot_dir2_val, valence_filename)
                
                self._video_files.append(video_file_path)
                self._arousal_files.append(arousal_file_path)
                self._valence_files.append(valence_file_path)
                self._subj_id_list.append(subj_id)
                # self._instance_id_list.append(instance_id)
            
                        
        print('Partition {} loaded'.format(self._data_partition_str))
    
    def load_data_video(self, filename: str):
        try:
            video = np.load(filename)['arr_0']
            # (mean, std) = (0.493, 0.207) #MSP
            (mean, std) = (0.421, 0.165) #SEWA I think
            # (mean, std) = (0.27079829566888647, 0.1581663185206455) #RECOLA
            crop_size = (80, 80)
            # print('MAX: ', np.max(video))
            if np.max(video) > 1:
                mean255 = 0.0
                std255 = 255.0
            else:
                mean255 = 0.0
                std255 = 1.0
            if 'train' in self._data_partition_str:
                self.video_transforms_cv = Compose([
                                        # Normalize( mean255, std255),
                                         
                                        # RandomCrop(crop_size),
                                        # Cutout(5, 21),#Cutout(5, 28), SEWA
                                        # HorizontalFlip(0.5),
                                        
                                        # Normalize(mean, std), 
                                        # TimeOut(0.2),
                                        # SaltAndPepper(0.05)
                                        # Solarization(0.2, 0.9), #
                                        ])
                # print('TRAIN AUGMENTATIONS')
            else:
                self.video_transforms_cv = Compose([
                                        # Normalize( mean255,std255 ),
                                        # Normalize(mean, std) 
                                        ])      
                # print('TEST AUGMENTATIONS')          

            video = self.video_transforms_cv(video)
            # print('SIZE: ',(np.shape(video)))
            
            # video = torch.from_numpy(video).float()
            video = torch.from_numpy(video).float()
            # sr = np.load(filename)['sr']

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
            list_of_lines_arousal = read_lines_from_file(annot_path)
                        
        except IOError:
            print( f"Error when reading file: {annot_path}")
            # sys.exit()
            
        #del listOfLinesArousal[0] # delete headers if using original files
        # print('L:', len(list_of_lines_arousal))
        annot_list = np.zeros((len(list_of_lines_arousal), self.bins+1))
        time_list = []    
        indx = 0    
        for l in list_of_lines_arousal:
            elem_list = l.split(',')
            # print(elem_list, 'ELEM_LIST')
            # elem_list_float = [int(x) for x in elem_list]
            elem_list_float = [int(x) for x in elem_list[:-1]]
            elem_list_float.append(float(elem_list[-1]))

            # print(elem_list_float, 'ELEM_arr')
            
            time_ind = elem_list_float[0]
            time_list.append(time_ind)
            del elem_list_float[0] # first element is timestamp
            
            selected_annot = elem_list_float#[elem_list_float[x] for x in self._annotators_list]
            # print('ANN. SELECTED: ', selected_annot)
            # print('l: ', indx)
            # mean_annot = mean(selected_annot)
            # print('ANN. MEAN: ', mean_annot)
            annot_list[indx, :] = selected_annot
            indx +=1
            
        # annot_list = np.array(annot_list).reshape((len(list_of_lines_arousal), 7))
        # print('ANN. LIST: ', np.shape(annot_list))
        time_list = np.array(time_list)
        
                
        return annot_list, time_list      

    def load_annotations_sewa(self, annot_path: str):

        # print(annot_path)
        # annot_path = annot_path + '_0.csv' 
        # files = [x for x in os.listdir(annot_path[:-58]) if annot_path[-57:] in x]
        files = [x for x in os.listdir(annot_path[:-53]) if annot_path[-52:] in x]
        # print(files, len(files))
 
        
        indx = 0
        for annot_it in files: 
            time_list = []
            # print(annot_path[-52:] + '/' + annot_it)
            try:
                # print(annot_path[-52:]+ annot_it)
                list_of_lines_arousal = read_lines_from_file(annot_path[:-52] + annot_it)
                            
            except IOError:
                print( f"Error when reading file: {annot_it}")
                # sys.exit()
                
            #del listOfLinesArousal[0] # delete headers if using original files  
        annot_list = np.zeros((len(list_of_lines_arousal), self.bins+1))

        for l in list_of_lines_arousal:
            elem_list = l.split(',')
            # print('elem_list: ', elem_list)
            elem_list_float = [float(x) for x in elem_list]
            # elem_list_float = [int(x) for x in elem_list[:-1]]
            # elem_list_float.append(float(elem_list[-1]))

            time_ind = elem_list_float[0]
            time_list.append(time_ind)
            del elem_list_float[0] # first element is timestamp

            # print(elem_list_float, 'ELEM_arr')
            
            selected_annot = elem_list_float#[elem_list_float[x] for x in self._annotators_list]
            # mean_annot = mean(selected_annot)
            # annot_list.append(mean_annot)
            annot_list[indx, :] = selected_annot
            indx +=1

        try:
            l = int(len(annot_list)/len(files))
        except:
            print(annot_path)
        # print('LENGTH: ', l)
        # annot_list = np.array(annot_list).reshape((l, len(files)))
        # annot_list = np.mean(annot_list, 1)
        time_list = np.array(time_list)
        # print('annot_list: ', np.shape(annot_list))
                
        return annot_list, time_list          
    
    def _sort_files(self, string_list: list[str]):
        instance_list = [int(s.split('.')[0].split('_')[-1]) for s in string_list] # get instances as an int list
        sorted_ind = np.argsort(instance_list)
        sorted_list = [string_list[i] for i in sorted_ind]
        return sorted_list
    
    def __getitem__(self, idx: int):
        
        arousal_path = self._arousal_files[idx]
        valence_path = self._valence_files[idx]
        video_path = self._video_files[idx]
        subj_id = self._subj_id_list[idx]
        # instance_id = self._instance_id_list[idx]
        
        if 'sewa' in self._data_dir:
            # print('SEWA ANNOtations')
            arousal_annot, timestamps_arousal = self.load_annotations_sewa(arousal_path)
            valence_annot, timestamps_valence = self.load_annotations_sewa(valence_path)
        else:
            arousal_annot, timestamps_arousal = self.load_annotations(arousal_path)
            valence_annot, timestamps_valence = self.load_annotations(valence_path)
            
        # raw_stream, sr = self.load_data(audio_path)
        raw_video = self.load_data_video(video_path)

        if self._clip_length_in_sec != 0:
                
            len_in_frames = len(arousal_annot)
            start_ind, end_ind = get_start_end_ind(self._clip_length_in_sec, self._fps, len_in_frames)
            
            arousal_annot = arousal_annot[start_ind:end_ind]
            valence_annot = valence_annot[start_ind:end_ind]
            timestamps_arousal = timestamps_arousal[start_ind:end_ind]
            timestamps_valence = timestamps_valence[start_ind:end_ind]
            
            
            # raw_stream = raw_stream[start_ind_in_samples:end_ind_in_samples]
            raw_video = raw_video[start_ind:end_ind]

    

        # sample = {'video': raw_stream, 'sr' : sr, 'arousal_annot': arousal_annot, 'valence_annot': valence_annot , 'subj_id': subj_id,'instance_id': instance_id, 'paths': [audio_path, arousal_path,valence_path], 'timestamps': timestamps_arousal}
        sample = {'video': raw_video, 'arousal_annot': arousal_annot, 'valence_annot': valence_annot , 'subj_id': subj_id, 'paths': [video_path, arousal_path,valence_path], 'timestamps': timestamps_arousal}
        # print('Loaded video sample')
        return sample                                


    def __len__(self):
        return len(self._video_files)
    