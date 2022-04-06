
import torch
from dataloaders.datasetVideo import datasetVideo



def get_data_loaders(args): 
    if args.n_classes > 15:
        print('-------------classification mode---------------')
        dsets = {partition: datasetVideo(data_part_str=partition,   
                                data_dir=args.video_dir[partition],  
                                annot_dir=args.annot_dir[partition],   
                                annot_type = args.annot_type,
                                annot_list= args.annot_list,
                                clip_length_in_sec = args.clip_length_dict[partition],
                                segments_per_file = args.segments_per_file_dict[partition],
                                augs = args.augment,
                                fps = args.fps) for partition in ['train','val','test']}  
    else:
        print('Type of the dataset is not recognized. Please refer to dataloaders.py')  

    dset_loaders = {}
    dset_loaders['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True) 


    batch_size_val_test = 1 # loading sequences of variable length is not supported at the moment 
    
    dset_loaders['val'] = torch.utils.data.DataLoader(dsets['val'],
                                                   batch_size=batch_size_val_test,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   pin_memory=True) 
   
    dset_loaders['test'] = torch.utils.data.DataLoader(dsets['test'],
                                                   batch_size=batch_size_val_test,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   pin_memory=True) 
                                                   
    
    print('\nStatistics: train: {}, val: {}, test: {}'.format(len(dsets['train']), len(dsets['val']), len(dsets['test'])))
        

    
    return dset_loaders
