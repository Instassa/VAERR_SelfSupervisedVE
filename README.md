# VAERR_SelfSupervisedVE
## CATEGORICAL EMOTIONS. One-hot label per video-clip. Tested on CREMA only.

This is a version of the code in the main branch. The differences being:
- designed for classifying categorical emotions;
- tested only for one label per video clip (as opposed to every-frame annotations);
- the above features are chosen to work for CREMA dataset.

## Results
Produced ~66.5% accuracy on a particular version of CREMA dataset test set.
The pre-trained checkpoint that could be used as an inference initialization can be found [here](https://drive.google.com/file/d/1uNja3HC62faeIogz35SF2aT2mIDVDNNa/view?usp=sharing).


## Setting up the environment
To reproduce the results please first create and activate a conda environment with the corresponding dependencies:
```
conda create --name myenv python=3.8.10 --file requirements_minimal.txt
conda activate myenv
```
OR
```
conda create --name myenv python=3.8.10
conda activate myenv
pip install numpy==1.20.2 scipy==1.1.0 torch==1.8.1+cu111 torchvision==0.9.1+cu111  opencv-python==4.5.2.54 pillow==8.2.0 pykeops==1.5 sklearn==0.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Training
This repository contains to the downstream portion of the model. 
Please specify the the checkpoint you would like to use as ` --model-path='./ckpts/ckpt_3Dconv.pth' ` Please download it [here](https://drive.google.com/file/d/1GxR74bBnuJDSeXFjmUoeV_seiV2Lxmod/view?usp=sharing).

To launch the downstream training and subsequent evaluation please run:
```
python -W ignore main.py --lr 0.0003 --epochs 30 --batch-size 10 --workers 20 --clip_length 0 --model-path='./ckpts/ckpt_3Dconv.pth' --dataset='MSP_video' --allow-size-mismatch --loss-type 15 --fine-tuning FT
```

