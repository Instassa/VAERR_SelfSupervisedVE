# VAERR_SelfSupervisedVE
## Self-Supervised Apparent Emotion Recognition from Video

This code is provided to accompany the homonymous paper for the sake of the reproduceability of the results.
The metiod is based on the self-supervised learning paradigm, where the _pretext_ architecture is trained on an auxiliary and often unrelated task in prder to provide good initialization or learn useful representations for the _downstream_ target task.

In our case we are interested in natural apparent emotional reaction recognition (in terms of arousal and valence) based on the video-only input, or VAERR. 

## Architecture
The part of the architecture pretrained and shared between the pretext and the downstream tasks is the 3D convolutional layer + ResNet18, as shown on the picture below. Followed by a GRU.

<img src="./misc/VideoEmotions_models.png" alt="comparativeArchitectures" width="1100"/>

<!-- ![alt text](https://github.com/Instassa/VAERR_SelfSupervisedVE/blob/main/misc/VideoEmotions_models_.pdf) -->


## Results
This is the first work using the self-supervised setting in this context, presenting the state-of-the-art results for the natural apparent emotional reaction recognition.


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
Pretext pretraining is very similar to the original pretext models presented in the paper, with some minor adjustments to the architecture conducted for comparability sake. The codes and papers can be found here:
- LiRA [paper](https://arxiv.org/abs/2106.09171) and [code](https://github.com/Instassa/Lipreading_ICASSP21_Release) (private repository, would need to request an access from the author);
- BYOL [paper](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf) and [code](https://github.com/lucidrains/byol-pytorch);
- DINO [paper](https://arxiv.org/abs/2104.14294) and [code](https://github.com/facebookresearch/dino).

This repository contains to the downstream portion of the model. 
Please specify the the checkpoint you would like to use as ` --model-path='./ckpts/ckpt_3Dconv.pth' `

To launch the downstream training and subsequent evaluation please run:
```
python -W ignore main.py --lr 0.00007 --epochs 10 --batch-size 5 --workers 20 --clip_length 4 --model-path='./ckpts/ckpt_3Dconv.pth' --dataset='sewa_video' --allow-size-mismatch --loss_type 5 
```
Specifying the parameters of iterest.

_TODO PARAMETERS_

## Evaluation

_TODO_


### For any questions that might arise please contact [Dr. Marija Jegorova](mailto:marijajegorova@fb.com?subject=[GitHub]%20Question%20about%20VAERR)
### If you are using this code in any way please kindly cite _add BibTex here_
