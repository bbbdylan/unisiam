# Self-Supervision Can Be a Good Few-Shot Learner

<p align="center">
  <img src="https://user-images.githubusercontent.com/60600462/179650280-165db647-31d9-42b1-a69f-f726f2f0c12d.png" width="700">
</p>

## Updates
- **20.03.2023**: It is recommended to run the code on 4 GPUs.


This is a PyTorch re-implementation of the paper [Self-Supervision Can Be a Good Few-Shot Learner (ECCV 2022)](https://arxiv.org/abs/2207.09176).


```
@inproceedings{Lu2022Self,
	title={Self-Supervision Can Be a Good Few-Shot Learner},
	author={Lu, Yuning and Wen, Liangjian and Liu, Jianzhuang and Liu, Yajing and Tian, Xinmei},
	booktitle={European Conference on Computer Vision (ECCV)},
	year={2022}
}
```
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on?p=self-supervision-can-be-a-good-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on-1)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on-1?p=self-supervision-can-be-a-good-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on-2)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on-2?p=self-supervision-can-be-a-good-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on-3)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on-3?p=self-supervision-can-be-a-good-few-shot)


## Data Preparation
### mini-ImageNet
* download the mini-ImageNet dataset from [google drive](https://drive.google.com/file/d/1BfEBMlrf5UT4aNOoJPaa83CgbGWZAAAk/view?usp=sharing) and unzip it.
* download the [split files](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet) of mini-ImageNet which created by [Ravi and Larochelle](https://openreview.net/pdf?id=rJY0-Kcll).
* move the split files to the folder `./split/miniImageNet`

### tiered-ImageNet
* download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).
* download the [raw split files](https://github.com/yaoyao-liu/tiered-imagenet-tools/tree/master/tiered_imagenet_split) of tiered-ImageNet which created by [Ren et al.](https://arxiv.org/pdf/1803.00676.pdf).
* move the raw split files to the folder `./split/tieredImageNet/raw`
* generate split files:
```
python ./split/tiered_split.py \
  --imagenet_path [your imagenet-train-folder]
```


## Unsupervised Training

Only **DataParallel** training is supported.

Run 
```python ./train.py --data_path [your DATA FOLDER] --dataset [DATASET NAME] --backbone [BACKBONE] [--OPTIONARG]```

For example, to train UniSiam model with ResNet-50 backbone and strong data augmentations on the mini-ImageNet dataset in 4*V100:
```
python train.py \
  --dataset miniImageNet \
  --backbone resnet50 \
  --lrd_step \
  --data_path [your mini-imagenet-folder] \
  --save_path [your save-folder]
```

More configs can be found in `./config`.


## Unsupervised Training with Distillation

Run 
```python ./train.py --teacher_path [your TEACHER MODEL] --data_path [your DATA FOLDER] --dataset [DATASET NAME] --backbone [BACKBONE] [--OPTIONARG]```

With a pre-trained UniSiam (teacher) model, to train UniSiam model with ResNet-18 backbone and on the mini-ImageNet dataset in 4*V100:
```
python train.py \
  --dataset miniImageNet \
  --backbone resnet18 \
  --lrd_step \
  --data_path [your mini-imagenet-folder] \
  --save_path [your save-folder] \
  --teacher_path [your teacher-model-path]
```

More configs can be found in `./config`. You can train a teacher model with ResNet-50 backbone yourself or just download the provided pre-trained model as the teacher model.


## Models
Our pre-trained ResNet models (with 224 image size and strong augmentations) can be downloaded from [google drive](https://drive.google.com/drive/folders/1N_5ZiI73TfFFFOudWDPuNXeYc2dHzKYU?usp=sharing).



## Acknowledgements

Some codes borrow from [SimSiam](https://github.com/facebookresearch/simsiam), [SupContrast](https://github.com/HobbitLong/SupContrast), [(unofficial) SimCLR](https://github.com/AndrewAtanov/simclr-pytorch), [RFS](https://github.com/WangYueFt/rfs).
