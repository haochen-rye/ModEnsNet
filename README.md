# ModEnsNet
PyTorch implementation of two ensemble paper
1. *[Modular Ensemble: Building Model Ensemble via Layer Reuse](https://arxiv.org/abs/2007.00649)* 
2. *[Group Ensemble: Learning an Ensemble of ConvNets in a single ConvNet](https://arxiv.org/abs/2007.00649)* 

## Requirements

First create a new virtual environment (conda or virtualenv).

Then

```bash
pip install -r requirements.txt
```
Depending on your cuda version, you may need to follow [homepage](https://pytorch.org/get-started/locally/) to install **pytorch**.

Our code is tested under **pytorch 1.8.1**, **cuda/10.2**.

## Dataset
- For CIFAR, the data will automatically be downloaded in *./data* directory.

- For ImageNet, download the dataset from [ImageNet](http://www.image-net.org/)
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)


## Training 
Sample commands to run on CIFAR
```bash
#  MobileNetV2 baseline on CIFAR 
python modens_all.py --dataset cifar10 --steps 160 200 --epochs 240 -b 128 --lr 0.1 \
     --suffix 'baseline'  -a mv2

# Group Ensemble, --boost_groups specify the group for each stage
python modens_all.py --dataset cifar10 --steps 160 200 --epochs 240 -b 128 --lr 0.1 \
     --suffix 'mdens'  -a mv2 --boost_groups 1 1 1 1 1 3 --avg-losses


# Modular Ensemble, --split_groups specify the group for each stage
python modens_all.py --dataset cifar10 --steps 160 200 --epochs 240 -b 128 --lr 0.1 \
     --suffix 'mdens'  -a mv2 --split_groups 1 1 1 1 1 3 --avg-losses

#  ResNet with 56 layers baseline on CIFAR 
python modens_all.py --dataset cifar10 --steps 160 200 --epochs 240 -b 128 --lr 0.1 \
     --suffix 'baseline'  -a rex_56_64_1

#  ResNeXt-32x4 with 29 layers baseline on CIFAR 
python modens_all.py --dataset cifar10 --steps 160 200 --epochs 240 -b 128 --lr 0.1 \
     --suffix 'baseline'  -a rex_29_32_4

     

```
Sample commands to run on ImageNet

```bash
#  MobileNetV2 on ImageNet 
python modens_all.py -a mv2 --suffix 'baseline' -j 16 --width-mult 2.5  \
    --dataset imagenet --epochs 150 --steps 75 120 145  -b 256 --lr 0.05 --cos-lr

# Group ensemble 
python modens_all.py -a mv2 --suffix 'gpens' -j 16 --width-mult 2.5  \
    --dataset imagenet --epochs 150 --steps 75 120 145  -b 256 --lr 0.05 --cos-lr \
    --boost_groups 1 1 1 1 2 

# Modular ensemble 
python modens_all.py -a mv2 --suffix 'mdens' -j 16 --width-mult 2.5  \
    --dataset imagenet --epochs 150 --steps 75 120 145  -b 256 --lr 0.05 --cos-lr \
    --split_groups 1 1 1 1 2 

# ResNeXt-32x4 with 50 layers on ImageNet
python dse_all.py  -a rex_50_32_4 --suffix 'baseline'  -j 16 --wd 1e-4 \
    --dataset imagenet --epochs 90 --steps 30 60  --lr 0.1 -b 256 

python dse_all.py  -a rex_50_32_4 --suffix 'mdens'  -j 16 --wd 1e-4 \
    --dataset imagenet --epochs 90 --steps 30 60  --lr 0.1 -b 256 --split_groups 1 1 1 3

```
 
## Cite
Please cite our papers if they help your research