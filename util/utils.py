# original code: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py

import torch
import random
import csv
import torch.nn as nn
from math import exp
from torchvision import datasets, transforms
import os
import numpy as np
import torch.distributed as dist

def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors
    
    
def GetDataloader(args):
    dataset = args.dataset
    # Cifar
    if 'cifar' in dataset:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        if dataset == 'cifar10':
            args.num_classes = 10
            data_func = datasets.CIFAR10
        elif dataset == 'cifar100':
            args.num_classes = 100
            data_func = datasets.CIFAR100
        else:
            NotImplementedError

        train_dataset = data_func(root='data/', train=True, transform=train_transform, download=True)
        val_dataset = data_func(root='data/', train=False, transform=test_transform, download=True)

    elif 'imagenet' in dataset:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])        
        jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4)
        lighting = Lighting(alphastd=0.1,
                            eigval=[0.2175, 0.0188, 0.0045],
                            eigvec=[[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])

        # tiny_imagenet
        if dataset == 'tiny_imagenet':
            args.num_classes = 200
            data_root = 'data/tiny_imagenet'
            img_size = args.img_size
            test_img_size = img_size
        # imagenet
        elif dataset == 'imagenet':
            args.num_classes = 1000
            data_root = 'data/imagenet'
            img_size = 224
            test_img_size = 256

        train_dataset = datasets.ImageFolder(
            os.path.join(data_root, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,            
                normalize,
            ]))    
        val_dataset = datasets.ImageFolder(
            os.path.join(data_root, 'val'),
            transforms.Compose([
                transforms.Resize(test_img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler, val_sampler = None, None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, )

    return train_loader, val_loader, args

class GaussianNoise(nn.Module):
    def __init__(self, net, sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.net = net 

    def forward(self, x):
        if self.training:
            x_mean = torch.mean(x)
            gaussian_noise = torch.empty_like(x).normal_(0, x_mean * self.sigma)
            x += gaussian_noise
        return self.net(x)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, groups=1, smoothing=0.0, dim=-1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.groups = groups
        self.cls = classes
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim).view(-1, self.groups, self.cls)
        target = target.view(-1, self.groups, 1)
        loss_list = []
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            smooth_factor = 0
            for i in range(self.groups):
                true_dist[:,i].fill_(smooth_factor / (self.cls - 1)) 
                true_dist[:,i].scatter_(1, target[:,i], 1 - smooth_factor)
                smooth_factor += self.smoothing
        # import pdb; pdb.set_trace; from IPython import embed; embed()
        loss = torch.sum(-true_dist.view(-1, self.cls) * pred.view(-1, self.cls), dim=self.dim)
        if self.reduction == 'mean':
            loss = loss.sum()
        elif self.reduction == 'sum':
            loss = loss.mean()
        return loss

class ClsAttLoss(nn.Module):
    def __init__(self, classes, groups=1, noise=0, cls_acc=None):
        super(ClsAttLoss, self).__init__()
        self.groups = groups
        self.cls = classes
        mean, noise, t = noise.split('_')
        mean, noise, t = float(mean), float(noise), float(t)
        if mean == 0:
            cls_weights = torch.ones(groups, classes)
        else:
            if cls_acc is None:
                if noise == 0:
                    # fixed step-wise value
                    cls_weights = torch.exp(mean * torch.arange(groups).view(-1, 1).repeat(1, classes))
                    for i in range(classes):
                        rand_idx = torch.randperm(groups)
                        cls_weights[:,i] = cls_weights[rand_idx, i]
                else:
                    # variable cls weights by gaussian or softmax 
                    cls_weights = torch.empty(groups, classes).normal_(1, noise)
                    if t != 0:
                        cls_weights = torch.softmax(cls_weights / t, 1) - 1. / groups
                        cls_weights = torch.exp(cls_weights * mean)
            else:
                # import pdb; pdb.set_trace; from IPython import embed; embed()
                if noise == 0:
                    # variable
                    norm_group_acc = torch.softmax((cls_acc - cls_acc.mean(0) ) / t, 0) - 1. / groups 
                    cls_weights = torch.exp(mean * norm_group_acc)
                elif noise < 0:
                    # direct exp wt
                    norm_group_acc = cls_acc - cls_acc.mean(0)
                    cls_weights = torch.exp(mean * norm_group_acc / t)
                else:
                    cls_weights = torch.zeros_like(cls_acc)
                    _, acc_sort = torch.sort(cls_acc, dim=0)
                    for i in range(groups):
                        cls_weights[acc_sort[i], torch.arange(classes)] = exp(i * mean / t)

        # print(f"cls_wt: {cls_weights}")
        self.register_buffer('cls_weights', cls_weights)

    def forward(self, pred, target):
        pred = pred.view(-1, self.groups, self.cls)
        target = target.view(-1, self.groups)
        loss_list = []
        for i in range(self.groups):
            group_cls_weight = self.cls_weights[i]
            loss = nn.functional.cross_entropy(pred[:,i], target[:,i], group_cls_weight, reduction='none')
            batch_cls_norm = target.size(0) / group_cls_weight[target[:,i]].sum()
            loss_list.append(loss[:,None] * batch_cls_norm)

        return torch.cat(loss_list, 1).view(-1)

class AddDropLayer(nn.Module):
    def __init__(self, net, ratio=.5):
        super(AddDropLayer, self).__init__()
        self.net = net
        self.drop = nn.Dropout2d(ratio)

    def forward(self, x):
        x = self.drop(x)
        return self.net(x)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 2, True, True)
    correct = pred.eq(target[:,None,None].expand_as(pred))

    res = []
    for k in topk:
        correct_k = (correct[:,:, :k].sum(2) > 0).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        # gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[0].mul_(0.299).add_(gs[1], alpha=0.587).add_( gs[2], alpha=0.114)
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        return transform(img)
