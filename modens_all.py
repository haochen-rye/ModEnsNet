import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from time import gmtime, strftime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import subprocess  
from util.utils import CSVLogger, GetDataloader, accuracy, AverageMeter, ProgressMeter
from util.resnext import ResNet as ResNeXt
from util.mobile_v2 import MobileNetV2
# from ptflops import get_model_complexity_info 
from fvcore.nn.flop_count import flop_count

parser = argparse.ArgumentParser(description='PyTorch DSE Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mv2')
parser.add_argument('--dataset',  default='cifar100')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--steps', default=[75, 125], type=int, nargs="+")
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--warmup', type=float, default=5)                    
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd',  default=1e-4, type=float, dest='weight_decay',
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')                    
parser.add_argument('--weight', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.66.41.62:2345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--boost_groups', type=int, default=[1,], nargs='+')
parser.add_argument('--split_groups', type=int, default=[1,], nargs='+')
parser.add_argument('--suffix', type=str, default='')                    
parser.add_argument('--no_resume', action='store_true', default=False)                    
parser.add_argument('--no_optimizer', action='store_true', default=False)                    
parser.add_argument('--cos-lr', action='store_true', default=False)                    
parser.add_argument('--avg-losses', action='store_true', default=False)    
parser.add_argument('--scale-gradient',  action='store_true', default=False)                  
parser.add_argument('--fuse-stages', default=0, type=int, help='number of stages for fuse convolution')
parser.add_argument('--debug',  action='store_true', default=False)                  
parser.add_argument('--img-size', default=112, type=int)
parser.add_argument('--no-relu',  action='store_true', default=False)                  
parser.add_argument('--width-mult', default=1, type=float, help='width multiply for model')

best_single_acc1, best_ens_acc1 = 0, 0

def main():
    args = parser.parse_args()

    step_str = ','.join(str(x) for x in args.steps) 
    group_str =  'groups{}'.format(','.join([str(x) for x in args.boost_groups]))
    split_str =  'split{}'.format(','.join([str(x) for x in args.split_groups]))
    loss_str = '_{}Loss'.format('avg' if args.avg_losses else 'sum')
    extra_str = "{}{}".format('_scale_gradient' if args.scale_gradient else '', '_CosLr' if args.cos_lr else '')  
    log_dir = 'logs' + ('_debug' if args.debug else '')
    args.log_base_dir = f'{log_dir}/{args.dataset}/{args.arch}/DSE_e{args.epochs}_{step_str}_Warm{args.warmup}_lr{args.lr}_b{args.batch_size}_wd{args.weight_decay}' + \
        f'mult{args.width_mult}_{group_str}_{split_str}_{loss_str}{extra_str}/run_{args.suffix}'

    args.ckt_dir = os.path.join(args.log_base_dir, 'models')
    if not os.path.exists(args.ckt_dir):
        os.makedirs(args.ckt_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_single_acc1, best_ens_acc1
    args.gpu = gpu

    save_on_this_worker= not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.gpu == 0)

    if save_on_this_worker:
        writer = SummaryWriter(os.path.join(args.log_base_dir, 'tensorboard')) 
        csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc@1', 
            'train_ens_acc@1', 'test_acc@1', 'ens_acc@1', 'test_acc@5', 'best_test_acc@1'], 
            filename=os.path.join(args.log_base_dir, 'log.csv'))
    else:
        writer = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Get the dataloader and num_classes by datasets
    train_loader, val_loader, args = GetDataloader(args)

    if 'rex' in args.arch:
        stage_num = 6
    elif 'mv2' in args.arch:
        stage_num = 9
    elif 'ev2' in args.arch:
        stage_num = 8

    boost_group_list = [1] + args.boost_groups
    for _ in range(len(boost_group_list), stage_num):
        boost_group_list.append(boost_group_list[-1])

    split_group_list = [1] + args.split_groups
    for _ in range(len(split_group_list), stage_num):
        split_group_list.append(split_group_list[-1])


    total_groups = [x * y for x,y in zip(split_group_list, boost_group_list)]
    args.full_groups = total_groups[-1]
    print(f'boost_group_list: {boost_group_list}\t split_groups: {split_group_list}')

    ch_scale_mv = {1:1, 2:1.25, 3:1.4, 4:1.45}
    ch_scale_res = {1:1, 2:1.35, 3:1.65, 4:1.85}
    ch_scale_rex = {1:1, 2:1.35, 3:1.5, 4:1.7}

    # ONLY one version og group ensemble is supported
    assert np.prod(boost_group_list) == 1 or np.prod(split_group_list)

    if 'rex' in args.arch:
        depth, groups, width_per_group = [int(ele) for ele in args.arch.split('_')[1:]]
        if 'imagenet' in args.dataset:
            if depth == 50:
                layers = [3, 4, 6, 3]
            elif depth == 101:
                layers = [3, 4, 23, 3]
            else:
                NotImplementedError
        else:
            layers = [(depth - 2) // 9 for _ in range(3)]
        model = ResNeXt(layers, args.num_classes, groups, width_per_group, boost_groups=boost_group_list,
              split_groups=split_group_list, wide_factor=ch_scale_rex)
    elif 'mv2' in args.arch:
        model = MobileNetV2(args.num_classes, args.width_mult, boost_groups=boost_group_list, split_groups=split_group_list, 
            wide_factor=ch_scale_mv, fuse_stages=args.fuse_stages, scale_gradient=args.scale_gradient)
    else:
        NotImplementedError

    if save_on_this_worker:
        print(model)
        if 'cifar' in args.dataset:
            img_size = 32
        elif args.dataset == 'tiny_imagenet':
            img_size = args.img_size
        elif args.dataset == 'imagenet':
            img_size = 224
        inputs = torch.randn(1, 3, img_size, img_size)
        count_dict, *_ = flop_count(model, inputs)
        args.flops = sum(count_dict.values())
        args.params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        print(f'flops: {args.flops}G\t params: {args.params}M')
        print(args)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if not args.no_resume:
        if args.weight == '':
            checkpoint_path = os.path.join(args.ckt_dir, 'checkpoint.pth.tar')
        else:
            checkpoint_path = args.weight
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(checkpoint_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_single_acc1 = checkpoint['best_single_acc1']
            best_ens_acc1 = checkpoint['best_ens_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_single_acc1 = best_single_acc1.to(args.gpu)
                best_ens_acc1 = best_ens_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            if not args.no_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True
   
    if args.evaluate:
        test_acc1, ens_test_acc1, ens_test_acc5 = validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_acc1, train_ens_acc1, train_ens_acc5 = train(train_loader, model, criterion, optimizer, epoch, writer, args)

        # evaluate on validation set
        if save_on_this_worker and ((epoch + 1) % args.eval_freq == 0 or epoch > 0.9 * args.epochs):
            test_acc1, ens_test_acc1, ens_test_acc5 = validate(val_loader, model, criterion, args)
            writer.add_scalar('Val/acc1', test_acc1.item(), epoch)
            writer.add_scalar('Val/ens_acc1', ens_test_acc1.item(), epoch)
            writer.add_scalar('Val/ens_acc_gap', ens_test_acc1.item() - test_acc1.item(), epoch)
            writer.add_scalar('Val/ens_acc5', ens_test_acc5.item(), epoch)
            writer.add_scalar('Val/val_train_gap', ens_test_acc1.item() - train_ens_acc1.item(), epoch)
        else:
            test_acc1, ens_test_acc1, ens_test_acc5 = [torch.zeros_like(train_acc1) for _ in range(3)]

        is_single_best = test_acc1 > best_single_acc1
        best_single_acc1 = max(test_acc1, best_single_acc1)

        is_ens_best = ens_test_acc1 > best_ens_acc1
        best_ens_acc1 = max(ens_test_acc1, best_ens_acc1)

        if save_on_this_worker:
            row = {'epoch': str(epoch), 'train_acc@1': str(round(train_acc1.item(), 2)),
                'train_ens_acc@1': str(round(train_ens_acc1.item(), 2)),
                'test_acc@1': str(round(test_acc1.item(), 2)), 
                'ens_acc@1': str(round(ens_test_acc1.item(), 2)), 
                'test_acc@5': str(round(ens_test_acc5.item(), 2)), 
                'best_test_acc@1': str(round(best_ens_acc1.item(), 2))}        
            csv_logger.writerow(row)
            writer.add_scalar('Train/train_acc', train_acc1.item(), epoch)
            writer.add_scalar('Train/train_ens_acc', train_ens_acc1.item(), epoch)
            writer.add_scalar('Train/ens_gap', train_ens_acc1.item() - train_acc1.item(), epoch)
            writer.add_scalar('Val/best_single_acc1', best_single_acc1.item(), epoch)
            writer.add_scalar('Val/best_ens_acc1', best_ens_acc1.item(), epoch)
            model_ckt = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_single_acc1': best_single_acc1,
                'best_ens_acc1': best_ens_acc1,
                'optimizer' : optimizer.state_dict(),
            }
            save_checkpoint(model_ckt, is_ens_best, args)
            if (epoch + 1) % 10 == 0:
                save_checkpoint(model_ckt, False, args, f'epoch{epoch+1}.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    ens_top1 = AverageMeter('Ens_Acc@1', ':6.2f')
    ens_top5 = AverageMeter('Ens_Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, ens_top1, ens_top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.debug and i > 5:
            break

        adjust_learning_rate(optimizer, epoch, i / len(train_loader), args)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, ens_output = model(images)
        loss = criterion(output.view(-1, args.num_classes), target.repeat_interleave(args.full_groups)).mean()
        loss *= 1 if args.avg_losses else args.full_groups

        # measure accuracy and record loss
        acc1,  = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(max(acc1), images.size(0))

        ens_acc1,  ens_acc5 = accuracy(ens_output, target, topk=(1, 5))
        ens_top1.update(max(ens_acc1), images.size(0))
        ens_top5.update(max(ens_acc5), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and writer is not None:
            progress.display(i)

    return top1.avg, ens_top1.avg, ens_top5.avg

@torch.no_grad()
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    ens_top1 = AverageMeter('Ens_Acc@1', ':6.2f')
    ens_top5 = AverageMeter('Ens_Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, ens_top1, ens_top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        if args.debug and i > 5:
            break
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, ens_output = model(images)
        loss = criterion(output.view(-1, args.num_classes), target.repeat_interleave(args.full_groups)).mean()

        # measure accuracy and record loss
        acc1,  = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.max(), images.size(0))

        ens_acc1,  ens_acc5 = accuracy(ens_output, target, topk=(1, 5))
        ens_top1.update(max(ens_acc1), images.size(0))
        ens_top5.update(max(ens_acc5), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' Testing Acc@1 {top1.avg:.3f} Ens_Acc@1 {ens_top1.avg:.3f} Ens_Acc@5 {ens_top5.avg:.3f}'
            .format(top1=top1, ens_top1=ens_top1, ens_top5=ens_top5))

    return top1.avg, ens_top1.avg, ens_top5.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    directory = args.ckt_dir
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        torch.save(state,  os.path.join(directory, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, ite_ratio, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    warmup = args.warmup
    if epoch < warmup:
        lr = args.lr * (0.1 + 0.9 * (epoch + ite_ratio) / warmup)
    else:
        if args.cos_lr:
            from math import cos, pi
            lr = args.lr * (1 + cos(pi * (epoch + ite_ratio - warmup) / (args.epochs - warmup))) * 0.5
        else:
            lr = args.lr * (0.1 ** sum(epoch >= np.array(args.steps)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()