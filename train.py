from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from model.unisiam import UniSiam
from model.resnet import resnet10, resnet18, resnet34, resnet50
from dataset.miniImageNet import miniImageNet
from dataset.tieredImageNet import tieredImageNet
from dataset.sampler import EpisodeSampler
from evaluate import evaluate_fewshot
from transform.build_transform import build_transform
from util import AverageMeter, adjust_learning_rate, save_model



def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--save_path', type=str, default=None, help='path for saving')
    parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
    parser.add_argument('--eval_path', type=str, default=None, help='path to tested model')
    parser.add_argument('--teacher_path', type=str, default=None, help='path to teacher model')
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['tieredImageNet', 'miniImageNet'], help='dataset')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')

    # optimization setting
    parser.add_argument('--lr', type=float, default=0.3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--lrd_step', action='store_true', help='decay learning rate per step')

    # self-supervision setting
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet10', 'resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--size', type=int, default=224, help='input size')
    parser.add_argument('--temp', type=float, default=2.0, help='temperature for loss function')
    parser.add_argument('--lamb', type=float, default=0.1, help='lambda for uniform loss')
    parser.add_argument('--dim_hidden', type=int, default=None, help='hidden dim. of projection')

    # few-shot evaluation setting
    parser.add_argument('--n_way', type=int, default=5, help='n_way')
    parser.add_argument('--n_query', type=int, default=15, help='n_query')
    parser.add_argument('--n_test_task', type=int, default=3000, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int, default=20, help='episode_batch_size')

    args = parser.parse_args()

    args.dist = args.teacher_path is not None
    
    args.lr = args.lr * args.batch_size / 256

    if (args.save_path is not None) and (not os.path.isdir(args.save_path)):
        os.makedirs(args.save_path)
    
    args.split_path =  os.path.join(os.path.abspath(os.path.dirname(__file__)), 'split')

    return args


def build_train_loader(args):
    train_transform = build_transform(args)

    if args.dataset == 'miniImageNet':
        train_dataset = miniImageNet(
            data_path=args.data_path, 
            split_path=args.split_path,
            partition='train',
            transform=train_transform)
    elif args.dataset == 'tieredImageNet':
        train_dataset = tieredImageNet(
            data_path=args.data_path,
            split_path=args.split_path,
            partition='train',
            transform=train_transform)
    else:
        raise ValueError(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    return train_loader


def build_fewshot_loader(args, mode='test'):

    assert mode in ['train', 'val', 'test']

    resize_dict = {160: 182, 224: 256, 288: 330, 320:366, 384:438}
    resize_size = resize_dict[args.size]
    print('Image Size: {}({})'.format(args.size, resize_size))

    test_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    print('test_transform: ', test_transform)

    if args.dataset == 'miniImageNet':
        test_dataset = miniImageNet(
            data_path=args.data_path, 
            split_path=args.split_path,
            partition=mode,
            transform=test_transform)
    elif args.dataset == 'tieredImageNet':
        test_dataset = tieredImageNet(
            data_path=args.data_path,
            split_path=args.split_path,
            partition=mode,
            transform=test_transform)
    else:
        raise ValueError(args.dataset)

    test_sampler = EpisodeSampler(
        test_dataset.labels, args.n_test_task//args.test_batch_size, args.n_way, 5+args.n_query, args.test_batch_size)
    test_loader =torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_sampler, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    return test_loader
  

def build_model(args):
    model_dict = {'resnet10': resnet10, 'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}

    encoder = model_dict[args.backbone]()

    model = UniSiam(encoder=encoder, lamb=args.lamb, temp=args.temp, dim_hidden=args.dim_hidden, dist=args.dist)

    model.encoder = torch.nn.DataParallel(model.encoder)
    model = model.cuda()

    print(model)
    
    return model

def load_teacher_model(args):
    encoder = resnet50()
    teacher_model = UniSiam(encoder=encoder)
    teacher_model.encoder = torch.nn.DataParallel(teacher_model.encoder)
    teacher_model.cuda()
    msg = teacher_model.load_state_dict(torch.load(args.teacher_path)['model'])
    print(f'load teacher model from: {args.teacher_path}, {msg}')
    teacher_model.eval()
    return teacher_model


def train_one_epoch(train_loader, model, optimizer, epoch, args, teacher_model=None):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_hist = AverageMeter()
    loss_pos_hist = AverageMeter()
    loss_neg_hist = AverageMeter()
    std_hist = AverageMeter()

    end = time.time()

    n_iter = len(train_loader)

    for idx, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.lrd_step:
            adjust_learning_rate(args, optimizer, idx*1.0/n_iter+epoch, args.epochs)
        
        bsz = images[0].shape[0]

        images = torch.cat([images[0], images[1]], dim=0).cuda(non_blocking=True)

        if teacher_model is not None:
            with torch.no_grad():
                dist_z = teacher_model.proj(teacher_model.encoder(images)).detach()
            loss, loss_pos, loss_neg, std = model(images, dist_z)
        else:
            loss, loss_pos, loss_neg, std = model(images)

        loss_hist.update(loss.item(), bsz)
        loss_pos_hist.update(loss_pos.item(), bsz)
        loss_neg_hist.update(loss_neg.item(), bsz)
        std_hist.update(std.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss_pos {lossp.val:.3f} ({lossp.avg:.3f})\t'
                  'loss_neg {lossn.val:.3f} ({lossn.avg:.3f})\t'
                   'std {std.val:.3f} ({std.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_hist, lossp=loss_pos_hist, 
                   lossn=loss_neg_hist, std=std_hist))
            sys.stdout.flush()

    return loss_hist.avg


def main():
    args = parse_option()
    print("{}".format(args).replace(', ', ',\n'))

    train_loader = build_train_loader(args)
    test_loader = build_fewshot_loader(args, 'test')

    model = build_model(args)

    teacher_model = load_teacher_model(args) if args.dist else None
    
    cudnn.benchmark = True

    if args.eval_path is not None:
        model.load_state_dict(torch.load(args.eval_path)['model'])
        evaluate_fewshot(model.encoder, test_loader, n_way=args.n_way, n_shots=[1,5], n_query=args.n_query, classifier='LR', power_norm=True)
        return
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    for epoch in range(args.epochs):

        if not args.lrd_step:
            adjust_learning_rate(args, optimizer, epoch+1, args.epochs)

        time1 = time.time()
        loss = train_one_epoch(train_loader, model, optimizer, epoch, args, teacher_model=teacher_model)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

    # evaluate_fewshot(model.encoder, test_loader, n_way=args.n_way, n_shots=[1,5], n_query=args.n_query, classifier='SVM')
    # evaluate_fewshot(model.encoder, test_loader, n_way=args.n_way, n_shots=[1,5], n_query=args.n_query, classifier='LR', power_norm=False)
    evaluate_fewshot(model.encoder, test_loader, n_way=args.n_way, n_shots=[1,5], n_query=args.n_query, classifier='LR', power_norm=True)

    if args.save_path is not None:
        save_file = os.path.join(args.save_path, 'last.pth')
        save_model(model, args.epochs, save_file)

if __name__ == '__main__':
    main()
