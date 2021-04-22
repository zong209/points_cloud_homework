#! /usr/bin/env python
#-*- encoding:utf-8 -*-
'''
@File:      train.py
@Time:      2021/04/19 16:04:12
@Author:    Gaozong/260243
@Contact:   260243@gree.com.cn/zong209@163.com
@Describe:  Train pointNet++
'''

import os
import sys
import torch
import numpy as np

import datetime
import logging
import argparse
import tqdm
import preprocess

from torch.utils.data import DataLoader
from pathlib import Path
from datasets import ModelNet40Dataset
from pointnet_cls import PointNetClassfier, PointNetClassfierLoss

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = base_dir
sys.path.append(os.path.join(root_dir, "models"))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=True, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=47, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

def train(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    args = parse_args()

    # params
    os.environ["CUDA_VISIBLE_DEVICE"] = args.gpu

    # create dir
    time_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    exp_dir = Path("./log/")
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classfication')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(time_str)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("Classfier Model")
    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETERS ...')
    log_string(args)

    log_string("Load dataset ...")
    data_path = "dataset/modelnet40_normal_resampled"

    train_dateset = ModelNet40Dataset(root_dir=data_path)
    test_dataset = ModelNet40Dataset(root_dir=data_path, train_mode=False)
    trainDataLoader = DataLoader(train_dateset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    num_class = args.num_category
    model = PointNetClassfier(num_class, normal_channel = args.use_normals)
    criterion = PointNetClassfierLoss()
    
    if not args.use_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(checkpoints_dir + '/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string("Use pretrain model")
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, betas=(0.9,0.999),eps=1e-8,weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc =0.0
    best_class_acc =0.0

    log_string("Start training ...")
    for epoch in range(start_epoch, args.epoch):
        log_string("Epoch %d (%d/%s):"%(global_epoch+1, epoch+1, args.epoch))
        mean_correct = []
        model = model.train()

        log_string("Train Dataset length:{}".format(len(trainDataLoader)))
        for batch_id, (points, target) in tqdm.tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = preprocess.random_point_dropout(points)
            points[:,:,0:3] = preprocess.random_scale_point_cloud(points[:,:,0:3])
            points[:,:,0:3] = preprocess.shift_point_cloud(points[:,:,0:3])
            points = torch.Tensor(points)
            points = points.transpose(2,1)
            if not args.use_cpu:
                points, target = points.cuda() ,target.cuda()
            pred, feature = model(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step +=1
        
        train_instance_acc = np.mean(mean_correct)
        log_string("Train Instance Accuracy: %f" %train_instance_acc)

        with torch.no_grad():
            instance_acc,class_acc = test(model.eval(), testDataLoader, num_class=num_class)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch+1
            
            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                log_string("Save model ...")
                save_path = os.path.join(checkpoints_dir,"best_model_{}.pth".format(best_instance_acc))
                state = {
                    epoch: best_epoch,
                    instance_acc: instance_acc,
                    class_acc: class_acc,
                    model_state_dict: model.state_dict(),
                    optimizer_state_dict: model.state_dict()
                }
                torch.save(state, save_path)
            global_epoch +=1

        scheduler.step()
    log_string("End of training ...")

if __name__ =="__main__":
    args = parse_args()
    train(args)