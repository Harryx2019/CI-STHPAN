

import numpy as np
import pandas as pd
import os
import torch
import random
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from datetime import datetime


import argparse

parser = argparse.ArgumentParser()
# random seed
parser.add_argument('--random_seed', type=int, default=2023, help='random seed')
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='stock', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=1, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--market', type=str, default='NASDAQ', help='stock market name')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# CI
parser.add_argument('--ci', type=int, default=1, help='channel independence')
# HGAT
parser.add_argument('--graph', type=int, default=1, help='HyperGraph; True 1 False 0')
parser.add_argument('--rel_type', type=int, default=0, help='relation type; all 0 industry 1 wiki 2 dtw 3 industry_wiki 4 industry_dtw 5 wiki_dtw 6')
# DTW
parser.add_argument('--k', type=int, default=20, help='dtw topk relation')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) \
                            + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + \
                            '_mask' + str(args.mask_ratio) + '_revin' + str(args.revin) + '_ci' + str(args.ci) + \
                            '_graph' + str(args.graph) + '_rel_type' + str(args.rel_type) + '_k' + str(args.k)

args.save_path = '../finetune/saved_models/' + args.market + '/pretrained/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
if not os.path.exists(args.save_path+args.save_pretrained_model+'/'): os.makedirs(args.save_path+args.save_pretrained_model+'/')


# get available GPU devide
set_device()


def get_model(c_in, args):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                ci=args.ci,
                graph=args.graph,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='pretrain',
                res_attention=False
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]
        
    # define learner
    learn = Learner(dls, model, 
                        args.graph, 
                        args.ci,
                        args.rel_type,
                        args.market,
                        args.k,
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    # get model     
    model = get_model(dls.vars, args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname='model',                       
                        path=args.save_path+args.save_pretrained_model+'/model'+str(args.pretrained_model_id)+'/')
        ]
    # define learner
    learn = Learner(dls, model,
                        args.graph, 
                        args.ci,
                        args.rel_type,
                        args.market,
                        args.k,
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        #metrics=[mse]
                        )                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model+'/model'+str(args.pretrained_model_id) + '/losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':
    for itr in range(1):
        starttime = datetime.now()
        
        args.pretrained_model_id = itr
        if not os.path.exists(args.save_path+args.save_pretrained_model+'/model'+str(itr)+'/'):
            os.makedirs(args.save_path+args.save_pretrained_model+'/model'+str(itr)+'/')
        
        args.dset = args.dset_pretrain
        suggested_lr = find_lr()
        # Pretrain
        pretrain_func(suggested_lr)

        endtime = datetime.now()
        time = str((endtime - starttime).seconds)
        print('itr:',str(itr),' pretraining completed！ time:'+time)
    
