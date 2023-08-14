

import numpy as np
import pandas as pd
import os
import torch
import random
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.core import *
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
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='stock', help='dataset name')
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
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--alpha', type=int, default=4, help='the alpha of the trr_loss_mse_rank')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print('args:', args)

args.save_path = 'saved_models/' + args.market + '/finetuned/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

# pretrained model path
pretrained_model_path = 'patchtst_pretrained_cw512_patch'+str(args.patch_len)+'_stride'+str(args.stride)+'_epochs-pretrain100_mask0.4_revin'+str(args.revin) + \
                        '_ci'+str(args.ci)+'_graph'+str(args.graph)+'_rel_type'+str(args.rel_type)+'_k'+str(args.k)+'/model0'
pretrained_model_path = os.path.join(args.save_path,'..','pretrained',pretrained_model_path,'model.pth')
args.pretrained_model = pretrained_model_path

# transfer learning model path
# transfer_market = 'NASDAQ'
# transfer_model_path = 'saved_models/' + transfer_market + '/pretrained/'
# pretrained_model_path = 'patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain100_mask0.4_revin1_ci1_graph1_rel_type3_k20/model0'
# pretrained_model_path = os.path.join(transfer_model_path,pretrained_model_path,'model.pth')
# args.pretrained_model = pretrained_model_path


# get available GPU devide
set_device()

def get_model(c_in, args, head_type, weight_path=None):
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
                head_type=head_type,
                res_attention=False,
                store_attn=False
                )    
    if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr(head_type):
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args, head_type)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    if head_type == 'pretrain':
        loss_func = torch.nn.MSELoss(reduction='mean')
    else:
        loss_func = trr_loss_mse_rank
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
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
                        head_type=head_type,
                        alpha=args.alpha
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_finetuned_model + '/losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr,head_type='prediction'):
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    # weight_path = args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    if head_type == 'pretrain':
        loss_func = torch.nn.MSELoss(reduction='mean')
    else:
        loss_func = trr_loss_mse_rank   
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname='model', path=args.save_finetuned_model)
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
                        # metrics=[mse],
                        head_type=head_type,
                        alpha=args.alpha
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10)
    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )                            
    # fit the data to the model
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    save_recorders(learn)


def test_func(market,weight_path):
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, 
                        args.graph, 
                        args.ci,
                        args.rel_type,
                        args.market,
                        args.k,cbs=cbs)
    out  = learn.test(dls.test,market = market, weight_path=os.path.join(weight_path,'model.pth'), evaluate=evaluate)         # out: a list of [results, preds, trues, preformance, irr5]
    print('Test preformance:', out[3])
    # save results
    np.save(weight_path + '/result.npy', out[0])
    np.save(weight_path + '/pred.npy', out[1])
    # np.save(weight_path + '/true.npy', out[2])
    
    mse = out[3]['mse']
    btl5 = out[3]['btl5']
    sharpe5 = out[3]['sharpe5']
    ndcg5 = out[3]['ndcg_score_top5']
    
    pd.DataFrame(np.array([mse,btl5,sharpe5,ndcg5]).reshape(1,4), \
                 columns=['mse','btl5','sharpe5','ndcg5']) \
        .to_csv(weight_path+ '/pref.csv', float_format='%.6f', index=False)
    return out



if __name__ == '__main__':
        
    if args.is_finetune:
        for itr in range(5):
            starttime = datetime.now()
            args.dset = args.dset_finetune
            args.finetuned_model_id = itr
            
            suffix_name = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + \
                            '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + \
                            '_epochs-finetune' + str(args.n_epochs_finetune) + \
                            '_revin' + str(args.revin) + '_ci' + str(args.ci) + '_a' + str(args.alpha)+\
                            '_graph' + str(args.graph) + '_rel_type' + str(args.rel_type) + '_k' + str(args.k)

            save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name

            args.save_finetuned_model = os.path.join(args.save_path, save_finetuned_model, 'model'+str(itr))
            if not os.path.exists(args.save_finetuned_model): os.makedirs(args.save_finetuned_model)
            
            
            # Finetune
            suggested_lr = find_lr(head_type='prediction')        
            finetune_func(suggested_lr,head_type='prediction')        
            print('finetune completed')
            # Test
            out = test_func(args.market, args.save_finetuned_model)      
                
            endtime = datetime.now()
            time = str((endtime - starttime).seconds)         
            print('----------- alpha:',str(args.alpha),' itr:',str(itr),'Complete! Time:'+time+'-----------')

    elif args.is_linear_probe:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        linear_probe_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model)        
        print('----------- Complete! -----------')

    else:
        args.dset = args.dset_finetune
        weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
        # Test
        out = test_func(weight_path)        
        print('----------- Complete! -----------')


