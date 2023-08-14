
__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB' ]


""" 
Callback lists:
    > before_fit
        - before_epoch
            + before_epoch_train                
                ~ before_batch_train
                ~ after_batch_train                
            + after_epoch_train

            + before_epoch_valid                
                ~ before_batch_valid
                ~ after_batch_valid                
            + after_epoch_valid
        - after_epoch
    > after_fit

    - before_predict        
        ~ before_batch_predict
        ~ after_batch_predict          
    - after_predict

"""

from ..basics import *
import torch
import copy
import numpy as np

DTYPE = torch.float32

class Callback(GetAttr): 
    _default='learner'


class SetupLearnerCB(Callback): 
    def __init__(self):        
        self.device = default_device(use_cuda=True)

    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)        
        if self.n_inp > 1: 
            xb, yb, mask_batch, price_batch, gt_batch = batch
        else: 
            xb, yb = batch, None        
        self.learner.batch = xb, yb, mask_batch, price_batch, gt_batch
        
    def before_fit(self): 
        "Set model to cuda before training"                
        self.learner.model.to(self.device)
        self.learner.device = self.device                        


class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    def before_predict(self):
        self.preds = []        
    
    def after_batch_predict(self):        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)

    def after_predict(self):           
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()

         

class GetTestCB(Callback):
    def __init__(self, tickers_num, test_dates):
        self.i = 0
        self.tickers_num = tickers_num
        self.test_dates = test_dates
        self.device = default_device(use_cuda=True)
        super().__init__()

    def before_test(self):
        self.learner.device = self.device 
        self.results = np.zeros([self.tickers_num, self.test_dates],dtype=float)
        self.preds = np.zeros([self.tickers_num, self.test_dates],dtype=float)
        self.trues = np.zeros([self.tickers_num, self.test_dates],dtype=float)
        self.masks = np.zeros([self.tickers_num, self.test_dates],dtype=float)
    
    def after_batch_test(self):        
        # append the prediction after each forward batch 
        # self.result = self.result.detach().cpu().numpy()
        # self.pred = self.pred.detach().cpu().numpy()
        # self.true = self.true.detach().cpu().numpy()
        # self.mask = self.mask.detach().cpu().numpy()
        
        i = self.i
        self.results[:, i] = copy.copy(self.result[:,0].cpu().numpy().tolist())
        self.preds[:, i] = copy.copy(self.pred[:, 0].cpu().numpy().tolist())
        self.trues[:, i] = copy.copy(self.true[:, 0].cpu().numpy().tolist())
        self.masks[:, i] = copy.copy(self.mask[:, 0].cpu().numpy().tolist())
        self.i += 1

    def after_test(self):           
        print("self.results shape:",self.results.shape)
        print("self.preds shape:",self.preds.shape)
        print("self.trues shape:",self.trues.shape)
        print("self.masks shape:",self.masks.shape)


