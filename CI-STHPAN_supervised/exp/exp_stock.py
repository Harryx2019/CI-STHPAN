from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop,trr_loss_mse_rank
from utils.metrics import metric,evaluate

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from torch_geometric.utils import dense_to_sparse

import os
import time
import copy

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Stock(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stock, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args,self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader, hyperedge_index = data_provider(self.args, flag)
        return data_set, data_loader, hyperedge_index

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = trr_loss_mse_rank
        return criterion

    def vali(self, vali_data, vali_loader,vali_hyperedge_index, criterion):
        # hypergraph input
        hyperedge_index =None
        if self.args.graph:
            hyperedge_index = torch.from_numpy(vali_hyperedge_index)
            if self.args.ci == 1:
                hyperedge_index_ci = []
                for m in range(self.args.enc_in):
                    hyperedge_index_ = dense_to_sparse(hyperedge_index[m])
                    hyperedge_index_ = hyperedge_index_[0]
                    hyperedge_index_ci.append(hyperedge_index_.tolist())
                hyperedge_index = torch.LongTensor(hyperedge_index_ci).to(self.device)
            else:
                hyperedge_index = dense_to_sparse(hyperedge_index)
                hyperedge_index = hyperedge_index[0].to(self.device)

        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    mask_batch, price_batch, gt_batch) in enumerate(vali_loader):
                batch_x = batch_x[0]
                batch_y = batch_y[0]
                batch_x_mark = batch_x_mark[0]
                batch_y_mark = batch_y_mark[0]
                mask_batch = mask_batch[0]
                price_batch = price_batch[0]
                gt_batch = gt_batch[0]
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, hyperedge_index)
                        else:
                            batch_x_mark = batch_x_mark.float().to(self.device)
                            batch_y_mark = batch_y_mark.float().to(self.device)

                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x,hyperedge_index)
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss, reg_loss, rank_loss, rr = criterion(outputs.reshape((self.batch_size,1)), 
                                                          torch.FloatTensor(price_batch).to(self.device), 
                                                          torch.FloatTensor(gt_batch).to(self.device), 
                                                          torch.FloatTensor(mask_batch).to(self.device),
                                                          self.args.alpha, self.batch_size,self.device)

                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader, train_hyperedge_index = self._get_data(flag='train')
        vali_data, vali_loader, vali_hyperedge_index = self._get_data(flag='val')

        # hypergraph input
        hyperedge_index =None
        if self.args.graph:
            hyperedge_index = torch.from_numpy(train_hyperedge_index)
            if self.args.ci == 1:
                hyperedge_index_ci = []
                for m in range(self.args.enc_in):
                    hyperedge_index_ = dense_to_sparse(hyperedge_index[m])
                    hyperedge_index_ = hyperedge_index_[0]
                    hyperedge_index_ci.append(hyperedge_index_.tolist())
                hyperedge_index = torch.LongTensor(hyperedge_index_ci).to(self.device)
            else:
                hyperedge_index = dense_to_sparse(hyperedge_index)
                hyperedge_index = hyperedge_index[0].to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # 对称+陡降
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        tarin_loss_epoch = []
        val_loss_epoch = []
        # test_loss_epoch = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    mask_batch, price_batch, gt_batch) in enumerate(train_loader):
                '''
                the batch_size of stock data_loader is 1 [1*len(tickers)*time*features]
                and we should downscaling the data to [len(tickers)*time*features]
                so that the batch_size is len(tickers)
                '''
                batch_x = batch_x[0]
                batch_y = batch_y[0]
                batch_x_mark = batch_x_mark[0]
                batch_y_mark = batch_y_mark[0]
                mask_batch = mask_batch[0]
                price_batch = price_batch[0]
                gt_batch = gt_batch[0]
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
    
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, hyperedge_index)
                        else:
                            batch_x_mark = batch_x_mark.float().to(self.device)
                            batch_y_mark = batch_y_mark.float().to(self.device)

                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    # 线性模型和PatchTST
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, hyperedge_index)
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    
                loss, reg_loss, rank_loss, rr = criterion(outputs.reshape((self.batch_size,1)), 
                                                            torch.FloatTensor(price_batch).to(self.device), 
                                                            torch.FloatTensor(gt_batch).to(self.device), 
                                                            torch.FloatTensor(mask_batch).to(self.device),
                                                            self.args.alpha, self.batch_size,self.device)
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, vali_hyperedge_index, criterion)
            
            tarin_loss_epoch.append(train_loss)
            val_loss_epoch.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path,map_location='cuda'))
        
        # loss save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'train_loss_epoch.npy', tarin_loss_epoch)
        np.save(folder_path + 'val_loss_epoch.npy', val_loss_epoch)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader,test_hyperedge_index  = self._get_data(flag='test')
        # hypergraph input
        hyperedge_index =None
        if self.args.graph:
            hyperedge_index = torch.from_numpy(test_hyperedge_index)
            if self.args.ci == 1:
                hyperedge_index_ci = []
                for m in range(self.args.enc_in):
                    hyperedge_index_ = dense_to_sparse(hyperedge_index[m])
                    hyperedge_index_ = hyperedge_index_[0]
                    hyperedge_index_ci.append(hyperedge_index_.tolist())
                hyperedge_index = torch.LongTensor(hyperedge_index_ci).to(self.device)
            else:
                hyperedge_index = dense_to_sparse(hyperedge_index)
                hyperedge_index = hyperedge_index[0].to(self.device)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                                                  map_location='cuda'))

        results = np.zeros([self.batch_size, len(test_loader)],dtype=float)
        preds = np.zeros([self.batch_size, len(test_loader)],dtype=float)
        trues = np.zeros([self.batch_size, len(test_loader)],dtype=float)
        masks = np.zeros([self.batch_size, len(test_loader)],dtype=float)

        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    mask_batch, price_batch, gt_batch) in enumerate(test_loader):
                batch_x = batch_x[0]
                batch_y = batch_y[0]
                batch_x_mark = batch_x_mark[0]
                batch_y_mark = batch_y_mark[0]
                mask_batch = mask_batch[0]
                price_batch = price_batch[0]
                gt_batch = gt_batch[0]
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                price_batch = price_batch.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, hyperedge_index)
                        else:
                            batch_x_mark = batch_x_mark.float().to(self.device)
                            batch_y_mark = batch_y_mark.float().to(self.device)

                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x,hyperedge_index)
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                outputs = outputs.reshape((self.batch_size,1))
                rr = torch.div((outputs- price_batch), price_batch)
                rr = rr.detach().cpu().numpy()
                
                mask_batch = mask_batch.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                gt_batch = gt_batch.detach().cpu().numpy()
                
                results[:, i] = copy.copy(outputs[:,0])
                preds[:, i] = copy.copy(rr[:, 0])
                trues[:, i] = copy.copy(gt_batch[:, 0])
                masks[:, i] = copy.copy(mask_batch[:, 0])


        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        perf = evaluate(preds, trues, masks)
        print('Test preformance:', perf)
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('Test preformance:{}'.format(perf))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'masks.npy', masks)
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'result.npy', results)
        
        # visualize save
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
