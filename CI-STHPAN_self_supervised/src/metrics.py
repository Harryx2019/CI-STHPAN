
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics import ndcg_score

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(y_true, y_pred)

# stock loss function
def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, num_stocks, device):
    return_ratio = torch.div((pred- base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    
    all_ones = torch.ones(num_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) 
                   - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) 
                 - torch.matmul(ground_truth, torch.transpose(all_ones, 0,1)))

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0,1))
    rank_loss = torch.mean(F.relu(((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio

def evaluate(prediction, ground_truth, mask):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2 / np.sum(mask)
    
    # top5
    bt_long5 = 1.0
    ndcg_score_top5 = 0.0
    sharpe_li5 = []

    num_stocks = prediction.shape[0]
    y_score = np.array([[5,4,3,2,1]])

    for i in range(prediction.shape[1]):
        # sort index
        rank_gt = np.argsort(ground_truth[:, i])
        # ground truth top 5
        gt_top5 = []
        y_true_ = np.zeros(num_stocks,np.int32)
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top5) < 5:
                gt_top5.append(cur_rank)
            y_true_[cur_rank] = num_stocks - j + 1

        # predict top 5
        rank_pre = np.argsort(prediction[:, i])
        k = 0
        pre_top5 = []
        y_true = np.zeros((1,5),np.int32)
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < 5:
                pre_top5.append(cur_rank)
                y_true[0][k] = y_true_[cur_rank]
                k+=1
            
        # sklearn.metrics.ndcg_score(y_true, y_score, k)
        ndcg_score_top5 = ndcg_score(y_true, y_score)

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        # 累计收益率计算公式
        bt_long5 *= (1+real_ret_rat_top5)
        sharpe_li5.append(real_ret_rat_top5)
    
    performance['btl5'] = bt_long5 - 1
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = (np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87 #To annualize
    performance['ndcg_score_top5'] = ndcg_score_top5

    # 返回技术指标[mse,ndcg_score_top5,btl5(累计收益率),sharpe5(夏普比率)]
    return performance