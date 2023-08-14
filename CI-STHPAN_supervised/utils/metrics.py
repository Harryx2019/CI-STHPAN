import numpy as np
from sklearn.metrics import ndcg_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


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

def evaluate_topk(prediction, ground_truth, mask):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    # print('gt_rt',np.max(ground_truth))
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2 / np.sum(mask)
    
    # top1
    bt_long1 = 1.0
    sharpe_li1 = []
    irr1 = []
    # top5
    bt_long5 = 1.0
    ndcg_score_top5 = 0.0
    sharpe_li5 = []
    irr5 = []
    selected_stock5 = []
    # top10
    bt_long10 = 1.0
    ndcg_score_top10 = 0.0
    sharpe_li10 = []
    irr10 = []
    # top15
    bt_long15 = 1.0
    ndcg_score_top15 = 0.0
    sharpe_li15 = []
    irr15 = []
    # top20
    bt_long20 = 1.0
    ndcg_score_top20 = 0.0
    sharpe_li20 = []
    irr20 = []

    for i in range(prediction.shape[1]):
        # 返回索引
        rank_gt = np.argsort(ground_truth[:, i])
        # 真实前20名排序
        gt_top20 = []
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top20) < 20:
                gt_top20.append(cur_rank)

        # 预测前20名排序
        rank_pre = np.argsort(prediction[:, i])
        pre_top20 = []
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top20) < 20:
                pre_top20.append(cur_rank)
        # 保存前5选股
        selected_stock5.append(pre_top20[:5])
        # sklearn.metrics.ndcg_score(y_true, y_score, k)
        ndcg_score_top5 += ndcg_score(np.array(list(gt_top20[:5])).reshape(1,-1), np.array(list(pre_top20[:5])).reshape(1,-1))
        ndcg_score_top10 += ndcg_score(np.array(list(gt_top20[:10])).reshape(1,-1), np.array(list(pre_top20[:10])).reshape(1,-1))
        ndcg_score_top15 += ndcg_score(np.array(list(gt_top20[:15])).reshape(1,-1), np.array(list(pre_top20[:15])).reshape(1,-1))
        ndcg_score_top20 += ndcg_score(np.array(list(gt_top20[:20])).reshape(1,-1), np.array(list(pre_top20[:20])).reshape(1,-1))
        
        
        # back testing on top1
        real_ret_rat_top1 = 0
        pre = pre_top20[0]
        real_ret_rat_top1 += ground_truth[pre][i]
        # 累计收益率计算公式
        bt_long1 *= (1+real_ret_rat_top1)
        sharpe_li1.append(real_ret_rat_top1)
        irr1.append(bt_long1)

        # back testing on top 5
        real_ret_rat_top5 = 0
        pre_top5 = pre_top20[:5]
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        # 累计收益率计算公式
        bt_long5 *= (1+real_ret_rat_top5)
        sharpe_li5.append(real_ret_rat_top5)
        irr5.append(bt_long5)
        
        # back testing on top 10
        real_ret_rat_top10 = 0
        pre_top10 = pre_top20[:10]
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        # 累计收益率计算公式
        bt_long10 *= (1+real_ret_rat_top10)
        sharpe_li10.append(real_ret_rat_top10)
        
        # back testing on top 15
        real_ret_rat_top15 = 0
        pre_top15 = pre_top20[:15]
        for pre in pre_top15:
            real_ret_rat_top15 += ground_truth[pre][i]
        real_ret_rat_top15 /= 15
        # 累计收益率计算公式
        bt_long15 *= (1+real_ret_rat_top15)
        sharpe_li15.append(real_ret_rat_top15)
        
        # back testing on top 20
        real_ret_rat_top20 = 0
        for pre in pre_top20:
            real_ret_rat_top20 += ground_truth[pre][i]
        real_ret_rat_top20 /= 20
        # 累计收益率计算公式
        bt_long20 *= (1+real_ret_rat_top20)
        sharpe_li20.append(real_ret_rat_top20)

    performance['btl1'] = bt_long1 - 1
    sharpe_li1 = np.array(sharpe_li1)
    performance['sharpe1'] = (np.mean(sharpe_li1)/np.std(sharpe_li1))*15.87 #To annualize
    
    performance['btl5'] = bt_long5 - 1
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = (np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87 #To annualize
    performance['ndcg_score_top5'] = ndcg_score_top5/prediction.shape[1]
    
    performance['btl10'] = bt_long10 - 1
    sharpe_li10 = np.array(sharpe_li10)
    performance['sharpe10'] = (np.mean(sharpe_li10)/np.std(sharpe_li10))*15.87 #To annualize
    performance['ndcg_score_top10'] = ndcg_score_top10/prediction.shape[1]
    
    performance['btl15'] = bt_long15 - 1
    sharpe_li15 = np.array(sharpe_li15)
    performance['sharpe15'] = (np.mean(sharpe_li15)/np.std(sharpe_li15))*15.87 #To annualize
    performance['ndcg_score_top15'] = ndcg_score_top15/prediction.shape[1]
    
    performance['btl20'] = bt_long20 - 1
    sharpe_li20 = np.array(sharpe_li20)
    performance['sharpe20'] = (np.mean(sharpe_li20)/np.std(sharpe_li20))*15.87 #To annualize
    performance['ndcg_score_top20'] = ndcg_score_top20/prediction.shape[1]

    # 返回技术指标[mse,ndcg_score_top5,btl5(累计收益率),sharpe5(夏普比率)]
    # irr: 收益率序列
    # selected_stock5: 选股序列
    return performance,irr1,irr5,selected_stock5
