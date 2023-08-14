import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from datetime import datetime

data_path = '../dataset/stock'
# market_name = 'NASDAQ'
market_name = 'NYSE'

selected_tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'

# read tickers' name
tickers = np.genfromtxt(
    os.path.join(data_path, selected_tickers_fname),
    dtype=str, delimiter='\t', skip_header=False
)
print('#tickers selected:', len(tickers))

eod_data = []
masks = []
ground_truth = []
base_price = []
data_time_stamp = []

steps = 1
# seq_len = 336
seq_len = 512
target = 'Close'
features = 'MS'
# train:0 valid:1 test:2
# set_type = 0
set_type = 1
timeenc = 0


valid_index = 756
test_index = 1008
trade_dates = 1245

border1s = [0, valid_index - seq_len, test_index - seq_len]
border2s = [valid_index, test_index, trade_dates]
border1 = border1s[set_type]
border2 = border2s[set_type]

# read tickers' eod data
for index, ticker in enumerate(tickers):
    '''
    df_raw.columns: ['date', ...(other features), target feature]
    '''
    df_raw = pd.read_csv(os.path.join(data_path, '2013-01-01', market_name + '_' + ticker + '_1.csv'))

    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + [target]]
    # print(cols)
    if market_name == 'NASDAQ':
        # remove the last day since lots of missing data
        df_raw = df_raw[:-1]

    if features == 'M' or features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    elif features == 'S':
        df_data = df_raw[[target]]
    data = df_data.values
    data = data[border1:border2]

    df_stamp = df_raw[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], axis = 1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

    if index == 0:
        print('#single EOD data shape:', data.shape)
        # [股票数*交易日数*5[5-day,10-day,20-day,30-day,Close]]
        eod_data = np.zeros([len(tickers), data.shape[0],data.shape[1]], dtype=np.float32)
        masks = np.ones([len(tickers), data.shape[0]],dtype=np.float32)
        ground_truth = np.zeros([len(tickers), data.shape[0]],dtype=np.float32)
        base_price = np.zeros([len(tickers), data.shape[0]],dtype=np.float32)

    for row in range(data.shape[0]):
        if abs(data[row][-1] + 1234) < 1e-8:
            masks[index][row] = 0.0
        elif row > steps - 1 and abs(data[row - steps][-1] + 1234) > 1e-8:
            ground_truth[index][row] = (data[row][-1] - data[row - steps][-1]) / data[row - steps][-1]
        for col in range(data.shape[1]):
            if abs(data[row][col] + 1234) < 1e-8:
                data[row][col] = 1.0 # 空值处理
    eod_data[index, :, :] = data
    base_price[index, :] = data[:, -1]
    data_time_stamp.append(data_stamp)
data_stamp = np.array(data_time_stamp)
print('#eod_data shape:', eod_data.shape)
print('#masks shape:', masks.shape)
print('#ground_truth shape:', ground_truth.shape)
print('#base_price shape:', base_price.shape)
print('#data_stamp shape:', data_stamp.shape)



starttime = datetime.now()
fastdtw_ij = np.zeros([5, len(tickers), len(tickers)], dtype=np.float32)

for i in range(len(tickers)):
    X = eod_data[i]
    for j in range(i,len(tickers)):
        Y = eod_data[j]
        for m in range(5):
            x = X[:,m].reshape(1,-1)
            y = Y[:,m].reshape(1,-1)
            dtw,path = fastdtw(x,y,dist=euclidean)

            fastdtw_ij[m][i][j] = dtw
            fastdtw_ij[m][j][i] = dtw
    
    endtime = datetime.now()
    print((endtime - starttime).seconds)

    # np.save(os.path.join(data_path,'relation/',market_name + '_fastdtw_train_sup'), fastdtw_ij)
    # np.save(os.path.join(data_path,'relation/',market_name + '_fastdtw_valid_sup'), fastdtw_ij)

    # np.save(os.path.join(data_path,'relation/',market_name + '_fastdtw_train_fine'), fastdtw_ij)
    np.save(os.path.join(data_path,'relation/',market_name + '_fastdtw_valid_fine'), fastdtw_ij)

