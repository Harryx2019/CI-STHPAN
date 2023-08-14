from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Stock, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
import os
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'stock': Dataset_Stock
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    tickers_fname = args.market + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    print('#data_provider tickers_fname:',tickers_fname)

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        market_name = args.market,
        tickers_fname = tickers_fname,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    # drop_last：Whether to discard the last batch of data 
    # when the sample size is not divisible by the batchsize
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    # load relation data
    hyperedge_index = None
    if args.graph:
        # all relation
        if args.rel_type == 0:
            # channel independent
            if args.ci == 1:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/all',
                                                        args.market + '_all_relation_train_sup.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/all',
                                                        args.market + '_all_relation_valid_sup.npy'))
            # channel mixing
            else:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/all',
                                                        args.market + '_all_relation_train_mix.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/all',
                                                        args.market + '_all_relation_valid_mix.npy'))
        # industry relation
        elif args.rel_type == 1:
            hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/industry',
                                                   args.market + '_industry_relation.npy'))
            # channel independent
            if args.ci == 1:
                num_stocks = hyperedge_index.shape[0]
                num_edges = hyperedge_index.shape[1]
                industry_relation = np.zeros([args.enc_in,num_stocks,num_edges],dtype=int)
                for m in range(args.enc_in):
                    industry_relation[m] = hyperedge_index
                hyperedge_index = industry_relation
        # wiki relation
        elif args.rel_type == 2:
            hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/wikidata',
                                                   args.market + '_wiki_relation.npy'))
            # channel independent
            if args.ci == 1:
                num_stocks = hyperedge_index.shape[0]
                num_edges = hyperedge_index.shape[1]
                industry_relation = np.zeros([args.enc_in,num_stocks,num_edges],dtype=int)
                for m in range(args.enc_in):
                    industry_relation[m] = hyperedge_index
                hyperedge_index = industry_relation
        # dtw relation
        elif args.rel_type == 3:
            # channel independent
            if args.ci == 1:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/dtw',
                                                        args.market + '_dtw_train_sup_top'+str(args.k)+'.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/dtw',
                                                        args.market + '_dtw_valid_sup_top'+str(args.k)+'.npy'))
            # channel mixing
            else:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/dtw',
                                                        args.market + '_dtw_relation_train_top'+str(args.k)+'_mix.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/dtw',
                                                        args.market + '_dtw_relation_valid_top'+str(args.k)+'_mix.npy'))
        # industry_wiki relation
        elif args.rel_type == 4:
            hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/industry_wiki',
                                                   args.market + '_industry_wiki_relation.npy'))
            # channel independent
            if args.ci == 1:
                num_stocks = hyperedge_index.shape[0]
                num_edges = hyperedge_index.shape[1]
                industry_relation = np.zeros([args.enc_in,num_stocks,num_edges],dtype=int)
                for m in range(args.enc_in):
                    industry_relation[m] = hyperedge_index
                hyperedge_index = industry_relation
        # industry_dtw relation
        elif args.rel_type == 5:
            # channel independent
            if args.ci == 1:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/industry_dtw',
                                                        args.market + '_industry_dtw_relation_train_sup.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/industry_dtw',
                                                        args.market + '_industry_dtw_relation_valid_sup.npy'))
            # channel mixing
            else:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/industry_dtw',
                                                        args.market + '_industry_dtw_relation_train_mix.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/industry_dtw',
                                                        args.market + '_industry_dtw_relation_valid_mix.npy'))
        # wiki_dtw relation
        elif args.rel_type == 6:
            # channel independent
            if args.ci == 1:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/wiki_dtw',
                                                        args.market + '_wiki_dtw_relation_train_sup.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/wiki_dtw',
                                                        args.market + '_wiki_dtw_relation_valid_sup.npy'))
            # channel mixing
            else:
                if flag == 'train':
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/wiki_dtw',
                                                        args.market + '_wiki_dtw_relation_train_mix.npy'))
                else:
                    hyperedge_index = np.load(os.path.join(args.root_path, args.data_path, '..', 'relation/wiki_dtw',
                                                        args.market + '_wiki_dtw_relation_valid_mix.npy'))
        print('#hyperedge_index shape:',hyperedge_index.shape)
    return data_set, data_loader, hyperedge_index
