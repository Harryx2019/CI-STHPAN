import os
import json
import numpy as np

def build_wiki_relation(data_path,market_name, connection_file, tic_wiki_file, sel_path_file):
    # readin tickers
    tickers = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',', skip_header=False)
    print('#tickers selected:', tickers.shape)
    
    ticind_wikiid_dic = {}
    for ind, tw in enumerate(tickers):
        if not tw[-1] == 'unknown':
            ticind_wikiid_dic[ind] = tw[-1]
    print('#tickers aligned:', len(ticind_wikiid_dic))

    # readin selected paths/connections
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ', skip_header=False)
    print('#paths selected:', len(sel_paths))
    sel_paths = set(sel_paths[:, 0])

    # readin connections
    with open(connection_file, 'r') as fin:
        connections = json.load(fin)
    print('#connection items:', len(connections))

    # get occured paths
    occur_first_paths = set()
    occur_second_paths = set()
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                if(len(p) == 1):
                    if(p[0] in sel_paths):
                        print('{}-{}-{}'.format(sou_item,tar_item,p[0]))
                        occur_first_paths.add(p[0])
                else:
                    path_key = '_'.join(p)
                    if path_key in sel_paths:
                        occur_second_paths.add(path_key)
    # generate
    valid_first_path_index = {}
    for ind, path in enumerate(occur_first_paths):
        valid_first_path_index[path] = ind
    print('#valid first paths:', len(valid_first_path_index))
    for path, ind in valid_first_path_index.items():
        print(path, ind)
    
    valid_second_path_index = {}
    for ind, path in enumerate(occur_second_paths):
        valid_second_path_index[path] = ind
    print('#valid second paths:', len(valid_second_path_index))
    for path, ind in valid_second_path_index.items():
        print(path, ind)
    
    wiki_first_relation = np.zeros([len(valid_first_path_index),tickers.shape[0],tickers.shape[0]],dtype=int)  
    second_num = 0
    for i in range(len(tickers)):
        if(i in ticind_wikiid_dic.keys()):
            for j in range(len(tickers)):
                if(j in ticind_wikiid_dic.keys()):
                    sou_item = ticind_wikiid_dic[i]
                    tar_item = ticind_wikiid_dic[j]
                    if(sou_item in connections.keys() and tar_item in connections[sou_item].keys()):
                        connections_ij = connections[sou_item][tar_item]
                        for p in connections_ij:
                            path_key = '_'.join(p)
                            if path_key in valid_first_path_index.keys():
                                ccc = valid_first_path_index[path_key]
                                wiki_first_relation[ccc][i][j] = 1
                                print('{}-{}-{}'.format(i,j,path_key))
                            elif(i<=j and path_key in valid_second_path_index.keys()):
                                second_num += 1
    
    # first type
    first_num = 0
    for i in range(wiki_first_relation.shape[0]):
        for j in range(wiki_first_relation.shape[1]):
            if(wiki_first_relation[i][j].sum()!=0):
                first_num += 1
                
    wiki_first_relation_embedding = np.zeros([tickers.shape[0], first_num],dtype=int)
    conn_count = 0
    hyperedge_index = 0
    for i in range(wiki_first_relation.shape[0]):
        for j in range(wiki_first_relation.shape[1]):
            if(wiki_first_relation[i][j].sum()!=0):
                for k in range(wiki_first_relation.shape[2]):
                    if(wiki_first_relation[i][j][k]==1):
                        wiki_first_relation_embedding[j][hyperedge_index] = 1
                        wiki_first_relation_embedding[k][hyperedge_index] = 1
                        conn_count += 2
                hyperedge_index += 1
    print('first connections count:', conn_count, 'ratio:', conn_count / float(tickers.shape[0] * first_num))
    
    # second type
    wiki_second_relation_embedding = np.zeros([tickers.shape[0], second_num],dtype=int)
    conn_count = 0
    hyperedge_index = 0
    for i in range(len(tickers)):
        if(i in ticind_wikiid_dic.keys()):
            for j in range(i,len(tickers)):
                if(j in ticind_wikiid_dic.keys()):
                    sou_item = ticind_wikiid_dic[i]
                    tar_item = ticind_wikiid_dic[j]
                    if(sou_item in connections.keys() and tar_item in connections[sou_item].keys()):
                        connections_ij = connections[sou_item][tar_item]
                        for p in connections_ij:
                            path_key = '_'.join(p)
                            if path_key in valid_second_path_index.keys():
                                wiki_second_relation_embedding[i][hyperedge_index] = 1
                                wiki_second_relation_embedding[j][hyperedge_index] = 1
                                hyperedge_index += 1
                                conn_count += 2
    print('second connections count:', conn_count, 'ratio:', conn_count / float(tickers.shape[0] * second_num))

    print(wiki_first_relation_embedding.shape)
    print(wiki_second_relation_embedding.shape)
    np.save(os.path.join(data_path,market_name + '_wiki_first_relation'), wiki_first_relation_embedding)
    np.save(os.path.join(data_path,market_name + '_wiki_second_relation'), wiki_second_relation_embedding)


data_path = '../dataset/stock/relation/wikidata'
# market_name = 'NASDAQ'
market_name = 'NYSE'
connection_file = os.path.join(data_path, market_name+'_connections.json')
tic_wiki_file = os.path.join(data_path, '..','..', market_name+'_wiki.csv')
sel_path_file = os.path.join(data_path, 'selected_wiki_connections.csv')

build_wiki_relation(data_path, market_name, connection_file, tic_wiki_file, sel_path_file)
