# -*- encoding: utf-8 -*-

import os
import numpy as np
import json
import math
import argparse

def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value

def load_file(args):
    with open(args.label, 'r') as f:
        avglist = json.load(f)

    with open(os.path.join(args.pred, 'bert.json'), 'r') as f:
        blines = f.readlines()
    bertdics = [eval(blines[0]),eval(blines[1]),eval(blines[2]),eval(blines[3])]

    with open(os.path.join(args.pred, 'combined_top100.json'), 'r') as f:
        combdic = json.load(f)
    
    with open(os.path.join(args.pred, 'tfidf_top100.json'), 'r') as f:
        tdic = json.load(f)
    
    with open(os.path.join(args.pred, 'lm_top100.json'), 'r') as f:
        ldic = json.load(f)
    
    with open(os.path.join(args.pred, 'bm25_top100.json'), 'r') as f:
        bdic = json.load(f)

    with open(os.path.join(args.pred, 'res.json'), 'r') as f:
        longformer_dic=json.load(f)


    for key in list(combdic.keys())[:100]:
        tdic[key].reverse()
        bdic[key].reverse()

    
    return avglist, bertdics, combdic, tdic, ldic, bdic, longformer_dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help info:")
    parser.add_argument('--m', type=str, choices= ['NDCG', 'P', 'MAP'], default='NDCG', help='Metric.')
    parser.add_argument('--label', type=str, default='data/label/label_top30_dict.json', help='Label file path.')
    parser.add_argument('--pred', type=str, default='data/prediction', help='Prediction dir path.')
    parser.add_argument('--q', type=str, choices= ['all', 'common', 'controversial', 'test'], default='all', help='query set')

    args = parser.parse_args()

    avglist, bertdics, combdic, tdic, ldic, bdic, longformer_dic = load_file(args) #, lawformer_dic

    dics = [bdic, tdic, ldic]
    if args.q == 'all':
        keys = list(combdic.keys())[:100]    
    elif args.q == 'common':
        keys = list(combdic.keys())[:77]  
    elif args.q == 'controversial':
        keys = list(combdic.keys())[77:100]
    elif args.q == 'test':
        keys = [i for i in list(combdic.keys())[:100] if list(combdic.keys())[:100].index(i) % 5 == 0]
        dics=[longformer_dic]

    if args.m == 'NDCG':
        topK_list = [10, 20, 30]

        ndcg_list = []
        for topK in topK_list:
            temK_list = []
            for redic in dics:
                sndcg = 0.0
                for key in keys:
                    rawranks = []
                    for i in redic[key]:
                        if str(i) in avglist[key]:
                            rawranks.append(avglist[key][str(i)])
                        else:
                            rawranks.append(0)
                    ranks = rawranks + [0]*(30-len(rawranks))
                    if sum(ranks) != 0:
                        sndcg += ndcg(ranks, list(avglist[key].values()), topK)
                temK_list.append(sndcg/len(keys))
            ndcg_list.append(temK_list)
        print(ndcg_list)

    elif args.m == 'P': 
        topK_list = [5, 10]
        sp_list = []

        for topK in topK_list:
            temK_list = []
            for rdic in dics:
                sp = 0.0
                for key in keys:
                    ranks = [i for i in rdic[key] if i in list(combdic[key][:30])] 
                    # sp += float(len([j for j in ranks[:topK] if avglist[key][list(combdic[key][:30]).index(j)] == 1])/topK)
                    sp += float(len([j for j in ranks[:topK] if avglist[key][str(j)] == 3])/topK)
                temK_list.append(sp/len(keys))
            sp_list.append(temK_list)
        print(sp_list)

    elif args.m == 'MAP':
        map_list = []
        for rdic in dics:
            smap = 0.0
            for key in keys:
                ranks = [i for i in rdic[key] if i in list(combdic[key][:30])] 
                # rels = [ranks.index(i) for i in ranks if avglist[key][list(combdic[key][:30]).index(i)] == 1]
                rels = [ranks.index(i) for i in ranks if avglist[key][str(i)] == 3]
                tem_map = 0.0
                for rel_rank in rels:
                    # tem_map += float(len([j for j in ranks[:rel_rank+1] if avglist[key][list(combdic[key][:30]).index(j)] == 1])/(rel_rank+1))
                    tem_map += float(len([j for j in ranks[:rel_rank+1] if avglist[key][str(j)] == 3])/(rel_rank+1))
                if len(rels) > 0:
                    smap += tem_map / len(rels)
            map_list.append(smap/len(keys))
        print(map_list)