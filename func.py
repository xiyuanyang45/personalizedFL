import copy
import torch
import numpy as np
from options import args_parser
args = args_parser()

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    # print(w_avg.keys())
    for k in w_avg.keys():
        for i in range(1, args.numOfClients):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], args.numOfClients)

    # for x in w_avg.values():
    #     print('-----------------------------------------------------------')
    #     print(type(x))
    #     print(f'norm:{x.norm()}')
    return w_avg

def Clip(gradList):
    normList, sensitivityList = [], []
    for key in gradList[0].keys():
        for grad in gradList:
            normList.append(grad[key].norm())
        sensitivityList.append(np.median(normList))
        normList = []
    

