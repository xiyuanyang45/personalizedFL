import copy
import torch
from options import args_parser
args = args_parser()

# def FedAvg(stateDictList):
#     wAvg = copy.deepcopy(stateDictList[0])

#     for wIdx in wAvg.keys():
#         for clientIdx in range(1, args.numOfClients):
#             print(wAvg.keys())
#             print(range(1, args.numOfClients))
#             wAvg[wIdx] += (stateDictList[clientIdx])[wIdx]

#         wAvg[wIdx] = torch.div(wAvg[wIdx], args.numOfClients)
#     return wAvg

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    print(w_avg.keys())
    for k in w_avg.keys():
        for i in range(1, args.numOfClients):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], args.numOfClients)

    print(w_avg.keys())
    for x in w_avg.values():
        print('-----------------------------------------------------------')
        print(type(x))
        print(f'norm:{x.norm()}')
    return w_avg
