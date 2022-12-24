import copy
import torch
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


def median(listIn):
    listIn.sort()
    l = len(listIn)
    if(l % 2) == 1:
        return listIn[int(l / 2)]
    else:
        return (listIn[int(l/2)] + listIn[int(l/2)-1])/2


def Clip(gradList):
    normList, sensitivityList = [], []
    # generate sensitivity list
    # sensitivity = median(norm)
    for key in gradList[0].keys():
        for grad in gradList:
            normList.append(grad[key].norm())
        sensitivityList.append(median(normList))
        normList = []
    for key, idx in zip(gradList[0].keys(), range(len(sensitivityList))):
        for grad in gradList:
            S = sensitivityList[idx]
            bound = max(1, grad[key].norm()/S)
            grad[key] = grad[key]/bound
    print(sensitivityList)
    return sensitivityList, gradList
