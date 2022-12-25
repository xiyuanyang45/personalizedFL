from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from tqdm import tqdm
from opacus import PrivacyEngine
import time
from dataSetup import retMnist
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from options import args_parser
from func import FedAvg, Clip, addGaussian
from net import Net
from opacus.accountants.rdp import RDPAccountant
import copy

args = args_parser()

# set device
if(torch.cuda.is_available()):
    # device = "cuda"
    device = 'cuda:{}'.format(args.gpu)
elif(torch.backends.mps.is_available()):
    device = "mps"
else:
    device = "cpu"


def personalizedTrain(args, model, train_loader, optimizer, epoch, clientIdx, globalModel, keyToAlign):

    globalModel = copy.deepcopy(globalModel)
    globalDict = globalModel.state_dict()
    # priv = RDPAccountant()

    # model.load_state_dict(originalDict)

    modelDict = model.state_dict()
    for key in keyToAlign:
        modelDict[key] = globalDict[key]
    model.load_state_dict(modelDict)

    model.train()
    losses = []
    loss_func = nn.CrossEntropyLoss()

    # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % (args.log_interval * 10) == 0:
            tmp = round(loss.item(), 3)
            losses.append(tmp)

    finalDict = model.state_dict()

    for key in globalDict.keys():
        finalDict[key] = finalDict[key] - globalDict[key]

    print(
        f"Client: {clientIdx} \n"
        f"Train Epoch: {epoch} \n"
        f"loss:{losses}"
    )

    return finalDict

def test(model, test_loader, clientIdx):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Client: {}\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        clientIdx, 
        test_loss, 
        correct, 
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

def main():
    
    train_loader_list, test_loader = retMnist(args.numOfClients)
    modelList, optimizerList, gradList = [], [], []

    keyList = []

    for clientIdx in range(args.numOfClients):
        modelList.append(
            Net().to(device)
        )
        optimizerList.append(
            torch.optim.SGD(modelList[clientIdx].parameters(), lr=args.lr)
        )

    globalModel = Net().to(device)
    globalDict = globalModel.state_dict()

    for key in globalDict.keys():
        keyList.append(key)
    keyToAlign = []
    for idx in range(2 * args.layers):
        key =  keyList[-1-idx]
        keyToAlign.append(key)
    print(f'key to align : {keyToAlign}')

    # for epoch in tqdm(range(1, args.epochs + 1)):
    for epoch in tqdm(range(1, args.epochs + 1)):
        print("------------------------------------------------------")
        for clientIdx in range(args.numOfClients):
            grad = personalizedTrain(
                args, 
                modelList[clientIdx], 
                train_loader_list[clientIdx], 
                optimizerList[clientIdx], 
                epoch, 
                clientIdx, 
                globalModel, 
                keyToAlign
            )
            test(
                modelList[clientIdx], 
                test_loader, 
                clientIdx
            )
            gradList.append(grad)

        sensitivityList, gradList = Clip(gradList)
        gradList = addGaussian(sensitivityList, gradList, args.sigma, device)
        gradAvg = FedAvg(gradList, keyToAlign)
        for key in gradAvg.keys():
            globalDict[key] = globalDict[key] + gradAvg[key]
        globalModel.load_state_dict(globalDict)
        gradList = []


if __name__ == '__main__':
    main()