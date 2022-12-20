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
from func import FedAvg, Clip
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

def train(args, model, train_loader, optimizer, epoch, epsilonPerClient, clientIdx, globalModel):
    model.train()

    originalModel = copy.deepcopy(globalModel)
    originalDict = originalModel.state_dict()
    # priv = RDPAccountant()

    losses = []
    loss_func = nn.CrossEntropyLoss()

    model.load_state_dict(originalDict)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % (args.log_interval * 10) == 0:
            tmp = round(loss.item(), 3)
            losses.append(tmp)

    finalModel = copy.deepcopy(model)
    finalDict = finalModel.state_dict()

    for key in originalDict.keys():
        finalDict[key] = finalDict[key] - originalDict[key]

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
    
    epsilonPerClient = args.epsilonInTotal/args.numOfClients
    train_loader_list, test_loader = retMnist(args.numOfClients)
    modelList, optimizerList, stateDictList, gradList = [], [], [], []
    for clientIdx in range(args.numOfClients):
        modelList.append(
            Net().to(device)
        )
        optimizerList.append(
            torch.optim.SGD(modelList[clientIdx].parameters(), lr=args.lr)
        )

    # engine = PrivacyEngine()
    # model, optimizer, train_loader = engine.make_private(
    #     module=model, 
    #     optimizer=optimizer, 
    #     data_loader=train_loader, 
    #     noise_multiplier=1, 
    #     max_grad_norm=1, 
    # )

    selectedClients = max(1, int(args.numOfClients * args.frac))

    globalModel = Net().to(device)
    globalDict = globalModel.state_dict()

    for epoch in range(1, args.epochs + 1):
        for clientIdx in range(args.numOfClients):
            grad = train(
                args, 
                modelList[clientIdx], 
                train_loader_list[clientIdx], 
                optimizerList[clientIdx], 
                epoch, 
                epsilonPerClient, 
                clientIdx, 
                globalModel
            )
            test(
                modelList[clientIdx], 
                test_loader, 
                clientIdx
            )
            gradList.append(grad)

        Clip(gradList)

        if epoch == 1:
            for clientIdx in range(args.numOfClients):
                stateDictList.append(modelList[clientIdx].state_dict())
            wAvg = FedAvg(stateDictList)
            globalModel.load_state_dict(wAvg)
        else:
            gradAvg = FedAvg(gradList)
            for key in gradAvg.keys():
                globalDict[key] = globalDict[key] + gradAvg[key]
            globalModel.load_state_dict(globalDict)
            gradList = []

        # print("ATTENTION________________global test")
        # test(globalModel, test_loader, "server")

if __name__ == '__main__':
    main()