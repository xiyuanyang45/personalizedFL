import torchvision as tv
import torch
from options import args_parser
from torch.utils.data import Subset
args = args_parser()

def retMnist(numOfClients):

    # numOfClients = 3

    transform=tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # setup data
    dataset_train = tv.datasets.MNIST(
        '../data/mnist', 
        train=True, 
        download=True,
        transform=transform
    )
    dataset_test = tv.datasets.MNIST(
        '../data/mnist', 
        train=False, 
        transform=transform
    )

    trainSetList = []
    for i in range(numOfClients):
        trainSetList.append(
            Subset(
                dataset_train, range(
                    int(i*60000/numOfClients), int((i+1)*60000/numOfClients)
        )))

    # set up dataloader
    trainLoaderList = []
    for subset in trainSetList:
        trainLoaderList.append(torch.utils.data.DataLoader(
            subset, 
            batch_size=args.batch_size, 
            num_workers=0, 
            pin_memory=True, 
            shuffle=True
        ))
    test_loader = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.test_batch_size, 
        num_workers=0, 
        pin_memory=True, 
        shuffle=True
    )

    return trainLoaderList, test_loader