import argparse

def args_parser():
# Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 14)')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gpu', type=int, default=0, 
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--numOfClients', type=int, default=4)
    parser.add_argument('--frac', type=float, default=0.4)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--sigma', type=float, default=0.001)

    args = parser.parse_args()

    return args