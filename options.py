import argparse

def args_parser():
# Training settings

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--epochs', type=int, default=2)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=1000)

    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0, )
    parser.add_argument('--numOfClients', type=int, default=4)
    parser.add_argument('--frac', type=float, default=0.4)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--sigma', type=float, default=0.001)
    parser.add_argument('--layers', type=int, default=2)

    args = parser.parse_args()

    return args