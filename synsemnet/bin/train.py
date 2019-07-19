import sys
import os
import argparse

from synsemnet.config import Config
from synsemnet.kwargs import SYN_SEM_NET_KWARGS
from synsemnet.data import Dataset
from synsemnet.model import SynSemNet

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a SynSemNet model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-c', '--force_cpu', action='store_true', help='Do not use GPU. If not specified, GPU usage defaults to the value of the **use_gpu_if_available** configuration parameter.')
    args = argparser.parse_args()

    p = Config(args.config)

    if args.force_cpu or not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    kwargs = {}

    for kwarg in SYN_SEM_NET_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    train_data = Dataset(p.parsing_train_data_path, p.sts_train_data_path)

    m = SynSemNet(train_data.get_vocabulary(), train_data.get_charset(), **kwargs)

