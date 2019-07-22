import sys
import os
import pickle
import argparse

from synsemnet.config import Config
from synsemnet.kwargs import SYN_SEM_NET_KWARGS
from synsemnet.data import Dataset
from synsemnet.model import SynSemNet
from synsemnet.util import stderr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a SynSemNet model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
    argparser.add_argument('-c', '--force_cpu', action='store_true', help='Do not use GPU. If not specified, GPU usage defaults to the value of the **use_gpu_if_available** configuration parameter.')
    args = argparser.parse_args()

    p = Config(args.config)

    if args.force_cpu or not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    kwargs = {}

    for kwarg in SYN_SEM_NET_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    data_path = 'data'
    if p['os']:
        data_path += '_os'
    if p['root']:
        data_path += '_root'
    data_path += '.obj'

    if not args.preprocess and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            stderr('Loading data...\n')
            train_data = pickle.load(f)
    else:
        stderr('Reading and processing data...\n')
        train_data = Dataset(p.parsing_train_data_path, p.sts_train_data_path)
        train_data.cache_data(factor_parse_labels=p['factor_parse_labels'])
        with open(data_path, 'wb') as f:
            pickle.dump(train_data, f)

    char_set = train_data.get_char_set()
    pos_label_set = train_data.get_pos_label_set()
    if p['factor_parse_labels']:
        parse_label_set = train_data.get_parse_ancestor_set()
    else:
        parse_label_set = train_data.get_parse_label_set()

    m = SynSemNet(
        char_set,
        pos_label_set,
        parse_label_set,
        **kwargs
    )

    m.fit(train_data, 1000)

