import sys
import os
import pickle
import argparse
import pdb

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
    argparser.add_argument('-P', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
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
            data = pickle.load(f)
    else:
        stderr('Reading and processing data...\n')
        data = Dataset(p.parsing_train_data_path, p.sts_train_data_path)
        data.initialize_parsing_file(p.parsing_dev_data_path, 'dev')
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    stderr('Caching numeric train data...\n')
    data.cache_numeric_parsing_data(name='train', factor_parse_labels=p['factor_parse_labels'])
    stderr('Caching numeric dev data...\n')
    data.cache_numeric_parsing_data(name='dev', factor_parse_labels=p['factor_parse_labels'])

    char_set = data.char_list
    pos_label_set = data.pos_label_list
    if p['factor_parse_labels']:
        parse_label_set = data.parse_ancestor_list
    else:
        parse_label_set = data.parse_label_list
    sts_label_set = data.sts_label_set

    m = SynSemNet(
        char_set,
        pos_label_set,
        parse_label_set,
        sts_label_set,
        **kwargs
    )

    m.fit(data, 1000)

