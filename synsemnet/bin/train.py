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

    if not args.preprocess and os.path.exists('saved_data.obj'):
        with open('saved_data.obj', 'rb') as f:
            stderr('Loading data...\n')
            train_data = pickle.load(f)
    else:
        stderr('Reading and processing data...\n')
        train_data = Dataset(p.parsing_train_data_path, p.sts_train_data_path)
        train_data.cache_processed_data()
        with open('saved_data.obj', 'wb') as f:
            pickle.dump(train_data, f)

    x, mask = train_data.cache['syn_text'], train_data.cache['syn_text_mask']

    m = SynSemNet(
        train_data.get_charset(),
        train_data.get_pos_label_set(),
        train_data.get_parse_label_set(),
        **kwargs
    )

