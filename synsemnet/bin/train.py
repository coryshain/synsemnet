import sys
import os
import pickle
import argparse
import pdb

from ..config import Config
from ..kwargs import SYN_SEM_NET_KWARGS
from ..data import Dataset
from ..model import SynSemNet
from ..util import stderr


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a SynSemNet model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-P', '--preprocess', action='store_true', help='IGNORED!')
    argparser.add_argument('-c', '--force_cpu', action='store_true', help='Do not use GPU. If not specified, GPU usage defaults to the value of the **use_gpu_if_available** configuration parameter.')
    args = argparser.parse_args()

    p = Config(args.config)

    if args.force_cpu or not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    kwargs = {}

    for kwarg in SYN_SEM_NET_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    stderr('Reading and processing data...\n')
    data = Dataset(p.parsing_train_data_path, p.sts_train_data_path, os=p['os'])
    data.initialize_parsing_file(p.parsing_dev_data_path, 'dev')
    data.initialize_sts_file(p.sts_dev_data_path, 'dev')

    stderr('Caching numeric parsing train data...\n')
    data.cache_numeric_parsing_data(
        name='train',
        factor_parse_labels=p['factor_parse_labels'],
        rel_depth=p['parsing_relative_depth'],
        clip_vocab=p['target_vocab_size'] - Dataset.N_SPECIAL
    )
    stderr('Caching numeric parsing dev data...\n')
    data.cache_numeric_parsing_data(
        name='dev',
        factor_parse_labels=p['factor_parse_labels'],
        rel_depth=p['parsing_relative_depth'],
        clip_vocab=p['target_vocab_size'] - Dataset.N_SPECIAL
    )

    stderr('Caching numeric STS train data...\n')
    data.cache_numeric_sts_data(
        name='train',
        clip_vocab=p['target_vocab_size'] - Dataset.N_SPECIAL
    )
    stderr('Caching numeric STS dev data...\n')
    data.cache_numeric_sts_data(
        name='dev',
        clip_vocab=p['target_vocab_size'] - Dataset.N_SPECIAL
    )

    char_set = data.char_list
    pos_label_set = data.pos_label_list
    if p['factor_parse_labels']:
        parse_label_set = data.parse_ancestor_list
    else:
        parse_label_set = data.parse_label_list
    parse_depth_set = data.parse_depth_rel_list
    sts_label_set = data.sts_label_set

    stderr('Initializing SynSemNet...\n')

    m = SynSemNet(
        char_set,
        pos_label_set,
        parse_label_set,
        parse_depth_set,
        sts_label_set,
        **kwargs
    )

    m.fit(
        data,
        1000000,
        train_gold_tree_path=p.parsing_train_gold_trees_path,
        dev_gold_tree_path=p.parsing_dev_gold_trees_path
    )

