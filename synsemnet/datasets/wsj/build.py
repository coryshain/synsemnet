import sys
import os
import re
import argparse

from ...tree import Tree
from ...util import stderr


def clean_up_trees(path):
    t = Tree()
    n_left = 0
    n_right = 0
    trees = []
    tree_cur = ''
    with open(path, 'r') as f:
        for l in f:
            line = l.strip()
            if line:
                tree_cur += line + ' '
                n_left += line.count('(')
                n_right += line.count(')')
                if tree_cur and n_left == n_right:
                    tree_cur = re.sub(r'^\s*\((?:TOP)?\s*\((.*)\s*\)\s*\)\s*$', r'(\1)', tree_cur)
                    tree_cur = re.sub(' *\)', ')', tree_cur)
                    tree_cur = re.sub(' +', ' ', tree_cur)
                    t.read(tree_cur)
                    t.remove_traces()
                    t.remove_subcats()
                    t.collapse_unary()
                    tree_cur = str(t)
                    trees.append(tree_cur)
                    tree_cur = ''
                    n_left = 0
                    n_right = 0

    return trees


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Builds WSJ dataset into appropriate format for use with SynSemNet.
    ''')
    argparser.add_argument('dir_path', help='Path to Penn Treebank source directory')
    argparser.add_argument('-o', '--outdir', default='./wsj/', help='Path to output directory.')
    argparser.add_argument('-s', '--os', action='store_true', help='Whether to added sequence boundary tokens (-BOS-, -EOS-) to sentences.')
    argparser.add_argument('-r', '--root', action='store_true', help='Whether to use a designated root token.')
    args = argparser.parse_args()

    if not os.path.exists('tree2labels'):
        stderr('Cloning sequence labeling code from Gomez-Rodriguez & Vilares 2018...\n')
        os.system(r'git clone https://github.com/aghie/tree2labels.git')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.outdir + '/trees'):
        os.makedirs(args.outdir + '/trees')
    if args.os:
        label_dir = args.outdir + '/labels_os'
    else:
        label_dir = args.outdir + '/labels'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Initialize tree instance
    t = Tree()

    # Build training set
    stderr('Post-processing training set trees...\n')
    training_set = []
    for d in ['%02d' % i for i in range(1, 22)]:
        dir_path = args.dir_path + '/parsed/mrg/wsj/%s' % d
        for p in sorted(os.listdir(dir_path)):
            new_trees = clean_up_trees(dir_path + '/' + p)
            training_set += new_trees

    # Build dev set
    stderr('Post-processing dev set trees...\n')
    dev_set = []
    d = '22'
    dir_path = args.dir_path + '/parsed/mrg/wsj/%s' % d
    for p in sorted(os.listdir(dir_path)):
        new_trees = clean_up_trees(dir_path + '/' + p)
        dev_set += new_trees

    # Build test set
    stderr('Post-processing test set trees...\n')
    test_set = []
    d = '23'
    dir_path = args.dir_path + '/parsed/mrg/wsj/%s' % d
    for p in sorted(os.listdir(dir_path)):
        new_trees = clean_up_trees(dir_path + '/' + p)

        test_set += new_trees

    for i in range(len(test_set)):
        s = t.read(test_set[i])

    with open(args.outdir + '/trees/wsj-train.txt', 'w') as f:
        f.write('\n'.join(training_set))

    with open(args.outdir + '/trees/wsj-dev.txt', 'w') as f:
        f.write('\n'.join(dev_set))

    with open(args.outdir + '/trees/wsj-test.txt', 'w') as f:
        f.write('\n'.join(test_set))

    stderr('Converting trees into label sequence...\n')

    out_path = args.outdir + '/labels'
    train_path = args.outdir + '/trees/wsj-train.txt'
    dev_path = args.outdir + '/trees/wsj-dev.txt'
    test_path = args.outdir + '/trees/wsj-test.txt'

    if args.os:
        seq_bounds = '--os'
        out_path += '_os'
    else:
        seq_bounds = ''
    if args.root:
        root = '--root_label'
        out_path += '_root'
    else:
        root = ''

    exit_status = os.system(r'python -m tree2labels.dataset --train %s --dev %s --test %s --output %s --treebank wsj %s %s' % (train_path, dev_path, test_path, out_path, seq_bounds, root))

    if exit_status == 0:
        stderr('Data build complete. Labels for training can be found in %s.\n' % label_dir)
    else:
        stderr('Data build failed. See traceback for details.\n')

