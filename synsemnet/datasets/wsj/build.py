import sys
import os
import re
import argparse

from synsemnet.tree import Tree

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
    argparser.add_argument('-o', '--outdir', default='./wsj/', help='')
    args = argparser.parse_args()

    if not os.path.exists('tree2labels'):
        sys.stderr.write('Cloning sequence labeling code from Gomez-Rodriguez & Vilares 2018...\n')
        sys.stderr.flush()
        os.system(r'git clone https://github.com/aghie/tree2labels.git')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.outdir + '/trees'):
        os.makedirs(args.outdir + '/trees')
    if not os.path.exists(args.outdir + '/labels'):
        os.makedirs(args.outdir + '/labels')

    # Initialize tree instance
    t = Tree()

    # Build training set
    sys.stderr.write('Post-processing training set trees...\n')
    sys.stderr.flush()
    training_set = []
    for d in ['%02d' % i for i in range(1, 22)]:
        dir_path = args.dir_path + '/parsed/mrg/wsj/%s' % d
        for p in sorted(os.listdir(dir_path)):
            new_trees = clean_up_trees(dir_path + '/' + p)
            training_set += new_trees

    # Build dev set
    sys.stderr.write('Post-processing dev set trees...\n')
    sys.stderr.flush()
    dev_set = []
    d = '22'
    dir_path = args.dir_path + '/parsed/mrg/wsj/%s' % d
    for p in sorted(os.listdir(dir_path)):
        new_trees = clean_up_trees(dir_path + '/' + p)
        dev_set += new_trees

    # Build test set
    sys.stderr.write('Post-processing test set trees...\n')
    sys.stderr.flush()
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

    sys.stderr.write('Converting trees into label sequence...\n')
    sys.stderr.flush()
    exit_status = os.system(r'python2 tree2labels/dataset.py --train %s --dev %s --test %s --output %s --treebank wsj --os' % (args.outdir + '/trees/wsj-train.txt', args.outdir + '/trees/wsj-dev.txt', args.outdir + '/trees/wsj-test.txt', args.outdir + '/labels'))

    if exit_status == 0:
        sys.stderr.write('Data build complete. Labels for training can be found in %s.\n' % (args.outdir + '/labels/'))
    else:
        sys.stderr.write('Data build failed. See traceback for details.\n')

