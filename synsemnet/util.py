import sys
import pickle

def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

def pretty_print_seconds(s):
    s = int(s)
    h = s // 3600
    m = s % 3600 // 60
    s = s % 3600 % 60
    return '%02d:%02d:%02d' % (h, m, s)

def load_synsemnet(dir_path):
    """
    Convenience method for reconstructing a saved SynSemNet object. First loads in metadata from ``m.obj``, then uses
    that metadata to construct the computation graph. Then, if saved weights are found, these are loaded into the
    graph.

    :param dir_path: Path to directory containing the DTSR checkpoint files.
    :return: The loaded SynSemNet instance.
    """

    with open(dir_path + '/m.obj', 'rb') as f:
        m = pickle.load(f)
    m.build(outdir=dir_path)
    m.load(outdir=dir_path)
    return m