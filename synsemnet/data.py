import math
import numpy as np

from synsemnet.util import stderr


def get_random_permutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv


def get_seq_shape(seq):
    seq_shape = []
    if isinstance(seq, list):
        seq_shape = [len(seq)]
        child_seq_shapes = [get_seq_shape(x) for x in seq]
        for i in range(len(child_seq_shapes[0])):
            seq_shape.append(max([x[i] for x in child_seq_shapes]))
    elif isinstance(seq, np.ndarray):
        seq_shape += seq.shape

    return seq_shape


def pad_sequence(sequence, seq_shape=None, dtype='float32', reverse=False, padding='pre', value=0.):
    assert padding in ['pre', 'post'], 'Padding type "%s" not recognized' % padding
    if seq_shape is None:
        seq_shape = get_seq_shape(sequence)

    if len(seq_shape) == 0:
        return sequence
    if isinstance(sequence, list):
        sequence = np.array(
            [pad_sequence(x, seq_shape=seq_shape[1:], dtype=dtype, reverse=reverse, padding=padding, value=value) for x
             in sequence])
        pad_width = seq_shape[0] - len(sequence)
        if padding == 'pre':
            pad_width = (pad_width, 0)
        elif padding == 'post':
            pad_width = (0, pad_width)
        if len(seq_shape) > 1:
            pad_width = [pad_width]
            pad_width += [(0, 0)] * (len(seq_shape) - 1)
        sequence = np.pad(
            sequence,
            pad_width=pad_width,
            mode='constant',
            constant_values=(value,)
        )
    elif isinstance(sequence, np.ndarray):
        pad_width = [
            (seq_shape[i] - sequence.shape[i], 0) if padding == 'pre' else (0, seq_shape[i] - sequence.shape[i]) for i
            in range(len(sequence.shape))]

        if reverse:
            sequence = sequence[::-1]

        sequence = np.pad(
            sequence,
            pad_width=pad_width,
            mode='constant',
            constant_values=(value,)
        )

    return sequence


class Dataset(object):
    def __init__(
            self,
            syn_path,
            sem_path
    ):
        syn_text, pos_labels, parse_labels = self.read_parse_label_file(syn_path)
        self.syn_text = syn_text
        self.pos_labels = pos_labels
        self.parse_labels = parse_labels

        self.sem_text = [[]]

        self.char_list = self.get_charset()
        self.word_list = self.get_vocabulary()
        self.pos_list = self.get_pos_label_set()
        self.parse_label_list = self.get_parse_label_set()

        self.char_map = {c: i for i, c in enumerate(self.char_list)}
        self.word_map = {w: i for i, w in enumerate(self.word_list)}
        self.pos_map = {p: i for i, p in enumerate(self.pos_list)}
        self.parse_label_map = {l: i for i, l in enumerate(self.parse_label_list)}

        self.n_char = len(self.char_map)
        self.n_word = len(self.word_map)
        self.n_pos = len(self.pos_map)
        self.n_parse_label = len(self.parse_label_map)

        self.cache = {}

    def read_parse_label_file(self, path):
        text = []
        pos_labels = []
        parse_labels = []
        text_cur = []
        pos_labels_cur = []
        parse_labels_cur = []
        with open(path, 'r') as f:
            for l in f:
                if l.strip() == '':
                    assert len(text_cur) == len(pos_labels_cur) == len(parse_labels_cur), 'Mismatched text and labels: [%s] vs. [%s] vs. [%s].' % (' '.join(text_cur), ' '.join(pos_labels_cur), ' '.join(parse_labels_cur))
                    text.append(text_cur)
                    pos_labels.append(pos_labels_cur)
                    parse_labels.append(parse_labels_cur)
                    text_cur = []
                    pos_labels_cur = []
                    parse_labels_cur = []
                else:
                    w, p, l = l.strip().split()
                    text_cur.append(w)
                    pos_labels_cur.append(p)
                    parse_labels_cur.append(l)


        return text, pos_labels, parse_labels

    def get_vocabulary(self):
        vocab = set()
        for s in self.syn_text + self.sem_text:
            for w in s:
                vocab.add(w)
        return sorted(list(vocab))

    def get_charset(self):
        charset = set()
        for s in self.syn_text + self.sem_text:
            for w in s:
                for c in w:
                    charset.add(c)
        return ['', ' '] + sorted(list(charset))

    def get_pos_label_set(self):
        pos_label_set = set()
        for s in self.pos_labels:
            for p in s:
                pos_label_set.add(p)
        return sorted(list(pos_label_set))

    def get_parse_label_set(self):
        parse_label_set = set()
        for s in self.parse_labels:
            for l in s:
                parse_label_set.add(l)
        return sorted(list(parse_label_set))

    def get_seqs(self, src='syn_text', as_words=True):
        if src.lower() == 'syn_text':
            data = self.syn_text
        elif src.lower() == 'sem_text':
            data = self.sem_text
        elif src.lower() == 'pos':
            assert as_words, "Character sequences for PoS tags don't make sense and aren't supported"
            data = self.pos_labels
        elif src.lower() == 'parse_label':
            assert as_words, "Character sequences for parse labels don't make sense and aren't supported"
            data = self.parse_labels
        else:
            raise ValueError('Unrecognized task "%s".' % src)

        if as_words:
            return data
        else:
            text = []
            for s in data:
                text.append(' '.join(s))
            return text

    def char2int(self, c):
        return self.char_map.get(c, self.n_char)

    def word2int(self, w):
        return self.word_map.get(w, self.n_word)

    def pos2int(self, p):
        return self.pos_map.get(p, self.n_pos)

    def parselabel2int(self, l):
        return self.parse_label_map.get(l, self.n_parse_label)

    def get_padded_seqs(self, data_type, max_token=None, max_subtoken=None, as_char=False, verbose=True):
        if data_type.lower().startswith('syn'):
            src = 'syn_text'
        elif data_type.lower().startswith('sem'):
            src = 'sem_text'
        else:
            src = None

        if data_type.lower().endswith('char_tokenized'):
            as_words = True
            if as_char:
                f = lambda x: [y[:max_subtoken] for y in x]
            else:
                f = lambda x: list(map(self.char2int, x[:max_subtoken]))
        elif data_type.lower().endswith('word'):
            as_words = True
            if as_char:
                f = lambda x: x
            else:
                f = self.word2int
        elif data_type.lower().endswith('char'):
            as_words = False
            if as_char:
                f = lambda x: x
            else:
                f = self.char2int
        else:
            as_words = True
            if data_type.lower().endswith('pos_label'):
                src = 'pos'
                if as_char:
                    f = lambda x: x
                else:
                    f = self.pos2int
            elif data_type.lower().endswith('parse_label'):
                src = 'parse_label'
                if as_char:
                    f = lambda x: x
                else:
                    f = self.parselabel2int
            else:
                raise ValueError('Unrecognized data_type "%s".' % data_type)

        data = self.get_seqs(src=src, as_words=as_words)

        out = []
        mask = []
        n = len(data)
        for i, s in enumerate(data):
            # if verbose and i == 0 or i % 1000 == 999 or i == n-1:
            #     stderr('\r%d/%d' %(i+1, n))
            newline = list(map(f, s))[:max_token]
            out.append(newline)
            if data_type.lower().endswith('char_tokenized'):
                mask.append([[1] * len(x) for x in newline])
            else:
                mask.append([1] * len(newline))

        # if verbose:
        #     stderr('\n')

        out = pad_sequence(out, value=0)
        if not as_char:
            out = out.astype(np.uint8)
        mask = pad_sequence(mask)

        return out, mask

    def cache_data(self):
        self.cache['syn_text'], self.cache['syn_text_mask'] = self.get_padded_seqs(
            'syn_text_char_tokenized',
            as_char=False
        )
        self.cache['pos_label'], _ = self.get_padded_seqs('pos_label')
        self.cache['parse_label'], _ = self.get_padded_seqs('parse_label')

    def get_data_feed(
            self,
            minibatch_size=128,
            randomize=False
    ):
        syn_text = self.cache['syn_text']
        syn_text_mask = self.cache['syn_text_mask']
        pos_label = self.cache['pos_label']
        parse_label = self.cache['parse_label']
        n = self.get_n()

        i = 0

        if randomize:
            ix, ix_inv = get_random_permutation(n)
        else:
            ix = np.arange(n)

        while i < n:
            indices = ix[i:i+minibatch_size]

            out = {
                'syn_text': syn_text[indices],
                'syn_text_mask': syn_text_mask[indices],
                'pos_label': pos_label[indices],
                'parse_label': parse_label[indices],
            }

            yield out

            i += minibatch_size

    def get_n(self):
        return len(self.cache['syn_text'])

    def get_n_minibatch(self, minibatch_size):
        return math.ceil(self.get_n() / minibatch_size)








