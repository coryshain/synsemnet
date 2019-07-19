import numpy as np


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
        self.pos_list = self.get_pos_tagset()
        self.parse_label_list = self.get_parse_labelset()

        self.char_map = {c: i for i, c in enumerate(self.char_list)}
        self.word_map = {w: i for i, w in enumerate(self.word_list)}
        self.pos_map = {p: i for i, p in enumerate(self.pos_list)}
        self.parse_label_map = {l: i for i, l in enumerate(self.parse_label_list)}

        self.n_char = len(self.char_map)
        self.n_word = len(self.word_map)
        self.n_pos = len(self.pos_map)
        self.n_parse_label = len(self.parse_label_map)

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
        return [' '] + sorted(list(charset))

    def get_pos_tagset(self):
        pos_tagset = set()
        for s in self.pos_labels:
            for p in s:
                pos_tagset.add(p)
        return sorted(list(pos_tagset))

    def get_parse_labelset(self):
        parse_labelset = set()
        for s in self.parse_labels:
            for l in s:
                parse_labelset.add(l)
        return sorted(list(parse_labelset))

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

    def as_numeric(self, data_type, max_token=None, max_subtoken=None):
        if data_type.lower().startswith('syn_text'):
            src = 'syn_text'
        elif data_type.lower().startswith('sem_text'):
            src = 'sem_text'
        else:
            src = None

        if data_type.lower().endswith('char_tokenized'):
            as_words = True
            f = lambda x: list(map(self.char2int, x[:max_subtoken]))
        elif data_type.lower().endswith('word'):
            as_words = True
            f = self.word2int
        elif data_type.lower().endswith('char'):
            as_words = False
            f = self.char2int
        else:
            as_words = True
            if data_type.lower().endswith('pos'):
                src = 'pos'
                f = self.pos2int
            elif data_type.lower().endswith('parse_label'):
                src = 'parse_label'
                f = self.parselabel2int
            else:
                raise ValueError('Unrecognized data_type "%s".' % data_type)

        data = self.get_seqs(src=src, as_words=as_words)

        out = []
        for s in data:
            out.append(list(map(f, s))[:max_token])

        out = pad_sequence(out)
        out = out.astype('int')

        return out






