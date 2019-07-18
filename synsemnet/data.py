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
        return ['<UNK>'] + sorted(list(vocab))

    def get_charset(self):
        charset = set()
        for s in self.syn_text + self.sem_text:
            for w in s:
                for c in w:
                    charset.add(c)
        return ['<UNK>'] + sorted(list(charset))

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
