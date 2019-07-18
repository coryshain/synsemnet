from synsemnet.data import Dataset

d = Dataset('wsj/labels/wsj-train.seq_lu', None)
charset = d.get_charset()
print(charset)
print(len(charset))
tagset = d.get_pos_tagset()
print(tagset)
print(len(tagset))
tagset = d.get_parse_labelset()
print(tagset)
print(len(tagset))