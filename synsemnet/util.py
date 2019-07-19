import sys

def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

def pretty_print_seconds(s):
    s = int(s)
    h = s // 3600
    m = s % 3600 // 60
    s = s % 3600 % 60
    return '%02d:%02d:%02d' % (h, m, s)