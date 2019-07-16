import sys
import os
import time
import pickle
import numpy as np
import tensorflow as tf

from .kwargs import SYN_SEM_NET_KWARGS
from .backend import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

class SynSemNet(object):

    ############################################################
    # Initialization methods
    ############################################################

    _INITIALIZATION_KWARGS = SYN_SEM_NET_KWARGS

    _doc_header = """
        Class implementing a SynSemNet.

    """
    _doc_args = "        :param vocab: ``list``; list of vocabulary items. Items outside this list will be treated as <unk>."
    _doc_args += "        :param charset: ``list`` of characters or ``None``; Characters to use in character-level encoder. If ``None``, no character-level representations will be used."
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (
                                 x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value)
                             for
                             x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    def __init__(self, vocab, charset=None, **kwargs):
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self.vocab = vocab
        self.charset = charset

        self._initialize_session()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        assert not (self.streaming and (self.task.lower() == 'classifier')), 'Streaming mode is not supported for the classifier task.'

        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        self.UINT_TF = getattr(np, 'u' + self.int_type)
        self.UINT_NP = getattr(tf, 'u' + self.int_type)
        self.regularizer_losses = []

        self.vocab_size = len(self.vocab)
        self.charset_size = len(self.charset)

        if isinstance(self.syn_n_units, str):
            self.syn_encoder_units = [int(x) for x in self.syn_n_units.split()]
            if len(self.syn_encoder_units) == 1:
                self.syn_encoder_units = [self.syn_encoder_units[0]] * (self.syn_n_layers - 1)
        elif isinstance(self.syn_n_units, int):
            self.syn_encoder_units = [self.syn_n_units] * (self.syn_n_layers - 1)
        else:
            self.syn_encoder_units = self.syn_n_units

        if isinstance(self.sem_n_units, str):
            self.sem_encoder_units = [int(x) for x in self.sem_n_units.split()]
            if len(self.sem_encoder_units) == 1:
                self.sem_encoder_units = [self.sem_encoder_units[0]] * (self.sem_n_layers - 1)
        elif isinstance(self.sem_n_units, int):
            self.sem_encoder_units = [self.sem_n_units] * (self.sem_n_layers - 1)
        else:
            self.sem_encoder_units = self.sem_n_units

        self.predict_mode = False

    def _pack_metadata(self):
        md = {}
        md['vocab'] = self.vocab
        md['charset'] = self.charset
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.vocab = md.get('vocab')
        self.charset = md.get('charset')
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    def __getstate__(self):
        return self._pack_metadata()

    def __setstate__(self, state):
        self._unpack_metadata(state)
        self._initialize_session()
        self._initialize_metadata()





    ############################################################
    # Private model construction methods
    ############################################################

    def build(self, outdir=None, restore=True, verbose=True):
        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './synsemnet_model/'

                self._initialize_inputs()
                self.synactic_encoding = self._initialize_encoding(self.syn_n_layers, self.syn_encoder_units, name='syntactic_encoder')
                self.semactic_encoding = self._initialize_encoding(self.sem_n_layers, self.sem_encoder_units, name='semantic_encoder')
                self._initialize_objective()
                self._initialize_ema()
                self._initialize_saver()
                self._initialize_logging()

                with self.sess.as_default():
                    with self.sess.graph.as_default():
                        self.report_uninitialized = tf.report_uninitialized_variables(
                            var_list=None
                        )
                self.load(restore=restore)

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.words = tf.placeholder(tf.string, shape=[None, None], name='words')
                self.vocab_table, self.vocab_embedding_matrix = initialize_embeddings(
                    self.vocab,
                    self.vocab_emb_dim,
                    name='vocab_embedding',
                    session=self.sess
                )
                self.words_one_hot = tf.one_hot(
                    self.vocab_table.lookup(self.words),
                    self.vocab_size + 1,
                    dtype=self.FLOAT_TF
                )
                if self.optim_name == 'Nadam':  # Nadam can't handle sparse embedding lookup, so do it with matmul
                    self.word_embeddings = tf.matmul(
                        self.words_one_hot,
                        self.vocab_embedding_matrix
                    )
                else:
                    self.word_embeddings = tf.nn.embedding_lookup(
                        self.vocab_embedding_matrix,
                        self.vocab_table.lookup(self.words)
                    )

                self.task = tf.placeholder(self.INT_TF, shape=[None], name='task')

    def _initialize_encoding(self, n_layers, n_units, name='encoder'):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                encoder = self.word_embeddings
                for l in range(n_layers):
                    encoder = RNNLayer(
                        training=self.training,
                        units=n_units[l],
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        return_sequences=True,
                        name=name + '_l%d' % l,
                        session=self.sess
                    )(encoder)

                if self.project_encodings:
                    if self.resnet_n_layers > 1:
                        encoder = DenseResidualLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer='identity_initializer',
                            layers_inner=self.resnet_n_layers,
                            activation_inner=self.activation,
                            activation=self.activation,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_projection'
                        )(encoder)
                    else:
                        encoder = DenseLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer='identity_initializer',
                            activation=self.activation,
                            session=self.sess,
                            name=name + '_projection'
                        )

                return encoder

    def _initialize_objective(self):
        raise NotImplementedError

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf.check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.ema_decay:
                    vars = [var for var in tf.get_collection('trainable_variables') if 'BatchNorm' not in var.name]

                    self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
                    self.ema_op = self.ema.apply(vars)
                    self.ema_map = {}
                    for v in vars:
                        self.ema_map[self.ema.average_name(v)] = v
                    self.ema_saver = tf.train.Saver(self.ema_map)

    def _initialize_logging(self):
        raise NotImplementedError






    ############################################################
    # Public methods
    ############################################################

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                uninitialized = self.sess.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def save(self, dir=None):

        assert not self.predict_mode, 'Cannot save while in predict mode, since this would overwrite the parameters with their moving averages.'

        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.sess, dir + '/model.ckpt')
                        with open(dir + '/m.obj', 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except Exception:
                        sys.stderr.write('Write failure during save. Retrying...\n')
                        time.sleep(1)
                        i += 1
                if i >= 10:
                    sys.stderr.write('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)

    def load(self, outdir=None, predict=False, restore=True, allow_missing=True):
        """
        Load weights from a DNN-Seg checkpoint and/or initialize the DNN-Seg model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """
        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                    tf.tables_initializer().run()
                if restore and os.path.exists(outdir + '/checkpoint'):
                    self._restore_inner(outdir + '/model.ckpt', predict=predict, allow_missing=allow_missing)
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def set_predict_mode(self, mode):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.ema_decay:
                    reload = mode != self.predict_mode
                    if reload:
                        self.load(predict=mode)

                self.predict_mode = mode

    def report_settings(self, indent=0):
        out = ' ' * indent + 'MODEL SETTINGS:\n'
        out += ' ' * (indent + 2) + 'k: %s\n' %self.k
        for kwarg in SYN_SEM_NET_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

    def report_n_params(self, indent=0):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.sess.run(tf.trainable_variables())
                out = ' ' * indent + 'TRAINABLE PARAMETERS:\n'
                for i in range(len(var_names)):
                    v_name = var_names[i]
                    v_val = var_vals[i]
                    cur_params = np.prod(np.array(v_val).shape)
                    n_params += cur_params
                    out += ' ' * indent + '  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params)
                out += ' ' * indent + '  TOTAL: %d\n\n' % n_params

                return out