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

    def __init__(self, char_set, pos_label_set, parse_label_set, **kwargs):
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self.char_set = char_set
        self.pos_label_set = pos_label_set
        self.parse_label_set = parse_label_set

        self._initialize_session()
        self._initialize_metadata()
        self.build()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        self.UINT_TF = getattr(np, 'u' + self.int_type)
        self.UINT_NP = getattr(tf, 'u' + self.int_type)
        self.regularizer_losses = []

        self.n_char = len(self.char_set)
        self.n_pos = len(self.pos_label_set)
        self.n_parse_label = len(self.parse_label_set)

        if isinstance(self.syn_n_units, str):
            self.syn_encoder_units = [int(x) for x in self.syn_n_units.split()]
            if len(self.syn_encoder_units) == 1:
                self.syn_encoder_units = [self.syn_encoder_units[0]] * self.syn_n_layers
        elif isinstance(self.syn_n_units, int):
            self.syn_encoder_units = [self.syn_n_units] * self.syn_n_layers
        else:
            self.syn_encoder_units = self.syn_n_units

        if isinstance(self.sem_n_units, str):
            self.sem_encoder_units = [int(x) for x in self.sem_n_units.split()]
            if len(self.sem_encoder_units) == 1:
                self.sem_encoder_units = [self.sem_encoder_units[0]] * self.sem_n_layers
        elif isinstance(self.sem_n_units, int):
            self.sem_encoder_units = [self.sem_n_units] * self.sem_n_layers
        else:
            self.sem_encoder_units = self.sem_n_units

        self.predict_mode = False

    def _pack_metadata(self):
        md = {}
        md['char_set'] = self.char_set
        md['pos_label_set'] = self.pos_label_set
        md['parse_label_set'] = self.parse_label_set
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.char_set = md.get('char_set')
        self.pos_label_set = md.get('pos_label_set')
        self.parse_label_set = md.get('parse_label_set')
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        self.build()

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
        self.word_embedding = self._initialize_word_embedding()
        self.synactic_encoding = self._initialize_encoding(
            self.word_embedding,
            self.syn_n_layers,
            self.syn_encoder_units,
            name='syntactic_encoder'
        )
        self.semantic_encoding = self._initialize_encoding(
            self.word_embedding,
            self.sem_n_layers,
            self.sem_encoder_units,
            name='semantic_encoder'
        )
        self._initialize_outputs()
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
                self.training = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='training')

                self.characters = tf.placeholder(tf.string, shape=[None, None, None], name='characters')
                self.char_mask = tf.placeholder(self.FLOAT_TF, shape=[None, None, None], name='char_mask')
                self.word_mask = tf.cast(tf.reduce_any(self.char_mask > 0, axis=-1), dtype=self.FLOAT_TF)
                self.char_table, self.char_embedding_matrix = initialize_embeddings(
                    self.char_set,
                    self.character_emb_dim,
                    name='character_embedding',
                    session=self.sess
                )
                self.char_one_hot = tf.one_hot(
                    self.char_table.lookup(self.characters),
                    self.n_char + 1,
                    dtype=self.FLOAT_TF
                )
                if self.optim_name == 'Nadam':  # Nadam can't handle sparse embedding lookup, so do it with matmul
                    self.character_embeddings = tf.matmul(
                        self.char_one_hot,
                        self.char_embedding_matrix
                    )
                else:
                    self.character_embeddings = tf.nn.embedding_lookup(
                        self.char_embedding_matrix,
                        self.char_table.lookup(self.characters)
                    )

                self.task = tf.placeholder(self.INT_TF, shape=[None], name='task')

    def _initialize_word_embedding(self):
        name = 'word_encoding'
        with self.sess.as_default():
            with self.sess.graph.as_default():
                encoder = self.character_embeddings

                char_encoder_fwd = RNNLayer(
                    training=self.training,
                    units=int(self.word_emb_dim / 2),
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    return_sequences=False,
                    name=name + '_char_encoder_fwd',
                    session=self.sess
                )
                char_encoder_bwd = RNNLayer(
                    training=self.training,
                    units=int(self.word_emb_dim / 2),
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    return_sequences=False,
                    name=name + '_char_encoder_bwd',
                    session=self.sess
                )

                B = tf.shape(encoder)[0]
                W = tf.shape(encoder)[1]
                C = tf.shape(encoder)[2]
                F = encoder.shape[3]

                encoder_flattened = tf.reshape(encoder, [B * W, C, F])
                mask_flattened = tf.reshape(self.char_mask, [B * W, C])

                char_encoder_fwd = char_encoder_fwd(encoder_flattened, mask=mask_flattened)
                char_encoder_bwd = char_encoder_bwd(tf.reverse(encoder_flattened, axis=[1]), tf.reverse(mask_flattened, axis=[1]))

                encoder = tf.concat([char_encoder_bwd, char_encoder_fwd], axis=1)

                encoder = tf.reshape(encoder, [B, W, self.word_emb_dim])

                if self.project_word_embeddings:
                    if self.resnet_n_layers_inner:
                        encoder = DenseResidualLayer(
                            training=self.training,
                            units=self.word_emb_dim,
                            kernel_initializer='identity_initializer',
                            layers_inner=self.resnet_n_layers_inner,
                            activation_inner=self.activation,
                            activation=self.activation,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_projection'
                        )(encoder)
                    else:
                        encoder = DenseLayer(
                            training=self.training,
                            units=self.word_emb_dim,
                            kernel_initializer='identity_initializer',
                            activation=self.activation,
                            session=self.sess,
                            name=name + '_projection'
                        )(encoder)

                return encoder

    def _initialize_encoding(self, word_embedding, n_layers, n_units, name='encoder'):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                encoder = word_embedding

                for l in range(n_layers):
                    encoder_fwd = RNNLayer(
                        training=self.training,
                        units=int(n_units[l] / 2),
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        return_sequences=True,
                        name=name + '_l%d' % l,
                        session=self.sess
                    )(encoder, mask=self.word_mask)

                    if self.bidirectional:
                        encoder_bwd = RNNLayer(
                            training=self.training,
                            units=int(n_units[l] / 2),
                            activation=self.activation,
                            recurrent_activation=self.recurrent_activation,
                            return_sequences=True,
                            name=name + '_l%d' % l,
                            session=self.sess
                        )(tf.reverse(encoder, axis=[1]), mask=tf.reverse(self.word_mask, axis=[1]))
                        encoder = tf.concat([encoder_fwd, encoder_bwd], axis=2)
                    else:
                        encoder = encoder_fwd

                if self.project_encodings:
                    if self.resnet_n_layers_inner:
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
                        )(encoder)

                    encoder *= self.word_mask[..., None]

                return encoder

    def _initialize_outputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.pos_label_prediction_from_syn = DenseLayer(
                    training=self.training,
                    units=self.n_pos,
                    kernel_initializer='he_normal_initializer',
                    activation=None,
                    session=self.sess,
                    name='pos_label_prediction_from_syn'
                )(self.synactic_encoding)
                self.parse_label_prediction_from_syn = DenseLayer(
                    training=self.training,
                    units=self.n_parse_label,
                    kernel_initializer='he_normal_initializer',
                    activation=None,
                    session=self.sess,
                    name='parse_label_prediction_from_syn'
                )(self.synactic_encoding)

                self.pos_label_prediction_from_sem = DenseLayer(
                    training=self.training,
                    units=self.n_pos,
                    kernel_initializer='he_normal_initializer',
                    activation=None,
                    session=self.sess,
                    name='pos_label_prediction_from_sem'
                )(self.semantic_encoding)
                self.parse_label_prediction_from_sem = DenseLayer(
                    training=self.training,
                    units=self.n_parse_label,
                    kernel_initializer='he_normal_initializer',
                    activation=None,
                    session=self.sess,
                    name='parse_label_prediction_from_sem'
                )(self.semantic_encoding)

    def _initialize_objective(self):
        pass

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
        pass






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