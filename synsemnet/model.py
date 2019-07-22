import sys
import os
import time
import pickle
import numpy as np
import tensorflow as tf

from .kwargs import SYN_SEM_NET_KWARGS
from .backend import *
from .util import *

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

        GRADIENT_FLIP_SCALE = 1.

        # Construct encoders
        self.syntactic_character_rnn = self._initialize_rnn_encoder(
            1,
            [self.word_emb_dim],
            bidirectional=self.bidirectional,
            project_encodings=self.project_word_embeddings,
            return_sequences=False,
            name='syntactic_character_rnn'
        )
        self.semantic_character_rnn = self._initialize_rnn_encoder(
            1,
            [self.word_emb_dim],
            bidirectional=self.bidirectional,
            project_encodings=self.project_word_embeddings,
            return_sequences=False,
            name='semantic_character_rnn'
        )
        self.syntactic_word_encoder = self._initialize_rnn_encoder(
            self.syn_n_layers,
            self.syn_encoder_units,
            bidirectional=self.bidirectional,
            project_encodings=self.project_word_embeddings,
            return_sequences=True,
            name='syntactic_word_encoder'
        )
        self.semantic_word_encoder = self._initialize_rnn_encoder(
            self.sem_n_layers,
            self.sem_encoder_units,
            bidirectional=self.bidirectional,
            project_encodings=self.project_word_embeddings,
            return_sequences=True,
            name='semantic_word_encoder'
        )

        # Construct encodings for syntactic tasks
        self.parsing_word_embeddings_syn = self._initialize_word_embedding(
            self.parsing_character_embeddings_syn,
            self.syntactic_character_rnn,
            character_mask=self.parsing_character_mask
        )
        self.parsing_word_embeddings_sem = self._initialize_word_embedding(
            self.parsing_character_embeddings_sem,
            self.semantic_character_rnn,
            character_mask=self.parsing_character_mask
        )
        self.parsing_word_encodings_syn = self._initialize_encoding(
            self.parsing_word_embeddings_syn,
            self.syntactic_word_encoder,
            mask=self.parsing_word_mask
        )
        self.parsing_word_encodings_sem = self._initialize_encoding(
            self.parsing_word_embeddings_sem,
            self.semantic_word_encoder,
            mask=self.parsing_word_mask
        )
        self.parsing_word_encodings_syn_adversarial = replace_gradient(
            tf.identity,
            lambda x: -(x * GRADIENT_FLIP_SCALE),
            session=self.sess
        )(self.parsing_word_encodings_syn)
        self.parsing_word_encodings_sem_adversarial = replace_gradient(
            tf.identity,
            lambda x: -(x * GRADIENT_FLIP_SCALE),
            session=self.sess
        )(self.parsing_word_encodings_sem)

        # Construct encodings for semantic tasks
        self.sts_s1_word_embeddings_syn = self._initialize_word_embedding(
            self.sts_s1_character_embeddings_syn,
            self.syntactic_character_rnn,
            character_mask=self.sts_s1_character_mask
        )
        self.sts_s1_word_embeddings_sem = self._initialize_word_embedding(
            self.sts_s1_character_embeddings_sem,
            self.semantic_character_rnn,
            character_mask=self.sts_s1_character_mask
        )
        self.sts_s2_word_embeddings_syn = self._initialize_word_embedding(
            self.sts_s2_character_embeddings_syn,
            self.syntactic_character_rnn,
            character_mask=self.sts_s2_character_mask
        )
        self.sts_s2_word_embeddings_sem = self._initialize_word_embedding(
            self.sts_s2_character_embeddings_sem,
            self.semantic_character_rnn,
            character_mask=self.sts_s2_character_mask
        )
        self.sts_s1_word_encodings_syn = self._initialize_encoding(
            self.sts_s1_word_embeddings_syn,
            self.syntactic_word_encoder,
            mask=self.sts_s1_word_mask
        )
        self.sts_s1_word_encodings_sem = self._initialize_encoding(
            self.sts_s1_word_embeddings_sem,
            self.semantic_word_encoder,
            mask=self.sts_s1_word_mask
        )
        self.sts_s2_word_encodings_syn = self._initialize_encoding(
            self.sts_s2_word_embeddings_syn,
            self.syntactic_word_encoder,
            mask=self.sts_s2_word_mask
        )
        self.sts_s2_word_encodings_sem = self._initialize_encoding(
            self.sts_s2_word_embeddings_sem,
            self.semantic_word_encoder,
            mask=self.sts_s2_word_mask
        )
        self.sts_s1_word_encodings_syn_adversarial = replace_gradient(
            tf.identity,
            lambda x: -(x * GRADIENT_FLIP_SCALE),
            session=self.sess
        )(self.sts_s1_word_encodings_syn)
        self.sts_s2_word_encodings_sem_adversarial = replace_gradient(
            tf.identity,
            lambda x: -(x * GRADIENT_FLIP_SCALE),
            session=self.sess
        )(self.sts_s2_word_encodings_sem)
        self.sts_s2_word_encodings_syn_adversarial = replace_gradient(
            tf.identity,
            lambda x: -(x * GRADIENT_FLIP_SCALE),
            session=self.sess
        )(self.sts_s2_word_encodings_syn)
        self.sts_s2_word_encodings_sem_adversarial = replace_gradient(
            tf.identity,
            lambda x: -(x * GRADIENT_FLIP_SCALE),
            session=self.sess
        )(self.sts_s2_word_encodings_sem)

        # Construct outputs for both tasks
        self._initialize_syntactic_outputs()
        self._initialize_semantic_outputs()

        # Construct losses
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.loss = self._initialize_syntactic_objective()
                self.loss += self._initialize_semantic_objective()

        self._initialize_train_op()
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
                self.task = tf.placeholder(self.INT_TF, shape=[None], name='task')

                self.syntactic_character_embedding_matrix = tf.get_variable(
                    shape=[self.n_char + 1, self.character_embedding_dim],
                    dtype=self.FLOAT_TF,
                    initializer=get_initializer('he_normal_initializer', session=self.sess),
                    name='syntactic_character_embedding_matrix'
                )
                self.semantic_character_embedding_matrix = tf.get_variable(
                    shape=[self.n_char + 1, self.character_embedding_dim],
                    dtype=self.FLOAT_TF,
                    initializer=get_initializer('he_normal_initializer', session=self.sess),
                    name='semantic_character_embedding_matrix'
                )

                self._initialize_syntactic_inputs()
                self._initialize_semantic_inputs()

                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_step'
                )
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_batch_step'
                )
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)
                
    def _initialize_syntactic_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.parsing_characters = tf.placeholder(self.INT_TF, shape=[None, None, None], name='parsing_characters')
                self.parsing_character_mask = tf.placeholder(self.FLOAT_TF, shape=[None, None, None], name='parsing_character_mask')
                self.parsing_word_mask = tf.cast(tf.reduce_any(self.parsing_character_mask > 0, axis=-1), dtype=self.FLOAT_TF)
                self.parsing_character_embeddings_syn = tf.gather(self.syntactic_character_embedding_matrix, self.parsing_characters)
                self.parsing_character_embeddings_sem = tf.gather(self.semantic_character_embedding_matrix, self.parsing_characters)

                self.pos_label = tf.placeholder(self.INT_TF, shape=[None, None], name='pos_label')

                self.parse_label = tf.placeholder(self.INT_TF, shape=[None, None], name='parse_label')
                if self.factor_parse_labels:
                    self.parse_depth = tf.placeholder(self.FLOAT_TF, shape=[None, None], name='parse_depth')

    def _initialize_semantic_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.sts_s1_characters = tf.placeholder(self.INT_TF, shape=[None, None, None], name='sts_s1_characters')
                self.sts_s1_character_mask = tf.placeholder(self.FLOAT_TF, shape=[None, None, None], name='sts_s1_character_mask')
                self.sts_s1_word_mask = tf.cast(tf.reduce_any(self.sts_s1_character_mask > 0, axis=-1), dtype=self.FLOAT_TF)
                self.sts_s1_character_embeddings_syn = tf.gather(self.syntactic_character_embedding_matrix, self.sts_s1_characters)
                self.sts_s1_character_embeddings_sem = tf.gather(self.semantic_character_embedding_matrix, self.sts_s1_characters)

                self.sts_s2_characters = tf.placeholder(self.INT_TF, shape=[None, None, None], name='sts_s2_characters')
                self.sts_s2_character_mask = tf.placeholder(self.FLOAT_TF, shape=[None, None, None], name='sts_s2_character_mask')
                self.sts_s2_word_mask = tf.cast(tf.reduce_any(self.sts_s2_character_mask > 0, axis=-1), dtype=self.FLOAT_TF)
                self.sts_s2_character_embeddings_syn = tf.gather(self.syntactic_character_embedding_matrix, self.sts_s2_characters)
                self.sts_s2_character_embeddings_sem = tf.gather(self.semantic_character_embedding_matrix, self.sts_s2_characters)

                # TODO: For Evan, placeholders for STS labels

    def _initialize_rnn_encoder(
            self,
            n_layers,
            n_units,
            bidirectional=True,
            project_encodings=True,
            return_sequences=True,
            name='character_rnn'
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = []
                for l in range(n_layers):
                    if bidirectional:
                        units_cur = int(n_units[l] / 2)
                    else:
                        units_cur = n_units[l]
                    char_encoder_fwd_rnn = RNNLayer(
                        training=self.training,
                        units=units_cur,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        return_sequences=return_sequences,
                        name=name + '_fwd_l%d' % l,
                        session=self.sess
                    )

                    if bidirectional:
                        char_encoder_bwd_rnn = RNNLayer(
                            training=self.training,
                            units=units_cur,
                            activation=self.activation,
                            recurrent_activation=self.recurrent_activation,
                            return_sequences=return_sequences,
                            name=name + '_bwd_l%d' % l,
                            session=self.sess
                        )
                        char_encoder_rnn = make_bi_rnn_layer(char_encoder_fwd_rnn, char_encoder_bwd_rnn, session=self.sess)
                    else:
                        char_encoder_rnn = char_encoder_fwd_rnn
                    out.append(make_lambda(char_encoder_rnn, session=self.sess, use_kwargs=True))

                if project_encodings:
                    if self.resnet_n_layers_inner:
                        projection = DenseResidualLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer='identity_initializer',
                            layers_inner=self.resnet_n_layers_inner,
                            activation_inner=self.activation,
                            activation=None,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_projection'
                        )
                    else:
                        projection = DenseLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer='identity_initializer',
                            activation=None,
                            session=self.sess,
                            name=name + '_projection'
                        )
                    out.append(make_lambda(projection, session=self.sess))

                out = compose_lambdas(out)

                return out

    def _initialize_word_embedding(self, inputs, encoder, character_mask=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                B = tf.shape(inputs)[0]
                W = tf.shape(inputs)[1]
                C = tf.shape(inputs)[2]
                F = inputs.shape[3]

                inputs_flattened = tf.reshape(inputs, [B * W, C, F])
                character_mask_flattened = tf.reshape(character_mask, [B * W, C])

                word_embedding = encoder(inputs_flattened, mask=character_mask_flattened)

                word_embedding = tf.reshape(word_embedding, [B, W, self.word_emb_dim])

                return word_embedding

    def _initialize_encoding(self, inputs, encoder, mask=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                return encoder(inputs, mask=mask) * mask[..., None]

    def _initialize_syntactic_outputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_pos + self.n_parse_label + self.factor_parse_labels

                self.syn_logits_from_syn = DenseLayer(
                    training=self.training,
                    units=units,
                    kernel_initializer='he_normal_initializer',
                    activation=None,
                    session=self.sess,
                    name='logits_from_syn'
                )(self.parsing_word_encodings_syn)

                self.pos_label_logits_from_syn = self.syn_logits_from_syn[..., :self.n_pos]
                self.pos_label_prediction_from_syn = tf.argmax(self.pos_label_logits_from_syn, axis=2)
                self.parse_label_logits_from_syn = self.syn_logits_from_syn[..., self.n_pos:self.n_pos + self.n_parse_label]
                self.parse_label_prediction_from_syn = tf.argmax(self.parse_label_logits_from_syn, axis=2)
                if self.factor_parse_labels:
                    self.parse_depth_logits_from_syn = self.syn_logits_from_syn[..., self.n_pos + self.n_parse_label]
                    self.parse_depth_prediction_from_syn = tf.cast(tf.round(self.parse_depth_logits_from_syn), dtype=self.INT_TF)

                self.syn_logits_from_sem = DenseLayer(
                    training=self.training,
                    units=units,
                    kernel_initializer='he_normal_initializer',
                    activation=None,
                    session=self.sess,
                    name='logits_from_sem'
                )(self.parsing_word_encodings_sem_adversarial)

                self.pos_label_logits_from_sem = self.syn_logits_from_sem[..., :self.n_pos]
                self.pos_label_prediction_from_sem = tf.argmax(self.pos_label_logits_from_sem, axis=2)
                self.parse_label_logits_from_sem = self.syn_logits_from_sem[..., self.n_pos:self.n_pos + self.n_parse_label]
                self.parse_label_prediction_from_sem = tf.argmax(self.parse_label_logits_from_sem, axis=2)
                if self.factor_parse_labels:
                    self.parse_depth_logits_from_sem = self.syn_logits_from_sem[..., self.n_pos + self.n_parse_label]
                    self.parse_depth_prediction_from_sem = tf.cast(tf.round(self.parse_depth_logits_from_sem), dtype=self.INT_TF)

    # TODO: For Evan
    def _initialize_semantic_outputs(self):
        with self.sess.as_default():

            # Define some new tensors for semantic predictions from both syntactic and semantic encoders.

            pass

    def _initialize_syntactic_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loss = 0.
                
                self.syn_pos_label_loss = tf.losses.sparse_softmax_cross_entropy(
                    self.pos_label,
                    self.pos_label_logits_from_syn,
                    weights=self.parsing_word_mask
                )
                self.syn_parse_label_loss = tf.losses.sparse_softmax_cross_entropy(
                    self.parse_label,
                    self.parse_label_logits_from_syn,
                    weights=self.parsing_word_mask
                )

                self.sem_pos_label_loss = tf.losses.sparse_softmax_cross_entropy(
                    self.pos_label,
                    self.pos_label_logits_from_sem,
                    weights=self.parsing_word_mask
                )
                self.sem_parse_label_loss = tf.losses.sparse_softmax_cross_entropy(
                    self.pos_label,
                    self.parse_label_logits_from_sem,
                    weights=self.parsing_word_mask
                )

                loss = self.syn_pos_label_loss + self.syn_parse_label_loss

                if self.factor_parse_labels:
                    self.syn_parse_depth_loss = tf.losses.mean_squared_error(
                        self.parse_depth,
                        self.parse_depth_logits_from_syn,
                        weights=self.parsing_word_mask
                    )
                    self.sem_parse_depth_loss = tf.losses.mean_squared_error(
                        self.parse_depth,
                        self.parse_depth_logits_from_sem,
                        weights=self.parsing_word_mask
                    )

                    # Define well-formedness losses.
                    #   ZERO SUM: In a valid tree, word-by-word changes in depth should sum to 0.
                    #             Encouraged by an L1 loss on the sum of the predicted depths.
                    #   NO NEG:   In a valid tree, no word should close more constituents than it has ancestors.
                    #             Encouraged by an L1 loss on negative cells in a cumsum over predicted depths.

                    masked_depth_logits = self.parse_depth_logits_from_syn * self.parsing_word_mask
                    depth_abs_sums = tf.abs(tf.reduce_sum(masked_depth_logits, axis=1)) # Trying to make these 0
                    depth_abs_clipped_cumsums = tf.abs(tf.clip_by_value(tf.cumsum(masked_depth_logits), -np.inf, 0.))

                    zero_sum_denom = tf.cast(tf.shape(self.parsing_characters)[0], dtype=self.FLOAT_TF) # Normalize by the minibatch size
                    no_neg_denom = tf.reduce_sum(self.parsing_word_mask) + self.epsilon # Normalize by the number of non-padding words

                    self.syn_zero_sum_loss = tf.reduce_sum(depth_abs_sums, axis=0) / zero_sum_denom
                    self.syn_no_neg_loss = tf.reduce_sum(depth_abs_clipped_cumsums, axis=0) / no_neg_denom

                    loss += self.syn_parse_depth_loss + self.syn_zero_sum_loss + self.syn_no_neg_loss
                    
                return loss

    # TODO: For Evan
    def _initialize_semantic_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loss = 0.

                # loss += SOME STUFF

                return loss

    def _initialize_train_op(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.optim = self._initialize_optimizer(self.optim_name)
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_batch_step)

    def _initialize_optimizer(self, name):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase

                    if self.lr_decay_iteration_power != 1:
                        t = tf.cast(self.step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
                    else:
                        t = self.step

                    if self.lr_decay_family.lower() == 'linear_decay':
                        if lr_decay_staircase:
                            decay = tf.floor(t / lr_decay_steps)
                        else:
                            decay = t / lr_decay_steps
                        decay *= lr_decay_rate
                        self.lr = lr - decay
                    else:
                        self.lr = getattr(tf.train, self.lr_decay_family)(
                            lr,
                            t,
                            lr_decay_steps,
                            lr_decay_rate,
                            staircase=lr_decay_staircase,
                            name='learning_rate'
                        )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_global_gradient_norm

                return {
                    'SGD': lambda x: self._clipped_optimizer_class(tf.train.GradientDescentOptimizer)(x, max_global_norm=clip) if clip else tf.train.GradientDescentOptimizer(x),
                    'Momentum': lambda x: self._clipped_optimizer_class(tf.train.MomentumOptimizer)(x, 0.9, max_global_norm=clip) if clip else tf.train.MomentumOptimizer(x, 0.9),
                    'AdaGrad': lambda x: self._clipped_optimizer_class(tf.train.AdagradOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: self._clipped_optimizer_class(tf.train.AdadeltaOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: self._clipped_optimizer_class(tf.train.AdamOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: self._clipped_optimizer_class(tf.train.FtrlOptimizer)(x, max_global_norm=clip) if clip else tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: self._clipped_optimizer_class(tf.train.RMSPropOptimizer)(x, max_global_norm=clip) if clip else tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: self._clipped_optimizer_class(tf.contrib.opt.NadamOptimizer)(x, max_global_norm=clip) if clip else tf.contrib.opt.NadamOptimizer(x)
                }[name](self.lr)

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
    # Private utility methods
    ############################################################

    ## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
    def _clipped_optimizer_class(self, base_optimizer):
        class ClippedOptimizer(base_optimizer):
            def __init__(self, *args, max_global_norm=None, **kwargs):
                super(ClippedOptimizer, self).__init__(*args, **kwargs)
                self.max_global_norm = max_global_norm

            def compute_gradients(self, *args, **kwargs):
                grads_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))
                return grads_and_vars

            def apply_gradients(self, grads_and_vars, **kwargs):
                if self.max_global_norm is None:
                    return grads_and_vars
                grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    # if grad is not None:
                    #     grad = tf.Print(grad, ['max grad', tf.reduce_max(grad), 'min grad', tf.reduce_min(grad)])
                    grads_and_vars.append((grad, var))

                return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

        return ClippedOptimizer





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
                        stderr('Write failure during save. Retrying...\n')
                        time.sleep(1)
                        i += 1
                if i >= 10:
                    stderr('Could not save model to checkpoint file. Saving to backup...\n')
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
                        stderr('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_inner(self, path, predict=False, allow_missing=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                try:
                    if predict:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    sys.stderr.write('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                    if allow_missing:
                        reader = tf.train.NewCheckpointReader(path)
                        saved_shapes = reader.get_variable_to_shape_map()
                        model_var_names = sorted(
                            [(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                        ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                                 if var.name.split(':')[0] in saved_shapes])

                        model_var_names_set = set([x[1] for x in model_var_names])
                        ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                        missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                        if len(missing_in_ckpt) > 0:
                            sys.stderr.write(
                                'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))
                        missing_in_model = ckpt_var_names_set - model_var_names_set
                        if len(missing_in_model) > 0:
                            sys.stderr.write(
                                'Checkpoint file contained the variables below which do not exist in the current model. They will be ignored.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))

                        restore_vars = []
                        name2var = dict(
                            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

                        with tf.variable_scope('', reuse=True):
                            for var_name, saved_var_name in ckpt_var_names:
                                curr_var = name2var[saved_var_name]
                                var_shape = curr_var.get_shape().as_list()
                                if var_shape == saved_shapes[saved_var_name]:
                                    restore_vars.append(curr_var)

                        if predict:
                            self.ema_map = {}
                            for v in restore_vars:
                                self.ema_map[self.ema.average_name(v)] = v
                            saver_tmp = tf.train.Saver(self.ema_map)
                        else:
                            saver_tmp = tf.train.Saver(restore_vars)

                        saver_tmp.restore(self.sess, path)
                    else:
                        raise err

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

    def fit(
            self,
            train_data,
            n_iter,
            verbose=True
    ):
        if self.global_step.eval(session=self.sess) == 0:
            if verbose:
                stderr('Saving initial weights...\n')
            self.save()

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)

        if verbose:
            stderr('*' * 100 + '\n')
            stderr(self.report_settings())
            stderr('\n')
            stderr(self.report_n_params())
            stderr('\n')
            stderr('*' * 100 + '\n\n')

        n_minibatch = train_data.get_n_minibatch(self.minibatch_size)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                while self.global_step.eval(session=self.sess) < n_iter:
                    if verbose:
                        t0_iter = time.time()
                        if verbose:
                            stderr('-' * 50 + '\n')
                            stderr('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                            stderr('\n')
                            pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                        data_feed_train = train_data.get_data_feed(
                            minibatch_size=self.minibatch_size,
                            randomize=True
                        )

                        loss = 0.

                        for i, batch in enumerate(data_feed_train):
                            syn_text_batch = batch['syn_text']
                            syn_text_mask_batch = batch['syn_text_mask']
                            pos_label_batch = batch['pos_label']
                            parse_label_batch = batch['parse_label']
                            if self.factor_parse_labels:
                                parse_depth_batch = batch['parse_depth']
                            else:
                                parse_depth_batch = None

                            fd_minibatch = {
                                self.parsing_characters: syn_text_batch,
                                self.parsing_character_mask: syn_text_mask_batch,
                                self.pos_label: pos_label_batch,
                                self.parse_label: parse_label_batch
                            }
                            if self.factor_parse_labels:
                                fd_minibatch[self.parse_depth] = parse_depth_batch


                            to_run = [
                                self.train_op,
                                self.syn_parse_label_loss,
                                self.syn_pos_label_loss,
                                self.pos_label_prediction_from_syn,
                                self.parse_label_prediction_from_syn
                            ]
                            to_run_names = [
                                'train_op',
                                'syn_parse_label_loss',
                                'syn_pos_label_loss',
                                'pos_label_prediction_from_syn',
                                'parse_label_prediction_from_syn'
                            ]
                            if self.factor_parse_labels:
                                to_run += [
                                    self.parse_depth_prediction_from_syn,
                                    self.syn_parse_depth_loss,
                                    self.syn_zero_sum_loss,
                                    self.syn_no_neg_loss,
                                ]
                                to_run_names += [
                                    'parse_depth_prediction_from_syn',
                                    'syn_parse_depth_loss',
                                    'syn_zero_sum_loss',
                                    'syn_no_neg_loss',
                                ]

                            out = self.sess.run(
                                to_run,
                                feed_dict=fd_minibatch
                            )

                            info_dict = {}
                            for j, x in enumerate(out):
                                info_dict[to_run_names[j]] = x

                            if i % 100 == 0:
                                print(
                                    train_data.pretty_print_syn_predictions(
                                        text=syn_text_batch[:10],
                                        pos_label_true=pos_label_batch[:10],
                                        pos_label_pred=info_dict['pos_label_prediction_from_syn'][:10],
                                        parse_label_true=parse_label_batch[:10],
                                        parse_label_pred=info_dict['parse_label_prediction_from_syn'][:10],
                                        parse_depth_true=parse_depth_batch[:10],
                                        parse_depth_pred=info_dict['parse_depth_prediction_from_syn'][:10] if self.factor_parse_labels else None,
                                        mask=syn_text_mask_batch[:10]
                                    )
                                )

                            if verbose:
                                values = [
                                    ('pos', info_dict['syn_pos_label_loss']),
                                    ('label', info_dict['syn_parse_label_loss'])
                                ]
                                if self.factor_parse_labels:
                                    values += [
                                        ('depth', info_dict['syn_parse_depth_loss']),
                                        ('zero', info_dict['syn_zero_sum_loss']),
                                        ('noneg', info_dict['syn_no_neg_loss']),
                                    ]
                                pb.update(i+1, values=values)

                        samples = train_data.pretty_print_syn_predictions(
                            text=syn_text_batch[:10],
                            pos_label_true=pos_label_batch[:10],
                            pos_label_pred=info_dict['pos_label_prediction_from_syn'][:10],
                            parse_label_true=parse_label_batch[:10],
                            parse_label_pred=info_dict['parse_label_prediction_from_syn'][:10],
                            parse_depth_true=parse_depth_batch[:10],
                            parse_depth_pred=info_dict['parse_depth_prediction_from_syn'][:10] if self.factor_parse_labels else None,
                            mask=syn_text_mask_batch[:10]
                        )

                        stderr('Sample training instances:\n\n' + samples)

                        self.sess.run(self.incr_global_step)

                        self.save()

                        if verbose:
                            t1_iter = time.time()
                            time_str = pretty_print_seconds(t1_iter - t0_iter)
                            stderr('Iteration time: %s\n' % time_str)
