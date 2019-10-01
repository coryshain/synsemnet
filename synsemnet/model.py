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

    def __init__(self, char_set, pos_label_set, parse_label_set, sts_label_set, **kwargs):
        for kwarg in SynSemNet._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self.char_set = char_set
        self.pos_label_set = pos_label_set
        self.parse_label_set = parse_label_set
        self.sts_label_set = sts_label_set

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
        self.n_sts_label = len(self.sts_label_set)

        # Encoder layers and units
        assert not self.n_units_encoder is None, 'You must provide a value for **n_units_encoder** when initializing a SynSemNet model.'
        if isinstance(self.n_units_encoder, str):
            self.units_encoder = [int(x) for x in self.n_units_encoder.split()]
        elif isinstance(self.n_units_encoder, int):
            if self.n_layers_encoder is None:
                self.units_encoder = [self.n_units_encoder]
            else:
                self.units_encoder = [self.n_units_encoder] * self.n_layers_encoder
        else:
            self.units_encoder = self.n_units_encoder

        if self.n_layers_encoder is None:
            self.layers_encoder = len(self.units_encoder)
        else:
            self.layers_encoder = self.n_layers_encoder
        if len(self.units_encoder) == 1:
            self.units_encoder = [self.units_encoder[0]] * self.layers_encoder

        assert len(self.units_encoder) == self.layers_encoder, 'Misalignment in number of layers between n_layers_encoder and n_units_encoder.'

        # Parsing decoder layers and units
        assert not self.n_units_parsing_decoder is None, 'You must provide a value for **n_units_parsing_decoder** when initializing a SynSemNet model.'
        if isinstance(self.n_units_parsing_decoder, str):
            self.units_parsing_decoder = [int(x) for x in self.n_units_parsing_decoder.split()]
        elif isinstance(self.n_units_parsing_decoder, int):
            if self.n_layers_parsing_decoder is None:
                self.units_parsing_decoder = [self.n_units_parsing_decoder]
            else:
                self.units_parsing_decoder = [self.n_units_parsing_decoder] * self.n_layers_parsing_decoder
        else:
            self.units_parsing_decoder = self.n_units_parsing_decoder

        if self.n_layers_parsing_decoder is None:
            self.layers_parsing_decoder = len(self.units_parsing_decoder)
        else:
            self.layers_parsing_decoder = self.n_layers_parsing_decoder
        if len(self.units_parsing_decoder) == 1:
            self.units_parsing_decoder = [self.units_parsing_decoder[0]] * self.layers_parsing_decoder

        assert len(self.units_parsing_decoder) == self.layers_parsing_decoder, 'Misalignment in number of layers between n_layers_parsing_decoder and n_units_parsing_decoder.'

        # STS decoder layers and units
        assert not self.n_units_sts_decoder is None, 'You must provide a value for **n_units_sts_decoder** when initializing a SynSemNet model.'
        if isinstance(self.n_units_sts_decoder, str):
            self.units_sts_decoder = [int(x) for x in self.n_units_sts_decoder.split()]
        elif isinstance(self.n_units_sts_decoder, int):
            if self.n_layers_sts_decoder is None:
                self.units_sts_decoder = [self.n_units_sts_decoder]
            else:
                self.units_sts_decoder = [self.n_units_sts_decoder] * self.n_layers_sts_decoder
        else:
            self.units_sts_decoder = self.n_units_sts_decoder

        if self.n_layers_sts_decoder is None:
            self.layers_sts_decoder = len(self.units_sts_decoder)
        else:
            self.layers_sts_decoder = self.n_layers_sts_decoder
        if len(self.units_sts_decoder) == 1:
            self.units_sts_decoder = [self.units_sts_decoder[0]] * self.layers_sts_decoder

        if isinstance(self.sts_conv_kernel_size, str):
            self.sts_kernel_size = [int(x) for x in self.sts_conv_kernel_size.split()]
        elif isinstance(self.sts_conv_kernel_size, int):
            self.sts_kernel_size  = [self.sts_conv_kernel_size] * self.layers_sts_decoder
        else:
            self.sts_kernel_size = self.sts_conv_kernel_size

        assert len(self.units_sts_decoder) == len(self.sts_conv_kernel_size) == self.layers_sts_decoder, 'Misalignment in number of layers between n_layers_sts_decoder, sts_conv_kernel_size, and n_units_sts_decoder.'

        # STS classifier layers and units
        assert not self.n_units_sts_classifier is None, 'You must provide a value for **n_units_sts_classifier** when initializing a SynSemNet model.'
        if isinstance(self.n_units_sts_classifier, str):
            self.units_sts_classifier = [int(x) for x in self.n_units_sts_classifier.split()]
        elif isinstance(self.n_units_sts_classifier, int):
            if self.n_layers_sts_classifier is None:
                self.units_sts_classifier = [self.n_units_sts_classifier]
            else:
                self.units_sts_classifier = [self.n_units_sts_classifier] * self.n_layers_sts_classifier
        else:
            self.units_sts_classifier = self.n_units_sts_classifier

        if self.n_layers_sts_classifier is None:
            self.layers_sts_classifier = len(self.units_sts_classifier)
        else:
            self.layers_sts_classifier = self.n_layers_sts_classifier
        if len(self.units_sts_classifier) == 1:
            self.units_sts_classifier = [self.units_sts_classifier[0]] * self.layers_sts_classifier

        assert len(self.units_sts_classifier) == self.layers_sts_classifier, 'Misalignment in number of layers between n_layers_sts_classifier and n_units_sts_classifier.'

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

        # Construct encoders
        self.syntactic_character_rnn = self._initialize_rnn_module(
            1,
            [self.word_emb_dim],
            activation=self.encoder_activation,
            recurrent_activation=self.encoder_recurrent_activation,
            bidirectional=self.bidirectional_encoder,
            project_encodings=self.project_word_embeddings,
            projection_activation_inner=self.encoder_projection_activation_inner,
            resnet_n_layers_inner=self.encoder_resnet_n_layers_inner,
            return_sequences=False,
            name='syntactic_character_rnn'
        )
        self.semantic_character_rnn = self._initialize_rnn_module(
            1,
            [self.word_emb_dim],
            activation=self.encoder_activation,
            recurrent_activation=self.encoder_recurrent_activation,
            bidirectional=self.bidirectional_encoder,
            project_encodings=self.project_word_embeddings,
            projection_activation_inner=self.encoder_projection_activation_inner,
            resnet_n_layers_inner=self.encoder_resnet_n_layers_inner,
            return_sequences=False,
            name='semantic_character_rnn'
        )
        self.syntactic_word_encoder = self._initialize_rnn_module(
            self.layers_encoder,
            self.units_encoder,
            activation=self.encoder_activation,
            recurrent_activation=self.encoder_recurrent_activation,
            bidirectional=self.bidirectional_encoder,
            project_encodings=self.project_encodings,
            projection_activation_inner=self.encoder_projection_activation_inner,
            resnet_n_layers_inner=self.encoder_resnet_n_layers_inner,
            return_sequences=True,
            name='syntactic_word_encoder'
        )
        self.semantic_word_encoder = self._initialize_rnn_module(
            self.layers_encoder,
            self.units_encoder,
            activation=self.encoder_activation,
            recurrent_activation=self.encoder_recurrent_activation,
            bidirectional=self.bidirectional_encoder,
            project_encodings=self.project_encodings,
            projection_activation_inner=self.encoder_projection_activation_inner,
            resnet_n_layers_inner=self.encoder_resnet_n_layers_inner,
            return_sequences=True,
            name='semantic_word_encoder',
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
            lambda x: -x,
            session=self.sess
        )(self.parsing_word_encodings_syn)
        self.parsing_word_encodings_sem_adversarial = replace_gradient(
            tf.identity,
            lambda x: -x,
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
            lambda x: -x,
            session=self.sess
        )(self.sts_s1_word_encodings_syn)
        self.sts_s1_word_encodings_sem_adversarial = replace_gradient(
            tf.identity,
            lambda x: -x,
            session=self.sess
        )(self.sts_s1_word_encodings_sem)
        self.sts_s2_word_encodings_syn_adversarial = replace_gradient(
            tf.identity,
            lambda x: -x,
            session=self.sess
        )(self.sts_s2_word_encodings_syn)
        self.sts_s2_word_encodings_sem_adversarial = replace_gradient(
            tf.identity,
            lambda x: -x,
            session=self.sess
        )(self.sts_s2_word_encodings_sem)

        # Construct outputs for both tasks
        self._initialize_parsing_outputs()
        self._initialize_sts_outputs()

        # Construct losses
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.loss = 0
                self.adversarial_loss = 0
                
                parsing_loss, parsing_adversarial_loss = self._initialize_parsing_objective()
                self.loss += parsing_loss
                self.adversarial_loss += parsing_adversarial_loss
                
                sts_loss, sts_adversarial_loss = self._initialize_sts_objective()
                self.loss += sts_loss
                self.adversarial_loss += sts_adversarial_loss
                
                self.total_loss = self.loss + self.adversarial_loss

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

                self._initialize_parsing_inputs()
                self._initialize_sts_inputs()

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
                
    def _initialize_parsing_inputs(self):
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

    def _initialize_sts_inputs(self):
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

                #self.sts_label = tf.placeholder(self.FLOAT_TF, shape=[None], name='sts_label')
                self.sts_label = tf.placeholder(self.INT_TF, shape=[None], name='sts_label')

    def _initialize_dense_module(
            self,
            n_layers,
            n_units,
            kernel_initializer='he_normal_initializer',
            activation=None,
            activation_inner='elu',
            resnet_n_layers_inner=None,
            reuse=None,
            name='word_dense'
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if reuse is None:
                    reuse = tf.AUTO_REUSE
                out = []
                for l in range(n_layers):
                    activation_inner_cur = activation_inner
                    if l < n_layers - 1:
                        activation_cur = activation_inner
                    else:
                        activation_cur = activation

                    units_cur = n_units[l]
                    
                    if resnet_n_layers_inner:
                        word_encoder_dense = DenseResidualLayer(
                            training=self.training,
                            units=units_cur,
                            layers_inner=resnet_n_layers_inner,
                            kernel_initializer=kernel_initializer,
                            activation_inner=activation_inner_cur,
                            activation=activation_cur,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_l%d' % l,
                            reuse=reuse
                        )
                    else:
                        word_encoder_dense = DenseLayer(
                            training=self.training,
                            units=units_cur,
                            kernel_initializer=kernel_initializer,
                            activation=activation_cur,
                            name=name + '_l%d' % l,
                            reuse=reuse,
                            session=self.sess
                        )

                    out.append(make_lambda(word_encoder_dense, session=self.sess))

                out = compose_lambdas(out)

                return out

    def _initialize_rnn_module(
            self,
            n_layers,
            n_units,
            kernel_initializer='he_normal_initializer',
            recurrent_initializer='orthogonal_initializer',
            activation='tanh',
            activation_inner='tanh',
            recurrent_activation='sigmoid',
            bidirectional=True,
            project_encodings=True,
            projection_activation_inner='elu',
            resnet_n_layers_inner=None,
            return_sequences=True,
            reuse=None,
            passthru=False,
            name='character_rnn'
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if reuse is None:
                    reuse = tf.AUTO_REUSE
                out = []
                for l in range(n_layers):
                    if l < n_layers - 1:
                        activation_cur = activation_inner
                    else:
                        activation_cur = activation

                    if bidirectional:
                        assert n_units[l] % 2 == 0, 'Bidirectional RNNs must have an even number of hidden units. Saw %d.' % n_units[l]
                        units_cur = int(n_units[l] / 2)
                    else:
                        units_cur = n_units[l]

                    char_encoder_fwd_rnn = RNNLayer(
                        training=self.training,
                        units=units_cur,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        activation=activation_cur,
                        recurrent_activation=recurrent_activation,
                        return_sequences=return_sequences,
                        name=name + '_fwd_l%d' % l,
                        session=self.sess
                    )

                    if bidirectional:
                        char_encoder_bwd_rnn = RNNLayer(
                            training=self.training,
                            units=units_cur,
                            kernel_initializer=kernel_initializer,
                            recurrent_initializer=recurrent_initializer,
                            activation=activation_cur,
                            recurrent_activation=recurrent_activation,
                            return_sequences=return_sequences,
                            name=name + '_bwd_l%d' % l,
                            session=self.sess
                        )
                        char_encoder_rnn = make_bi_rnn_layer(char_encoder_fwd_rnn, char_encoder_bwd_rnn, session=self.sess)
                    else:
                        char_encoder_rnn = char_encoder_fwd_rnn
                    if passthru: # Concat inputs and outputs to allow passthru connections
                        def char_encoder_rnn(x, layer=char_encoder_rnn, mask=None):
                            below = x
                            above = layer(x, mask=mask)
                            out = tf.concat([below, above], axis=-1)
                            return out

                    out.append(make_lambda(char_encoder_rnn, session=self.sess, use_kwargs=True))

                if project_encodings:
                    if resnet_n_layers_inner:
                        projection = DenseResidualLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer=kernel_initializer,
                            layers_inner=resnet_n_layers_inner,
                            activation_inner=projection_activation_inner,
                            activation=None,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
                        )
                    else:
                        projection = DenseLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer=kernel_initializer,
                            activation=None,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
                        )
                    out.append(make_lambda(projection, session=self.sess))

                out = compose_lambdas(out)

                return out

    def _initialize_cnn_module(
            self,
            n_layers,
            kernel_size,
            n_units,
            padding='valid',
            kernel_initializer='he_normal_initializer',
            activation='elu',
            activation_inner='elu',
            project_encodings=True,
            resnet_n_layers_inner=None,
            projection_activation_inner='elu',
            max_pooling_over_time=True,
            reuse=None,
            name='word_cnn'
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if reuse is None:
                    reuse = tf.AUTO_REUSE
                out = []
                for l in range(n_layers):
                    kernel_size_cur = kernel_size[l]

                    if l < n_layers - 1:
                        activation_cur = activation_inner
                    else:
                        activation_cur = activation

                    units_cur = n_units[l]

                    if resnet_n_layers_inner:
                        word_encoder_cnn = Conv1DResidualLayer(
                            training=self.training,
                            kernel_size=kernel_size_cur,
                            n_filters=units_cur,
                            kernel_initializer=kernel_initializer,
                            activation=activation_cur,
                            padding=padding,
                            layers_inner=resnet_n_layers_inner,
                            project_inputs=False,
                            name=name + '_l%d' % l,
                            reuse=reuse,
                            session=self.sess
                        )
                    else:
                        word_encoder_cnn = Conv1DLayer(
                            training=self.training,
                            kernel_size=kernel_size_cur,
                            n_filters=units_cur,
                            kernel_initializer=kernel_initializer,
                            activation=activation_cur,
                            padding=padding,
                            name=name + '_l%d' % l,
                            reuse=reuse,
                            session=self.sess
                        )

                    out.append(make_lambda(word_encoder_cnn, session=self.sess))

                    if max_pooling_over_time:
                        out.append(make_lambda(lambda x: tf.reduce_max(x, axis=1), session=self.sess))

                if project_encodings:
                    if resnet_n_layers_inner:
                        projection = DenseResidualLayer(
                            training=self.training,
                            units=n_units[-1],
                            layers_inner=resnet_n_layers_inner,
                            kernel_initializer=kernel_initializer,
                            activation_inner=projection_activation_inner,
                            activation=activation,
                            project_inputs=False,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
                        )
                    else:
                        projection = DenseLayer(
                            training=self.training,
                            units=n_units[-1],
                            kernel_initializer=kernel_initializer,
                            activation=activation,
                            session=self.sess,
                            name=name + '_projection',
                            reuse=reuse
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

    def _initialize_parsing_outputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_pos + self.n_parse_label + self.factor_parse_labels

                self.parsing_logits_syn = self._initialize_dense_module(
                    self.layers_parsing_decoder,
                    self.units_parsing_decoder,
                    activation=self.parsing_decoder_activation,
                    activation_inner=self.parsing_decoder_activation_inner,
                    resnet_n_layers_inner=self.parsing_decoder_resnet_n_layers_inner,
                    name='parsing_logits_syn'
                )(self.parsing_word_encodings_syn)

                self.pos_label_logits_syn = self.parsing_logits_syn[..., :self.n_pos]
                self.pos_label_prediction_syn = tf.argmax(self.pos_label_logits_syn, axis=2)
                self.parse_label_logits_syn = self.parsing_logits_syn[..., self.n_pos:self.n_pos + self.n_parse_label]
                self.parse_label_prediction_syn = tf.argmax(self.parse_label_logits_syn, axis=2)
                if self.factor_parse_labels:
                    self.parse_depth_logits_syn = self.parsing_logits_syn[..., self.n_pos + self.n_parse_label]
                    self.parse_depth_prediction_syn = tf.cast(tf.round(self.parse_depth_logits_syn), dtype=self.INT_TF)

                self.parsing_logits_sem = self._initialize_dense_module(
                    self.layers_parsing_decoder,
                    self.units_parsing_decoder,
                    activation=self.parsing_decoder_activation,
                    activation_inner=self.parsing_decoder_activation_inner,
                    resnet_n_layers_inner=self.parsing_decoder_resnet_n_layers_inner,
                    name='parsing_logits_syn'
                )(self.parsing_word_encodings_sem_adversarial)

                self.pos_label_logits_sem = self.parsing_logits_sem[..., :self.n_pos]
                self.pos_label_prediction_sem = tf.argmax(self.pos_label_logits_sem, axis=2)
                self.parse_label_logits_sem = self.parsing_logits_sem[..., self.n_pos:self.n_pos + self.n_parse_label]
                self.parse_label_prediction_sem = tf.argmax(self.parse_label_logits_sem, axis=2)
                if self.factor_parse_labels:
                    self.parse_depth_logits_sem = self.parsing_logits_sem[..., self.n_pos + self.n_parse_label]
                    self.parse_depth_prediction_sem = tf.cast(tf.round(self.parse_depth_logits_syn), dtype=self.INT_TF)

    def _initialize_sts_outputs(self):
        with self.sess.as_default():
            # Define some new tensors for semantic predictions from both syntactic and semantic encoders.
            # STS decoder (sem)
            if self.sts_decoder_type.lower() == 'cnn':
                self.sts_decoder_sem = self._initialize_cnn_module(
                    self.layers_sts_decoder,
                    self.sts_kernel_size,
                    self.units_sts_decoder,
                    activation=self.sts_decoder_activation,
                    activation_inner=self.sts_decoder_activation_inner,
                    padding='same',
                    project_encodings=self.project_sts_decodings,
                    projection_activation_inner=self.sts_projection_activation_inner,
                    resnet_n_layers_inner=self.sts_decoder_resnet_n_layers_inner,
                    max_pooling_over_time=True,
                    name='sts_decoder_sem'
                ) #confirm hyperparams with the shao2017 paper: CNN: 1 layer, filter_height=1, 300 filters, relu activation, no dropout or regularization.  then fed to difference and hadamard and concatenated.  then FCNN: 2 layers, 300 units, tanh activation, no regularization or dropout
            elif self.sts_decoder_type.lower() == 'rnn':
                self.sts_decoder_sem = self._initialize_rnn_module(
                    self.layers_sts_decoder,
                    self.units_sts_decoder,
                    bidirectional=self.bidirectional_sts_decoder,
                    activation=self.sts_decoder_activation,
                    activation_inner=self.sts_decoder_activation_inner,
                    recurrent_activation=self.sts_decoder_recurrent_activation,
                    project_encodings=self.project_sts_decodings,
                    projection_activation_inner=self.sts_projection_activation_inner,
                    resnet_n_layers_inner=self.sts_decoder_resnet_n_layers_inner,
                    return_sequences=False,
                    name='sts_decoder_sem'
                )
            else:
                raise ValueError('Unrecognized STS decoder type "%s".' % self.sts_decoder_type)
            self.sts_latent_s1_sem = self.sts_decoder_sem(self.sts_s1_word_encodings_sem)
            self.sts_latent_s2_sem = self.sts_decoder_sem(self.sts_s2_word_encodings_sem)
            #sts predictions from semantic encoders
            self.sts_difference_feats_sem = tf.subtract(
                    self.sts_latent_s1_sem,
                    self.sts_latent_s2_sem,
                    name='sts_difference_feats_sem')
            self.sts_product_feats_sem = tf.multiply( #this is element-wise, aka hadamard
                    self.sts_latent_s1_sem,
                    self.sts_latent_s2_sem,
                    name='sts_product_feats_sem') 
            self.sts_features_sem = tf.concat(
                    values=[self.sts_difference_feats_sem, self.sts_product_feats_sem], 
                    axis=1, 
                    name='sts_features_sem')
            #self.sts_logits_sem from self.sts_features_sem with 2 denselayer (section 2 fcnn) from Shao 2017
            self.sts_classifier_sem = self._initialize_dense_module(
                self.layers_sts_classifier,
                self.units_sts_classifier,
                activation=None,
                activation_inner=self.sts_classifier_activation_inner,
                resnet_n_layers_inner=self.sts_classifier_resnet_n_layers_inner,
                name='sts_classifier_sem'
            )
            self.sts_logits_sem = self.sts_classifier_sem(self.sts_features_sem)
            self.sts_prediction_sem = tf.argmax(self.sts_logits_sem, axis=-1)

            # STS decoder (syn)
            if self.sts_decoder_type.lower() == 'cnn':
                self.sts_decoder_syn = self._initialize_cnn_module(
                    self.layers_sts_decoder,
                    self.sts_kernel_size,
                    self.units_sts_decoder,
                    activation=self.sts_decoder_activation,
                    activation_inner=self.sts_decoder_activation_inner,
                    padding='same',
                    project_encodings=self.project_sts_decodings,
                    projection_activation_inner=self.sts_projection_activation_inner,
                    resnet_n_layers_inner=self.sts_decoder_resnet_n_layers_inner,
                    max_pooling_over_time=True,
                    name='sts_decoder_sem'
                ) #confirm hyperparams with the shao2017 paper: CNN: 1 layer, filter_height=1, 300 filters, relu activation, no dropout or regularization.  then fed to difference and hadamard and concatenated.  then FCNN: 2 layers, 300 units, tanh activation, no regularization or dropout
            elif self.sts_decoder_type.lower() == 'rnn':
                self.sts_decoder_syn = self._initialize_rnn_module(
                    self.layers_sts_decoder,
                    self.units_sts_decoder,
                    bidirectional=self.bidirectional_sts_decoder,
                    activation=self.sts_decoder_activation,
                    activation_inner=self.sts_decoder_activation_inner,
                    recurrent_activation=self.sts_decoder_recurrent_activation,
                    project_encodings=self.project_sts_decodings,
                    projection_activation_inner=self.sts_projection_activation_inner,
                    resnet_n_layers_inner=self.sts_decoder_resnet_n_layers_inner,
                    return_sequences=False,
                    name='sts_decoder_sem'
                )
            else:
                raise ValueError('Unrecognized STS decoder type "%s".' % self.sts_decoder_type)
            self.sts_latent_s1_syn = self.sts_decoder_syn(self.sts_s1_word_encodings_syn_adversarial)
            self.sts_latent_s2_syn = self.sts_decoder_syn(self.sts_s2_word_encodings_syn_adversarial)
            #sts predictions from syntactic encoders 
            self.sts_difference_feats_syn = tf.subtract(
                    self.sts_latent_s1_syn,
                    self.sts_latent_s2_syn,
                    name='sts_difference_feats_syn')
            self.sts_product_feats_syn = tf.multiply(
                    self.sts_latent_s1_syn,
                    self.sts_latent_s2_syn,
                    name='sts_product_feats_syn')
            self.sts_features_syn = tf.concat(
                    values=[self.sts_difference_feats_syn, self.sts_product_feats_syn], 
                    axis=1, 
                    name='sts_features_syn')
            self.sts_classifier_syn = self._initialize_dense_module(
                self.layers_sts_classifier,
                self.units_sts_classifier,
                activation=None,
                activation_inner=self.sts_classifier_activation_inner,
                resnet_n_layers_inner=self.sts_classifier_resnet_n_layers_inner,
                name='sts_classifier_syn'
            )
            self.sts_logits_syn = self.sts_classifier_syn(self.sts_features_syn)
            self.sts_prediction_syn = tf.argmax(self.sts_logits_syn, axis=-1)

    def _initialize_parsing_objective(self, well_formedness_loss=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.parsing_loss_scale:
                    self.pos_label_loss_syn = tf.losses.sparse_softmax_cross_entropy(
                        self.pos_label,
                        self.pos_label_logits_syn,
                        weights=self.parsing_word_mask
                    )
                    self.parse_label_loss_syn = tf.losses.sparse_softmax_cross_entropy(
                        self.parse_label,
                        self.parse_label_logits_syn,
                        weights=self.parsing_word_mask
                    )
                else:
                    self.pos_label_loss_syn = 0
                    self.parse_label_loss_syn = 0

                if self.parsing_adversarial_loss_scale:
                    self.pos_label_loss_sem = tf.losses.sparse_softmax_cross_entropy(
                        self.pos_label,
                        self.pos_label_logits_sem,
                        weights=self.parsing_word_mask
                    )
                    self.parse_label_loss_sem = tf.losses.sparse_softmax_cross_entropy(
                        self.parse_label,
                        self.parse_label_logits_sem,
                        weights=self.parsing_word_mask
                    )
                else:
                    self.pos_label_loss_sem = 0
                    self.parse_label_loss_sem = 0

                loss = self.pos_label_loss_syn + self.parse_label_loss_syn
                adversarial_loss = self.pos_label_loss_sem + self.parse_label_loss_sem

                if self.parsing_loss_scale or self.parsing_adversarial_loss_scale:
                    if self.factor_parse_labels:
                        self.parse_depth_loss_syn = tf.losses.mean_squared_error(
                            self.parse_depth,
                            self.parse_depth_logits_syn,
                            weights=self.parsing_word_mask
                        )
                        self.parse_depth_loss_sem = tf.losses.mean_squared_error(
                            self.parse_depth,
                            self.parse_depth_logits_sem,
                            weights=self.parsing_word_mask
                        )

                        loss += self.parse_depth_loss_syn
                        adversarial_loss += self.parse_depth_loss_sem

                        if well_formedness_loss:
                            # Define well-formedness losses.
                            #   ZERO SUM: In a valid tree, word-by-word changes in depth should sum to 0.
                            #             Encouraged by an L1 loss on the sum of the predicted depths.
                            #   NO NEG:   In a valid tree, no word should close more constituents than it has ancestors.
                            #             Encouraged by an L1 loss on negative cells in a cumsum over predicted depths.

                            zero_sum_denom = tf.cast(tf.shape(self.parsing_characters)[0], dtype=self.FLOAT_TF) # Normalize by the minibatch size
                            no_neg_denom = tf.reduce_sum(self.parsing_word_mask) + self.epsilon # Normalize by the number of non-padding words

                            masked_depth_logits = self.parse_depth_logits_syn * self.parsing_word_mask
                            depth_abs_sums = tf.abs(tf.reduce_sum(masked_depth_logits, axis=1)) # Trying to make these 0
                            depth_abs_clipped_cumsums = tf.abs(tf.clip_by_value(tf.cumsum(masked_depth_logits), -np.inf, 0.))

                            self.zero_sum_loss_syn = tf.reduce_sum(depth_abs_sums, axis=0) / zero_sum_denom
                            self.no_neg_loss_syn = tf.reduce_sum(depth_abs_clipped_cumsums, axis=0) / no_neg_denom

                            masked_depth_logits = self.parse_depth_logits_sem * self.parsing_word_mask
                            depth_abs_sums = tf.abs(tf.reduce_sum(masked_depth_logits, axis=1))  # Trying to make these 0
                            depth_abs_clipped_cumsums = tf.abs(tf.clip_by_value(tf.cumsum(masked_depth_logits), -np.inf, 0.))

                            self.zero_sum_loss_sem = tf.reduce_sum(depth_abs_sums, axis=0) / zero_sum_denom
                            self.no_neg_loss_sem = tf.reduce_sum(depth_abs_clipped_cumsums, axis=0) / no_neg_denom

                            loss += self.zero_sum_loss_syn + self.no_neg_loss_syn
                            adversarial_loss += self.zero_sum_loss_sem + self.no_neg_loss_sem
                else:
                    self.zero_sum_loss_syn = 0
                    self.no_neg_loss_syn = 0
                    self.zero_sum_loss_sem = 0
                    self.no_neg_loss_sem = 0

                if self.parsing_loss_scale:
                    loss *= self.parsing_loss_scale
                if self.parsing_adversarial_loss_scale:
                    loss *= self.parsing_adversarial_loss_scale

                return loss, adversarial_loss

    def _initialize_sts_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.sts_loss_scale:
                    self.sts_loss_sem = tf.losses.sparse_softmax_cross_entropy(
                        self.sts_label,
                        self.sts_logits_sem
                    ) * self.sts_loss_scale
                else:
                    self.sts_loss_sem = 0

                #self.sts_loss_sem = tf.losses.mean_squared_error(
                #    self.sts_label,
                #    self.sts_label_prediction_sem
                #    )

                if self.sts_adversarial_loss_scale:
                    self.sts_loss_syn = tf.losses.sparse_softmax_cross_entropy(
                        self.sts_label,
                        self.sts_logits_syn
                    ) * self.sts_adversarial_loss_scale

                #self.sts_loss_syn = tf.losses.mean_squared_error(
                #    self.sts_label,
                #    self.sts_label_prediction_syn
                #    )

                loss = self.sts_loss_sem
                adversarial_loss = self.sts_loss_syn

                return loss, adversarial_loss

    def _initialize_train_op(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.optim = self._initialize_optimizer(self.optim_name)
                self.train_op = self.optim.minimize(self.total_loss, global_step=self.global_batch_step)

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
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.log_graph:
                    self.train_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/train', self.sess.graph)
                    self.dev_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dev', self.sess.graph)
                else:
                    self.train_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/train')
                    self.dev_writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dev')

                self.parsing_log_entries = self._initialize_parsing_log_entries(syn=True, sem=True)
                self.parsing_log_summaries = self._initialize_parsing_log_summaries(self.parsing_log_entries)
                self.parsing_summary = tf.summary.merge_all(key='parsing_losses')
                
                self.sts_log_entries = self._initialize_sts_log_entries(syn=True, sem=True)
                self.sts_log_summaries = self._initialize_sts_log_summaries(self.sts_log_entries)
                self.sts_summary = tf.summary.merge_all(key='sts_losses')

    def _initialize_parsing_log_entries(self, syn=True, sem=True):
        log_entries = []
        if syn:
            log_entries += [
                'pos_label_loss_syn',
                'parse_label_loss_syn',
                'parse_depth_loss_syn',
            ]
        if sem:
            log_entries += [
                'pos_label_loss_sem',
                'parse_label_loss_sem',
                'parse_depth_loss_sem',
            ]
            
        return log_entries
        
    def _initialize_parsing_log_summaries(self, log_entries, collection='parsing_losses'):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                log_summaries = {}

                for x in log_entries:
                    log_summaries[x] = tf.placeholder(self.FLOAT_TF, shape=[], name=x + '_placeholder')
                    tf.summary.scalar('%s/%s' % (collection, x), log_summaries[x], collections=[collection])

                return log_summaries
            
    def _initialize_sts_log_entries(self, syn=True, sem=True):
        log_entries = []
        if syn:
            log_entries += ['sts_loss_syn']
        if sem:
            log_entries += ['sts_loss_sem']

        return log_entries

    def _initialize_sts_log_summaries(self, log_entries, collection='sts_losses'):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                log_summaries = {}

                for x in log_entries:
                    log_summaries[x] = tf.placeholder(self.FLOAT_TF, shape=[], name=x + '_placeholder')
                    tf.summary.scalar('%s/%s' % (collection, x), log_summaries[x], collections=[collection])

                return log_summaries
        

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

    def _get_n_batches(self, i):
        if self.eval_freq:
            eval_freq = self.eval_freq
        else:
            eval_freq = np.inf
        if self.save_freq:
            save_freq = self.save_freq
        else:
            save_freq = np.inf
        if self.log_freq:
            log_freq = self.log_freq
        else:
            log_freq = np.inf

        evaluate = ((i + 1) % eval_freq == 0) and (i > self.n_pretrain_steps)
        save = evaluate or ((i + 1) % save_freq == 0)
        log = (i + 1) % log_freq == 0

        if save_freq:
            next_save = save_freq - (i % save_freq)
        else:
            next_save = np.inf

        if log_freq:
            next_log = log_freq - (i % log_freq)
        else:
            next_log = np.inf

        if eval_freq:
            next_evaluate = eval_freq - (i % eval_freq)
        else:
            next_evaluate = np.inf

        n = min(
            next_save,
            next_log,
            next_evaluate
        )

        save = next_save == n
        log = next_log == n
        evaluate = next_evaluate == n

        return n, save, log, evaluate

    def _run_batches_inner(
            self,
            to_run,
            to_run_names,
            data_feed,
            n_minibatch,
            data_type='both',
            info_dict=None,
            update=False,
            return_syn_parsing_losses=False,
            return_sem_sts_losses=False,
            return_syn_parsing_predictions=False,
            return_sem_parsing_predictions=False,
            return_syn_sts_predictions=False,
            return_sem_sts_predictions=False,
            verbose=True
    ):
        if info_dict is None:
            info_dict = {}
        gold_keys = set()
        for k in to_run_names:
            if 'loss' in k:
                info_dict[k] = 0.
            elif 'prediction' in k:
                info_dict[k] = []
                gold_key = k.replace('_syn', '').replace('_sem', '').replace('prediction', 'true')
                if not gold_key in info_dict:
                    gold_keys.add(gold_key)
                    info_dict[gold_key] = []

        if data_type.lower() in ['parsing', 'both'] and return_syn_parsing_predictions or return_sem_parsing_predictions:
            info_dict['parsing_text'] = []
            info_dict['parsing_text_mask'] = []
            gold_keys.add('parsing_text')
            gold_keys.add('parsing_text_mask')

        if data_type.lower() in ['sts', 'both'] and return_syn_sts_predictions or return_sem_sts_predictions:
            info_dict['sts_s1_text'] = []
            info_dict['sts_s1_text_mask'] = []
            info_dict['sts_s2_text'] = []
            info_dict['sts_s2_text_mask'] = []
            gold_keys.add('sts_s1_text')
            gold_keys.add('sts_s1_text_mask')
            gold_keys.add('sts_s2_text')
            gold_keys.add('sts_s2_text_mask')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if verbose:
                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                i = 0
                while i < n_minibatch:
                    batch = next(data_feed)
                    if data_type.lower() in ['parsing', 'both']:
                        parsing_text_batch = batch['parsing_text']
                        parsing_text_mask_batch = batch['parsing_text_mask']
                        pos_label_batch = batch['pos_label']
                        parse_label_batch = batch['parse_label']
                        parse_depth_batch = batch['parse_depth']
                    if data_type.lower() in ['sts', 'both']:
                        sts_s1_text_batch = batch['sts_s1_text']
                        sts_s1_text_mask_batch = batch['sts_s1_text_mask']
                        sts_s2_text_batch = batch['sts_s2_text']
                        sts_s2_text_mask_batch = batch['sts_s2_text_mask']
                        sts_label_batch = batch['sts_label']

                    if 'parsing_text' in gold_keys:
                        info_dict['parsing_text'].append(parsing_text_batch)
                    if 'parsing_text_mask' in gold_keys:
                        info_dict['parsing_text_mask'].append(parsing_text_mask_batch)
                    if 'pos_label_true' in gold_keys:
                        info_dict['pos_label_true'].append(pos_label_batch)
                    if 'parse_label_true' in gold_keys:
                        info_dict['parse_label_true'].append(parse_label_batch)
                    if 'parse_depth_true' in gold_keys:
                        info_dict['parse_depth_true'].append(parse_depth_batch)
                    if 'sts_s1_text' in gold_keys:
                        info_dict['sts_s1_text'].append(sts_s1_text_batch)
                    if 'sts_s1_text_mask' in gold_keys:
                        info_dict['sts_s1_text_mask'].append(sts_s1_text_mask_batch)
                    if 'sts_s2_text' in gold_keys:
                        info_dict['sts_s2_text'].append(sts_s2_text_batch)
                    if 'sts_s2_text_mask' in gold_keys:
                        info_dict['sts_s2_text_mask'].append(sts_s2_text_mask_batch)
                    if 'sts_true' in gold_keys:
                        info_dict['sts_true'].append(sts_label_batch)

                    fd_minibatch = {}
                    if data_type.lower() in ['parsing', 'both']:
                        fd_minibatch.update({
                            self.parsing_characters: parsing_text_batch,
                            self.parsing_character_mask: parsing_text_mask_batch,
                            self.pos_label: pos_label_batch,
                            self.parse_label: parse_label_batch,
                        })
                        if self.factor_parse_labels:
                            fd_minibatch[self.parse_depth] = parse_depth_batch
                    if data_type.lower() in ['sts', 'both']:
                        fd_minibatch.update({
                            self.sts_s1_characters: sts_s1_text_batch,
                            self.sts_s1_character_mask: sts_s1_text_mask_batch,
                            self.sts_s2_characters: sts_s2_text_batch,
                            self.sts_s2_character_mask: sts_s2_text_mask_batch,
                            self.sts_label: sts_label_batch
                        })

                    out = self.sess.run(
                        to_run,
                        feed_dict=fd_minibatch
                    )

                    batch_dict = {}
                    for j, x in enumerate(out):
                        batch_dict[to_run_names[j]] = x

                    for k in batch_dict:
                        if 'loss' in k:
                            info_dict[k] += batch_dict[k]
                        elif 'prediction' in k:
                            info_dict[k].append(batch_dict[k])

                    if verbose:
                        values = []
                        if update:
                            if data_type.lower() in ['parsing', 'both'] and return_syn_parsing_losses:
                                values += [
                                    ('pos', batch_dict['pos_label_loss_syn']),
                                    ('label', batch_dict['parse_label_loss_syn'])
                                ]
                                if self.factor_parse_labels:
                                    values += [
                                        ('depth', batch_dict['parse_depth_loss_syn'])
                                    ]
                            if data_type.lower() in ['sts', 'both'] and return_sem_sts_losses:
                                values += [
                                    ('sts', batch_dict['sts_loss_sem']),
                                ]
                        pb.update(i + 1, values=values)

                    i += 1

                for k in to_run_names + sorted(list(gold_keys)):
                    if 'loss' in k:
                        info_dict[k] /= n_minibatch
                    elif 'prediction' in k or k in gold_keys:
                        if len(info_dict[k]) > 0:
                            info_dict[k] = np.concatenate(info_dict[k], axis=0)
                        else:
                            print('Empty list:')
                            print(k)
                            print()

        return info_dict

    def _run_batches(
            self,
            data,
            data_name='train',
            minibatch_size=None,
            n_minibatch=None,
            update=False,
            randomize=False,
            return_syn_parsing_losses=False,
            return_sem_parsing_losses=False,
            return_syn_sts_losses=False,
            return_sem_sts_losses=False,
            return_syn_parsing_predictions=False,
            return_sem_parsing_predictions=False,
            return_syn_sts_predictions=False,
            return_sem_sts_predictions=False,
            verbose=True
    ):
        parsing_loss_tensors, parsing_loss_tensor_names = self._get_parsing_loss_tensors(
            syn=return_syn_parsing_losses,
            sem=return_sem_parsing_losses
        )

        parsing_prediction_tensors, parsing_prediction_tensor_names = self._get_parsing_prediction_tensors(
            syn=return_syn_parsing_predictions,
            sem=return_sem_parsing_predictions
        )

        sts_loss_tensors, sts_loss_tensor_names = self._get_sts_loss_tensors(
            syn=return_syn_sts_losses,
            sem=return_sem_sts_losses
        )

        sts_prediction_tensors, sts_prediction_tensor_names = self._get_sts_prediction_tensors(
            syn=return_syn_sts_predictions,
            sem=return_sem_sts_predictions
        )

        info_dict = {}

        if update:
            to_run = [self.train_op]
            to_run_names = ['train_op']

            to_run += [
                self.loss
            ]
            to_run_names += [
                'loss'
            ]

            to_run += parsing_loss_tensors + parsing_prediction_tensors + sts_loss_tensors + sts_prediction_tensors
            to_run_names += parsing_loss_tensor_names + parsing_prediction_tensor_names + sts_loss_tensor_names + sts_prediction_tensor_names

            data_feed = data.get_training_data_feed(
                data_name,
                parsing=True,
                sts=True,
                minibatch_size=minibatch_size,
                randomize=randomize
            )

            info_dict = self._run_batches_inner(
                to_run,
                to_run_names,
                data_feed,
                n_minibatch,
                data_type='both',
                info_dict=info_dict,
                update=update,
                return_syn_parsing_losses=return_syn_parsing_losses,
                return_sem_sts_losses=return_sem_sts_losses,
                return_syn_parsing_predictions=return_syn_parsing_predictions,
                return_sem_parsing_predictions=return_sem_parsing_predictions,
                return_syn_sts_predictions=return_syn_sts_predictions,
                return_sem_sts_predictions=return_sem_sts_predictions,
                verbose=verbose
            )
        else:
            if return_syn_parsing_predictions or return_sem_parsing_predictions:
                if verbose:
                    stderr('Generating parses...\n')
                if minibatch_size is None:
                    minibatch_size = self.minibatch_size
                if n_minibatch is None:
                    n_minibatch_cur = data.get_n_minibatch(data_name, minibatch_size, task='parsing')
                else:
                    n_minibatch_cur = n_minibatch

                to_run = parsing_loss_tensors + parsing_prediction_tensors
                to_run_names = parsing_loss_tensor_names + parsing_prediction_tensor_names

                data_feed = data.get_parsing_data_feed(
                    data_name,
                    minibatch_size=minibatch_size,
                    randomize=randomize
                )

                info_dict = self._run_batches_inner(
                    to_run,
                    to_run_names,
                    data_feed,
                    n_minibatch_cur,
                    data_type='parsing',
                    info_dict=info_dict,
                    update=update,
                    return_syn_parsing_losses=return_syn_parsing_losses,
                    return_sem_sts_losses=False,
                    return_syn_parsing_predictions=return_syn_parsing_predictions,
                    return_sem_parsing_predictions=return_sem_parsing_predictions,
                    return_syn_sts_predictions=False,
                    return_sem_sts_predictions=False,
                    verbose=verbose
                )

            if return_syn_sts_predictions or return_sem_sts_predictions:
                if verbose:
                    stderr('Generating STS labels...\n')
                if minibatch_size is None:
                    minibatch_size = self.minibatch_size
                if n_minibatch is None:
                    n_minibatch_cur = data.get_n_minibatch(data_name, minibatch_size, task='sts')
                else:
                    n_minibatch_cur = n_minibatch

                to_run = sts_loss_tensors + sts_prediction_tensors
                to_run_names = sts_loss_tensor_names + sts_prediction_tensor_names

                data_feed = data.get_sts_data_feed(
                    data_name,
                    minibatch_size=minibatch_size,
                    randomize=randomize
                )

                info_dict = self._run_batches_inner(
                    to_run,
                    to_run_names,
                    data_feed,
                    n_minibatch_cur,
                    data_type='sts',
                    info_dict=info_dict,
                    update=update,
                    return_syn_parsing_losses=False,
                    return_sem_sts_losses=return_sem_sts_losses,
                    return_syn_parsing_predictions=False,
                    return_sem_parsing_predictions=False,
                    return_syn_sts_predictions=return_syn_sts_predictions,
                    return_sem_sts_predictions=return_sem_sts_predictions,
                    verbose=verbose
                )

        return info_dict

    def _get_parsing_loss_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            tensors += [
                self.parse_label_loss_syn,
                self.pos_label_loss_syn,
            ]
            tensor_names += [
                'parse_label_loss_syn',
                'pos_label_loss_syn',
            ]
            if self.factor_parse_labels:
                tensors.append(self.parse_depth_loss_syn)
                tensor_names.append('parse_depth_loss_syn')

        if sem:
            tensors += [
                self.parse_label_loss_sem,
                self.pos_label_loss_sem,
            ]
            tensor_names += [
                'parse_label_loss_sem',
                'pos_label_loss_sem',
            ]
            if self.factor_parse_labels:
                tensors.append(self.parse_depth_loss_sem)
                tensor_names.append('parse_depth_loss_sem')

        return tensors, tensor_names

    def _get_parsing_prediction_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            tensors += [
                self.pos_label_prediction_syn,
                self.parse_label_prediction_syn
            ]
            tensor_names += [
                'pos_label_prediction_syn',
                'parse_label_prediction_syn'
            ]
            if self.factor_parse_labels:
                tensors += [
                    self.parse_depth_prediction_syn
                ]
                tensor_names += [
                    'parse_depth_prediction_syn'
                ]
        if sem:
            tensors += [
                self.pos_label_prediction_sem,
                self.parse_label_prediction_sem
            ]
            tensor_names += [
                'pos_label_prediction_sem',
                'parse_label_prediction_sem'
            ]
            if self.factor_parse_labels:
                tensors += [
                    self.parse_depth_prediction_sem
                ]
                tensor_names += [
                    'parse_depth_prediction_sem'
                ]

        return tensors, tensor_names

    def _get_sts_loss_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get STS loss tensors and names from syntactic encoder
            tensors += [self.sts_loss_syn]
            tensor_names += ['sts_loss_syn']
        if sem:
            # Get STS loss tensors and names from semantic encoder
            tensors += [self.sts_loss_sem]
            tensor_names += ['sts_loss_sem']

        return tensors, tensor_names

    def _get_sts_prediction_tensors(self, syn=True, sem=True):
        tensors = []
        tensor_names = []
        if syn:
            # Get STS prediction tensors and names from syntactic encoder
            tensors += [self.sts_prediction_syn]
            tensor_names += ['sts_prediction_syn']
        if sem:
            # Get STS prediction tensors and names from semantic encoder
            tensors += [self.sts_prediction_sem]
            tensor_names += ['sts_prediction_sem']

        return tensors, tensor_names

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

    def update_logs(self, info_dict, name='train', task='parsing'):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if name.lower() == 'train':
                    writer = self.train_writer
                elif name.lower() == 'dev':
                    writer = self.dev_writer
                else:
                    raise ValueError('Unrecognized summary name "%s".' % name)

                fd_summary = {}

                if task.lower() == 'parsing':
                    log_summaries = self.parsing_log_summaries
                    summary = self.parsing_summary
                elif task.lower() == 'sts':
                    log_summaries = self.sts_log_summaries
                    summary = self.sts_summary
                else:
                    raise ValueError('Unrecognized task "%s".' % task)

                for k in self.parsing_log_summaries:
                    fd_summary[log_summaries[k]] = info_dict[k]

                summary_out = self.sess.run(summary, feed_dict=fd_summary)
                writer.add_summary(summary_out, self.global_batch_step.eval(session=self.sess))

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
            data,
            n_minibatch,
            n_print=5,
            run_initial_eval=False,
            verbose=True
    ):
        if self.global_batch_step.eval(session=self.sess) == 0:
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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if run_initial_eval and self.global_batch_step.eval(session=self.sess) == 0:
                    if verbose:
                        stderr('Running initial evaluation...\n')

                    info_dict_train = self._run_batches(
                        data,
                        data_name='train',
                        minibatch_size=self.eval_minibatch_size,
                        update=False,
                        randomize=False,
                        return_syn_parsing_losses=True,
                        return_sem_parsing_losses=True,
                        return_syn_parsing_predictions=False,
                        return_sem_parsing_predictions=False,
                        return_syn_sts_losses=True,
                        return_sem_sts_losses=True,
                        return_syn_sts_predictions=False,
                        return_sem_sts_predictions=False,
                        verbose=verbose
                    )

                    self.update_logs(info_dict_train, name='train', task='parsing')
                    self.update_logs(info_dict_train, name='train', task='sts')

                    info_dict_dev = self._run_batches(
                        data,
                        data_name='dev',
                        minibatch_size=self.eval_minibatch_size,
                        update=False,
                        randomize=False,
                        return_syn_parsing_losses=True,
                        return_sem_parsing_losses=True,
                        return_syn_parsing_predictions=False,
                        return_sem_parsing_predictions=False,
                        return_syn_sts_losses=True,
                        return_sem_sts_losses=True,
                        return_syn_sts_predictions=False,
                        return_sem_sts_predictions=False,
                        verbose=verbose
                    )

                    self.update_logs(info_dict_dev, name='dev', task='parsing')

                i = self.global_batch_step.eval(session=self.sess)

                while i < n_minibatch:
                    n_minibatch_cur, save, log, evaluate = self._get_n_batches(i)

                    t0_iter = time.time()
                    if verbose:
                        stderr('-' * 50 + '\n')
                        stderr('Updating on training set...\n')

                    info_dict_train = self._run_batches(
                        data,
                        data_name='train',
                        minibatch_size=self.minibatch_size,
                        n_minibatch=n_minibatch_cur,
                        update=True,
                        randomize=True,
                        return_syn_parsing_losses=True,
                        return_sem_parsing_losses=True,
                        return_syn_parsing_predictions=False,
                        return_sem_parsing_predictions=False,
                        return_syn_sts_losses=True,
                        return_sem_sts_losses=True,
                        return_syn_sts_predictions=False,
                        return_sem_sts_predictions=False,
                        verbose=verbose
                    )

                    if save:
                        self.save()

                    if log:
                        self.update_logs(info_dict_train, name='train', task='parsing')

                    if evaluate:
                        if verbose:
                            stderr('-' * 50 + '\n')
                            stderr('Dev set evaluation\n')

                        info_dict_dev = self._run_batches(
                            data,
                            data_name='dev',
                            minibatch_size=self.eval_minibatch_size,
                            update=False,
                            randomize=False,
                            return_syn_parsing_losses=True,
                            return_sem_parsing_losses=True,
                            return_syn_parsing_predictions=True,
                            return_sem_parsing_predictions=True,
                            return_syn_sts_losses=True,
                            return_sem_sts_losses=True,
                            return_syn_sts_predictions=True,
                            return_sem_sts_predictions=True,
                            verbose=verbose
                        )

                        self.update_logs(info_dict_dev, name='dev', task='parsing')

                        if verbose:
                            parse_samples = data.pretty_print_parse_predictions(
                                text=info_dict_dev['parsing_text'][:n_print],
                                pos_label_true=info_dict_dev['pos_label_true'][:n_print],
                                pos_label_pred=info_dict_dev['pos_label_prediction_syn'][:n_print],
                                parse_label_true=info_dict_dev['parse_label_true'][:n_print],
                                parse_label_pred=info_dict_dev['parse_label_prediction_syn'][:n_print],
                                parse_depth_true=info_dict_dev['parse_depth_true'][:n_print] if self.factor_parse_labels else None,
                                parse_depth_pred=info_dict_dev['parse_depth_prediction_syn'][:n_print] if self.factor_parse_labels else None,
                                mask=info_dict_dev['parsing_text_mask'][:n_print]
                            )
                            stderr('Sample parsing predictions:\n\n' + parse_samples)

                            sts_samples = data.pretty_print_sts_predictions(
                                s1=info_dict_dev['sts_s1_text'][:n_print],
                                s1_mask=info_dict_dev['sts_s1_text_mask'][:n_print],
                                s2=info_dict_dev['sts_s2_text'][:n_print],
                                s2_mask=info_dict_dev['sts_s2_text_mask'][:n_print],
                                sts_true=info_dict_dev['sts_true'][:n_print],
                                sts_pred=info_dict_dev['sts_prediction_sem'][:n_print]
                            )
                            stderr('Sample STS predictions:\n\n' + sts_samples)

                    if verbose:
                        t1_iter = time.time()
                        time_str = pretty_print_seconds(t1_iter - t0_iter)
                        stderr('Processing time: %s\n' % time_str)

                    i = self.global_batch_step.eval(session=self.sess)

    # TODO: Add STS predictions
    def predict(
            self,
            data,
            data_name='dev',
            from_syn=True,
            from_sem=True,
            verbose=True,
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                info_dict = self._run_batches(
                    data,
                    data_name=data_name,
                    minibatch_size=self.eval_minibatch_size,
                    update=False,
                    randomize=False,
                    return_syn_parsing_losses=False,
                    return_sem_parsing_losses=False,
                    return_syn_parsing_predictions=from_syn,
                    return_sem_parsing_predictions=from_sem,
                    verbose=verbose
                )
                return info_dict

    def predict_parses(
            self,
            data,
            data_name='dev',
            from_syn=True,
            from_sem=True,
            verbose=True,
    ):
        info_dict = self._run_batches(
            data,
            data_name=data_name,
            minibatch_size=self.eval_minibatch_size,
            update=False,
            randomize=False,
            return_syn_parsing_losses=False,
            return_sem_parsing_losses=False,
            return_syn_parsing_predictions=from_syn,
            return_sem_parsing_predictions=from_sem,
            verbose=verbose
        )

    def get_parse_seqs(
            self,
            data,
            info_dict
    ):
        parse_seqs = {}
        encoder = ['syn', 'sem']
        label_type = ['true', 'prediction']
        for e in encoder:
            for l in label_type:
                if 'pos_label_%s_syn' % l in info_dict:
                    if not e in parse_seqs:
                        parse_seqs[e] = {}
                    numeric_chars = info_dict['parsing_text']
                    numeric_pos = info_dict['pos_label_%s_syn' % l]
                    numeric_parse_label = info_dict['parse_label_%s_syn' % l]
                    mask = info_dict['parsing_text_mask']
                    if self.factor_parse_labels:
                        numeric_depth = info_dict['parse_depth_%s_syn' % l]
                    else:
                        numeric_depth = None

                    seqs = data.parse_predictions_to_sequences(
                        numeric_chars,
                        numeric_pos,
                        numeric_parse_label,
                        numeric_depth=numeric_depth,
                        mask=mask
                    )

        return parse_seqs
    
    def print_parse_seqs(
            self,
            data,
            data_name='dev',
            from_syn=True,
            from_sem=True,
            outdir=None,
            name=None,
            verbose=True
    ):
        if outdir is None:
            outdir = self.outdir

        seqs = self.get_parse_seqs(
            data,
            data_name=data_name,
            from_syn=from_syn,
            from_sem=from_sem,
            verbose=verbose
        )

        if from_syn:
            if name is not None:
                cur_name = name + '_syn_parse_seqs.txt'
            else:
                cur_name = 'syn_parse_seqs.txt'

            with open(outdir + '/' + cur_name, 'w') as f:
                f.write(seqs['syn'])

        if from_sem:
            if name is not None:
                cur_name = name + '_sem_parse_seqs.txt'
            else:
                cur_name = 'sem_parse_seqs.txt'

            with open(outdir + '/' + cur_name, 'w') as f:
                f.write(seqs['sem'])



